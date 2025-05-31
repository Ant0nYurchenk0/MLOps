# train_models.py

from __future__ import absolute_import, division
import json
import os
import time
import gc
import boto3
import numpy as np
import mlflow
import mlflow.keras
from datetime import datetime, timezone
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv
import requests

from build_model import build_model
from helpers import stream_embeddings
from load_data import load_data
from load_fast_text import load_fasttext
from load_glove import load_glove
from load_para import load_para
from training_config import PARAMS
from spacy.cli import download as spacy_download

load_dotenv()
spacy_download("en_core_web_lg")


def main():

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Quora_Moderation")

    # Database URI
    DB_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # Load data
    print("Loading train/test data …")
    train_texts, y_train, test_texts = load_data(DB_URI)
    num_train_data = len(train_texts)
    # Preprocess with SpaCy (build word_dict, lemma_dict, convert to sequences)
    # Call the preprocessing service to build vocab + padded sequences
    PREPROCESSOR_LINK = os.getenv("PREPROCESSOR_LINK")
    payload = {"texts": train_texts + test_texts, "max_length": PARAMS["max_length"]}
    resp = requests.post(f"{PREPROCESSOR_LINK}/build_vocab", json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Preprocess service build_vocab failed: {resp.text}")
    data = resp.json()
    word_dict = data["word_dict"]
    lemma_dict = data["lemma_dict"]
    padded_sequences = data["sequences"]
    # Split train/test back out
    train_seqs = padded_sequences[:num_train_data]
    # test_seqs = padded_sequences[num_train_data:]

    # S3 / MinIO client
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_USERNAME", "minio"),
        aws_secret_access_key=os.getenv("MINIO_PASSWORD", "minio123"),
    )
    EMBEDDINGS_BUCKET = os.getenv("EMBEDDINGS_BUCKET")
    s3.put_object(
        Bucket=EMBEDDINGS_BUCKET,
        Key="word_dict.json",
        Body=json.dumps(word_dict),
        ContentType="application/json",
    )
    print(f"Uploaded word_dict → {EMBEDDINGS_BUCKET}")

    # 2) Upload lemma_dict
    s3.put_object(
        Bucket=EMBEDDINGS_BUCKET,
        Key="lemma_dict.json",
        Body=json.dumps(lemma_dict),
        ContentType="application/json",
    )

    print(f"Uploaded lemma_dict → {EMBEDDINGS_BUCKET}")

    def load_embedding(key, loader_fn):
        resp = s3.get_object(Bucket=EMBEDDINGS_BUCKET, Key=key)
        return loader_fn(
            word_dict,
            lemma_dict,
            {w: coefs for w, coefs in stream_embeddings(resp["Body"])},
        )

    # Common training routine
    def train_and_save(emb_matrices, model_name_suffix):
        with mlflow.start_run(run_name=model_name_suffix):
            # Log hyperparameters
            mlflow.log_param("batch_size", PARAMS["batch_size"])
            mlflow.log_param("num_epoch", PARAMS["num_epoch"])
            mlflow.log_param("embedding_size", PARAMS["embedding_size"])
            mlflow.log_param("embedding_type", model_name_suffix)
            # build model
            model = build_model(
                emb_matrices,
                nb_words=len(word_dict) + 1,
                embed_size=PARAMS["embedding_size"],
            )
            # train full number of epochs
            history = model.fit(
                train_seqs,
                y_train,
                batch_size=PARAMS["batch_size"],
                epochs=PARAMS["num_epoch"],
                verbose=2,
            )
            for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                mlflow.log_metric(f"loss_epoch_{epoch_idx}", loss)

            mlflow.keras.log_model(model, artifact_path="model")
            # save
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            local_path = os.path.join(
                os.getenv("LOCAL_MODEL_PATH", "/tmp"),
                f"{model_name_suffix}_{timestamp}.h5",
            )
            model.save(local_path)
            MODEL_BUCKET = os.getenv("MODEL_BUCKET")
            existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
            if MODEL_BUCKET not in existing_buckets:
                s3.create_bucket(Bucket=MODEL_BUCKET)
                print(f"🪣 Created bucket: {MODEL_BUCKET}")
            # upload to S3
            s3.upload_file(
                local_path,
                MODEL_BUCKET,
                f"{model_name_suffix}/{os.path.basename(local_path)}",
            )
            print(f"Saved model → {model_name_suffix} at {local_path}")

            # cleanup
            K.clear_session()
            del model
            gc.collect()

    # 1) Glove + FastText
    print("Loading GloVe + FastText embeddings …")
    GLOVE_KEY = os.getenv("GLOVE_EMBED_KEY", "glove.840B.300d.txt")
    FT_KEY = os.getenv("WIKI_EMBED_KEY", "wiki-news-300d-1M.vec")

    emb_glove = load_embedding(GLOVE_KEY, load_glove)
    emb_ft = load_embedding(FT_KEY, load_fasttext)
    emb_gf = np.concatenate((emb_glove, emb_ft), axis=1)

    print("Training & saving Glove+FastText model …")
    train_and_save(emb_gf, "glove_fasttext")

    # 2) Glove + Paragram
    print("Loading GloVe + Paragram embeddings …")
    PARA_KEY = os.getenv("PARA_EMBED_KEY", "paragram_300_sl999.txt")
    emb_para = load_embedding(PARA_KEY, load_para)
    emb_gp = np.concatenate((emb_glove, emb_para), axis=1)

    print("Training & saving Glove+Paragram model …")
    train_and_save(emb_gp, "glove_paragram")
    # … entire training pipeline …
    # (load_data, preprocess, load embeddings, train, save to MinIO, etc.)


if __name__ == "__main__":
    main()
