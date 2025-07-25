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
from dotenv import load_dotenv
import requests
import psycopg2
import logging
import sys

from build_model import build_model
from helpers import stream_embeddings
from load_data import load_data
from load_fast_text import load_fasttext
from load_glove import load_glove
from load_para import load_para

load_dotenv()

# Configure logging to show in Airflow
logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Force stdout/stderr to be unbuffered

def register_model_in_registry(model_id, model_type, mlflow_run_id, minio_path, performance_metric, db_uri):
    """Register new model in the registry as challenger"""
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_registry 
            (model_id, model_type, mlflow_run_id, minio_path, status, performance_metric)
            VALUES (%s, %s, %s, %s, 'challenger', %s)
        """, (model_id, model_type, mlflow_run_id, minio_path, performance_metric))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Registered model {model_id} as challenger in registry")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to register model in registry: {e}")
        return False


def get_current_champion_metric(model_type, db_uri):
    """Get performance metric of current champion model"""
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT performance_metric, model_id 
            FROM model_registry 
            WHERE model_type = %s AND status = 'champion'
            ORDER BY promoted_at DESC LIMIT 1
        """, (model_type,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return result[0], result[1]  # metric, model_id
        return None, None
        
    except Exception as e:
        logger.error(f"❌ Error getting champion metric: {e}")
        return None, None


def auto_promote_if_better(model_id, model_type, new_metric, db_uri):
    """Automatically promote challenger if it's better than champion"""
    try:
        champion_metric, champion_model_id = get_current_champion_metric(model_type, db_uri)
        
        if champion_metric is None:
            # No champion exists, promote first model
            promote_challenger(model_id, db_uri)
            logger.info(f"🏆 Promoted {model_id} as first champion for {model_type}")
            return True
            
        # Lower loss is better
        if new_metric < champion_metric:
            promote_challenger(model_id, db_uri)
            logger.info(f"🏆 Promoted {model_id} (loss: {new_metric:.4f}) over {champion_model_id} (loss: {champion_metric:.4f})")
            return True
        else:
            logger.info(f"📊 New model {model_id} (loss: {new_metric:.4f}) not better than champion {champion_model_id} (loss: {champion_metric:.4f})")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in auto-promotion: {e}")
        return False


def promote_challenger(model_id, db_uri):
    """Promote challenger to champion"""
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        # Get model type
        cursor.execute("SELECT model_type FROM model_registry WHERE model_id = %s", (model_id,))
        model_type = cursor.fetchone()[0]
        
        # Retire current champion
        cursor.execute("""
            UPDATE model_registry 
            SET status = 'retired', retired_at = CURRENT_TIMESTAMP
            WHERE model_type = %s AND status = 'champion'
        """, (model_type,))
        
        # Promote challenger
        cursor.execute("""
            UPDATE model_registry 
            SET status = 'champion', promoted_at = CURRENT_TIMESTAMP
            WHERE model_id = %s
        """, (model_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error promoting model: {e}")
        return False


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
    logger.info("Loading train/test data …")
    train_texts, y_train, test_texts = load_data(DB_URI)
    num_train_data = len(train_texts)
    # Preprocess with SpaCy (build word_dict, lemma_dict, convert to sequences)
    # Call the preprocessing service to build vocab + padded sequences
    PREPROCESSOR_LINK = os.getenv("PREPROCESSOR_LINK")
    payload = {"texts": train_texts + test_texts, "max_length": 55}
    resp = requests.post(
        f"{PREPROCESSOR_LINK}/build_vocab", json=payload, timeout=60 * 60
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Preprocess service build_vocab failed: {resp.text}")
    data = resp.json()
    word_dict = data["word_dict"]
    lemma_dict = data["lemma_dict"]
    padded_sequences = data["sequences"]
    # Split train/test back out
    train_seqs = np.array(padded_sequences[:num_train_data])
    # test_seqs = padded_sequences[num_train_data:]

    # S3 / MinIO client
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_USERNAME", "minio"),
        aws_secret_access_key=os.getenv("MINIO_PASSWORD", "minio123"),
    )
    EMBEDDINGS_BUCKET = os.getenv("MINIO_BUCKET_NAME")

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
            # Auto-detect embedding size from the loaded matrices
            embedding_size = emb_matrices.shape[1]
            epochs = 2  # Reduced epochs to save memory
            batch_size = 256  # Reduced batch size for memory efficiency
            # Log hyperparameters
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epoch", epochs)
            mlflow.log_param("embedding_size", embedding_size)
            mlflow.log_param("embedding_type", model_name_suffix)
            # build model
            max_length = train_seqs.shape[1]
            model = build_model(
                emb_matrices,
                nb_words=len(word_dict) + 1,
                embedding_size=embedding_size,
                max_length=max_length,
            )
            # train full number of epochs
            history = model.fit(
                train_seqs,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,  # Add verbose to monitor progress
            )
            for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                mlflow.log_metric(f"loss_epoch_{epoch_idx}", loss)

            mlflow.keras.log_model(model, artifact_path="model")
            
            # Get final training loss for champion-challenger comparison
            final_loss = history.history.get("loss", [])[-1]
            mlflow.log_metric("final_loss", final_loss)
            
            # save
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            model_id = f"{model_name_suffix}_{timestamp}"
            local_path = os.path.join(
                os.getenv("LOCAL_MODEL_PATH", "/tmp"),
                f"{model_id}.h5",
            )
            model.save(local_path)
            MODEL_BUCKET = os.getenv("MODEL_BUCKET")
            existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
            if MODEL_BUCKET not in existing_buckets:
                s3.create_bucket(Bucket=MODEL_BUCKET)
                logger.info(f"🪣 Created bucket: {MODEL_BUCKET}")
            
            # upload to S3
            minio_path = f"{model_name_suffix}/{os.path.basename(local_path)}"
            s3.upload_file(local_path, MODEL_BUCKET, minio_path)
            logger.info(f"Saved model → {model_name_suffix} at {local_path}")
            
            # Register model in registry
            mlflow_run_id = mlflow.active_run().info.run_id
            register_model_in_registry(
                model_id=model_id,
                model_type=model_name_suffix,
                mlflow_run_id=mlflow_run_id,
                minio_path=minio_path,
                performance_metric=final_loss,
                db_uri=DB_URI
            )
            
            # Auto-promote if better than current champion
            auto_promote_if_better(
                model_id=model_id,
                model_type=model_name_suffix,
                new_metric=final_loss,
                db_uri=DB_URI
            )

            # Enhanced cleanup
            K.clear_session()
            del model, history
            gc.collect()
            
            # Force garbage collection between models
            import psutil
            import os
            process = psutil.Process(os.getpid())
            logger.info(f"Memory usage after {model_name_suffix}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 1) Glove + FastText
    logger.info("Loading GloVe + FastText embeddings …")
    GLOVE_KEY = os.getenv("GLOVE_EMBED_KEY")
    FT_KEY = os.getenv("WIKI_EMBED_KEY")

    emb_glove, _ = load_embedding(GLOVE_KEY, load_glove)
    emb_ft, _ = load_embedding(FT_KEY, load_fasttext)
    emb_gf = np.concatenate((emb_glove, emb_ft), axis=1)

    logger.info("Training & saving Glove+FastText model …")
    train_and_save(emb_gf, "glove_fasttext")
    
    # Explicit cleanup after first model
    del emb_gf, emb_ft
    gc.collect()

    # 2) Glove + Paragram
    logger.info("Loading GloVe + Paragram embeddings …")
    PARA_KEY = os.getenv("PARA_EMBED_KEY")
    emb_para, _ = load_embedding(PARA_KEY, load_para)
    emb_gp = np.concatenate((emb_glove, emb_para), axis=1)

    logger.info("Training & saving Glove+Paragram model …")
    train_and_save(emb_gp, "glove_paragram")
    
    # Final cleanup
    del emb_gp, emb_para, emb_glove
    gc.collect()
    
    logger.info("Training pipeline completed successfully!")
    # … entire training pipeline …
    # (load_data, preprocess, load embeddings, train, save to MinIO, etc.)


if __name__ == "__main__":
    main()
