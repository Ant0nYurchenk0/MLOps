import os
import json
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import spacy
import requests
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from spacy.cli import download as spacy_download


# Request / response schemas
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    probability: float
    label: int  # 0 or 1


# FastAPI app
app = FastAPI(
    title="Quora Moderation Prediction API",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Download and cache word_dict & lemma_dict from MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_USERNAME = os.getenv("MINIO_USERNAME", "minio")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "minio123")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USERNAME,
    aws_secret_access_key=MINIO_PASSWORD,
)


def download_json(key):
    obj = s3.get_object(Bucket=MINIO_BUCKET_NAME, Key=key)
    return json.loads(obj["Body"].read())


word_dict = download_json("word_dict.json")
lemma_dict = download_json("lemma_dict.json")


def text_to_sequence(text, maxlen):
    # Delegate to microservice
    payload = {"text": text, "max_length": maxlen, "word_dict": word_dict}
    resp = requests.post("http://preprocess:5001/sequence", json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500, detail=f"Preprocess API error: {resp.text}"
        )
    seq = resp.json()["sequence"]
    # Already padded by the microservice, so just return it as a 2D array
    return np.array([seq])


# Load latest model versions from MinIO
def load_latest_model(prefix):
    resp = s3.list_objects_v2(Bucket=MINIO_BUCKET_NAME, Prefix=prefix + "/")
    contents = resp.get("Contents", [])
    if not contents:
        raise RuntimeError(f"No models found under {prefix}/")
    latest = max(contents, key=lambda x: x["LastModified"])["Key"]
    fd, path = tempfile.mkstemp(suffix=".h5")
    with os.fdopen(fd, "wb") as f:
        s3.download_fileobj(Bucket=MINIO_BUCKET_NAME, Key=latest, Fileobj=f)
    return load_model(path)


model_gf = load_latest_model("glove_fasttext")
model_gp = load_latest_model("glove_paragram")


# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Preprocess
    maxlen = int(os.getenv("MAX_LENGTH", "100"))
    X = text_to_sequence(request.text, maxlen)

    # Ensemble predictions
    p1 = float(model_gf.predict(X)[0][0])
    p2 = float(model_gp.predict(X)[0][0])
    prob = (p1 + p2) / 2
    label = int(prob >= 0.5)

    return PredictResponse(probability=prob, label=label)
