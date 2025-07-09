import os
import json
import tempfile
from typing import List
from fastapi import FastAPI, HTTPException
import boto3
import spacy
import requests
import numpy as np
import psycopg2
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from spacy.cli import download as spacy_download

from models import (
    PredictRequest, PredictResponse, QuestionEntry, UpdateReadyRequest,
    ModelRegistryEntry, ModelPromotionRequest, ChampionModel
)


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
EMBEDDINGS_BUCKET = os.getenv("EMBEDDINGS_BUCKET", "embeddings")
PREPROCESSOR_LINK = os.getenv("PREPROCESSOR_LINK", "http://preprocess:5001")

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "quora_moderation")
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlpass")

DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USERNAME,
    aws_secret_access_key=MINIO_PASSWORD,
)


def download_json(key):
    try:
        obj = s3.get_object(Bucket=EMBEDDINGS_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        print(f"Warning: {key} not found in MinIO. This file is created during training.")
        return None
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return None


word_dict = download_json("word_dict.json")
lemma_dict = download_json("lemma_dict.json")


def text_to_sequence(text, maxlen):
    # Check if word_dict is available
    if word_dict is None:
        raise HTTPException(
            status_code=503, 
            detail="Word dictionary not available. Please train models first using the Airflow DAG."
        )
    
    # Delegate to microservice
    payload = {"text": text, "max_length": maxlen, "word_dict": word_dict}
    resp = requests.post(f"{PREPROCESSOR_LINK}/sequence", json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500, detail=f"Preprocess API error: {resp.text}"
        )
    seq = resp.json()["sequence"]
    # Already padded by the microservice, so just return it as a 2D array
    return np.array([seq])


# Load champion models from registry
def load_champion_model(model_type):
    """Load champion model for given type from model registry"""
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT minio_path FROM model_registry 
            WHERE model_type = %s AND status = 'champion'
            ORDER BY promoted_at DESC LIMIT 1
        """, (model_type,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            # Fallback to latest model if no champion exists
            print(f"No champion found for {model_type}, falling back to latest")
            return load_latest_model_fallback(model_type)
        
        minio_path = result[0]
        fd, local_path = tempfile.mkstemp(suffix=".h5")
        with os.fdopen(fd, "wb") as f:
            s3.download_fileobj(Bucket=MINIO_BUCKET_NAME, Key=minio_path, Fileobj=f)
        
        return load_model(local_path)
        
    except Exception as e:
        print(f"Error loading champion model for {model_type}: {e}")
        return load_latest_model_fallback(model_type)


def load_latest_model_fallback(model_type):
    """Fallback method to load latest model by timestamp"""
    prefix = model_type.replace("_", "")  # glove_fasttext -> glovefasttext
    resp = s3.list_objects_v2(Bucket=MINIO_BUCKET_NAME, Prefix=f"{prefix}/")
    contents = resp.get("Contents", [])
    if not contents:
        print(f"No models found under {prefix}/ - models need to be trained first")
        return None
    latest = max(contents, key=lambda x: x["LastModified"])["Key"]
    fd, path = tempfile.mkstemp(suffix=".h5")
    with os.fdopen(fd, "wb") as f:
        s3.download_fileobj(Bucket=MINIO_BUCKET_NAME, Key=latest, Fileobj=f)
    return load_model(path)


def reload_models():
    """Reload champion models - called after promotions"""
    global model_gf, model_gp
    model_gf = load_champion_model("glove_fasttext")
    model_gp = load_champion_model("glove_paragram")
    print(f"Champion models reloaded - GF: {'✅' if model_gf else '❌'}, GP: {'✅' if model_gp else '❌'}")


# Initial model loading
model_gf = load_champion_model("glove_fasttext")
model_gp = load_champion_model("glove_paragram")
print(f"Model loading complete - GF: {'✅' if model_gf else '❌'}, GP: {'✅' if model_gp else '❌'}")


# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Check if models are loaded
    if model_gf is None or model_gp is None:
        raise HTTPException(
            status_code=503, 
            detail="Models not available. Please train models first using the Airflow DAG."
        )
    
    # Preprocess
    maxlen = 55
    X = text_to_sequence(request.text, maxlen)

    # Ensemble predictions
    p1 = float(model_gf.predict(X)[0][0])
    p2 = float(model_gp.predict(X)[0][0])
    prob = (p1 + p2) / 2
    label = int(prob >= 0.5)

    return PredictResponse(probability=prob, label=label)


@app.get("/not-ready", response_model=List[QuestionEntry])
def get_not_ready_entries():
    """Get all database entries that are not ready to use"""
    try:
        conn = psycopg2.connect(DB_URI)
        df = pd.read_sql(
            "SELECT qid, question_text, target, prediction, ready_to_use FROM questions WHERE ready_to_use = FALSE",
            conn
        )
        conn.close()
        
        return [
            QuestionEntry(
                qid=row["qid"],
                question_text=row["question_text"],
                target=row["target"],
                prediction=row["prediction"],
                ready_to_use=row["ready_to_use"]
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/update-ready")
def update_ready_status(request: UpdateReadyRequest):
    """Update one or more entries to be ready to use by ID"""
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()
        
        # Handle single ID or list of IDs
        ids = request.ids if isinstance(request.ids, list) else [request.ids]
        
        # Update ready_to_use status
        cursor.execute(
            "UPDATE questions SET ready_to_use = TRUE WHERE qid = ANY(%s)",
            (ids,)
        )
        
        updated_count = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"message": f"Updated {updated_count} entries to ready status", "updated_ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# Model Registry Endpoints

@app.post("/models/register", response_model=ModelRegistryEntry)
def register_model(model_entry: ModelRegistryEntry):
    """Register a new model in the registry"""
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_registry 
            (model_id, model_type, mlflow_run_id, minio_path, status, performance_metric)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
        """, (
            model_entry.model_id,
            model_entry.model_type,
            model_entry.mlflow_run_id,
            model_entry.minio_path,
            model_entry.status,
            model_entry.performance_metric
        ))
        
        result = cursor.fetchone()
        model_entry.id = result[0]
        model_entry.created_at = result[1]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return model_entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/models/champions", response_model=List[ChampionModel])
def get_champion_models():
    """Get current champion models for all model types"""
    try:
        conn = psycopg2.connect(DB_URI)
        df = pd.read_sql("""
            SELECT model_type, model_id, minio_path, mlflow_run_id, 
                   performance_metric, promoted_at
            FROM model_registry 
            WHERE status = 'champion'
            ORDER BY model_type
        """, conn)
        conn.close()
        
        return [
            ChampionModel(
                model_type=row["model_type"],
                model_id=row["model_id"],
                minio_path=row["minio_path"],
                mlflow_run_id=row["mlflow_run_id"],
                performance_metric=row["performance_metric"],
                promoted_at=row["promoted_at"]
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/models/challengers", response_model=List[ModelRegistryEntry])
def get_challenger_models():
    """Get current challenger models"""
    try:
        conn = psycopg2.connect(DB_URI)
        df = pd.read_sql("""
            SELECT id, model_id, model_type, mlflow_run_id, minio_path, 
                   status, performance_metric, created_at, promoted_at, retired_at
            FROM model_registry 
            WHERE status = 'challenger'
            ORDER BY created_at DESC
        """, conn)
        conn.close()
        
        return [
            ModelRegistryEntry(
                id=row["id"],
                model_id=row["model_id"],
                model_type=row["model_type"],
                mlflow_run_id=row["mlflow_run_id"],
                minio_path=row["minio_path"],
                status=row["status"],
                performance_metric=row["performance_metric"],
                created_at=row["created_at"],
                promoted_at=row["promoted_at"],
                retired_at=row["retired_at"]
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/models/promote")
def promote_challenger(request: ModelPromotionRequest):
    """Promote a challenger model to champion"""
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()
        
        # Get challenger info
        cursor.execute("""
            SELECT model_type, performance_metric 
            FROM model_registry 
            WHERE model_id = %s AND status = 'challenger'
        """, (request.challenger_model_id,))
        
        challenger_info = cursor.fetchone()
        if not challenger_info:
            raise HTTPException(status_code=404, detail="Challenger model not found")
        
        model_type, performance_metric = challenger_info
        
        # Retire current champion
        cursor.execute("""
            UPDATE model_registry 
            SET status = 'retired', retired_at = CURRENT_TIMESTAMP
            WHERE model_type = %s AND status = 'champion'
        """, (model_type,))
        
        # Promote challenger to champion
        cursor.execute("""
            UPDATE model_registry 
            SET status = 'champion', promoted_at = CURRENT_TIMESTAMP
            WHERE model_id = %s
        """, (request.challenger_model_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Reload models to use new champion
        reload_models()
        
        return {
            "message": f"Model {request.challenger_model_id} promoted to champion",
            "model_type": model_type,
            "performance_metric": performance_metric,
            "reason": request.reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/models/reload")
def reload_champion_models():
    """Manually reload champion models"""
    try:
        reload_models()
        return {"message": "Champion models reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")
