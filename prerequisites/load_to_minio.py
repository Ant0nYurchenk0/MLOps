import os
from pathlib import Path
import pickle
import boto3

from dotenv import load_dotenv
import gensim

load_dotenv()
# Configuration
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_USERNAME = os.getenv("MINIO_USERNAME")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USERNAME,
    aws_secret_access_key=MINIO_PASSWORD,
)

# Ensure bucket exists (create if not)
existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
if MINIO_BUCKET_NAME not in existing_buckets:
    s3.create_bucket(Bucket=MINIO_BUCKET_NAME)
    print(f"🪣 Created bucket: {MINIO_BUCKET_NAME}")

# Upload all embedding files
for file_path in EMBEDDINGS_DIR.glob("*"):
    if file_path.is_file():
        key = file_path.name
        print(f"⬆️  Uploading {key} to bucket '{MINIO_BUCKET_NAME}'...")
        s3.upload_file(Filename=str(file_path), Bucket=MINIO_BUCKET_NAME, Key=key)
        if key.endswith(".vec"):
            spell_model = gensim.models.KeyedVectors.load_word2vec_format(
                file_path, binary=False
            )
            WORDS = {w: i for i, w in enumerate(spell_model.index_to_key)}
            # Cache it
            json_path = f"{file_path}.json"
            with open(json_path, "wb") as f:
                pickle.dump(WORDS, f)
            s3.upload_file(
                Filename=str(json_path), Bucket=MINIO_BUCKET_NAME, Key=f"{key}.json"
            )


print("✅ All embeddings uploaded successfully.")
