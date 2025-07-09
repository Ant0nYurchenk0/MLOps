#!/bin/bash
# Wait for services to be ready before loading data

echo "🔄 Waiting for PostgreSQL..."
until pg_isready -h postgres -p 5432 -U mluser; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done
echo "✅ PostgreSQL is ready"

echo "🔄 Waiting for MinIO..."
until curl -s http://minio:9000 > /dev/null; do
  echo "MinIO is unavailable - sleeping"
  sleep 2
done
echo "✅ MinIO is ready"

echo "🔄 Checking if data files exist..."
if [ ! -f "/app/data/csv/train.csv" ]; then
  echo "❌ train.csv not found at /app/data/csv/train.csv"
  echo "Please ensure your data files are mounted correctly"
  exit 1
fi

if [ ! -f "/app/data/csv/test.csv" ]; then
  echo "❌ test.csv not found at /app/data/csv/test.csv"
  echo "Please ensure your data files are mounted correctly"
  exit 1
fi

echo "✅ Data files found"

echo "🔄 Creating MinIO buckets..."
python -c "
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3', 
    endpoint_url='http://minio:9000',
    aws_access_key_id='minio', 
    aws_secret_access_key='minio123'
)

buckets = ['embeddings', 'models', 'data', 'artifacts']
for bucket in buckets:
    try:
        s3.create_bucket(Bucket=bucket)
        print(f'✅ Created bucket: {bucket}')
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f'✅ Bucket {bucket} already exists')
        else:
            print(f'❌ Error creating bucket {bucket}: {e}')
"

echo "🔄 Loading data to database..."
python /app/prerequisites/load_to_db.py

echo "🔄 Loading embeddings to MinIO..."
if [ -d "/app/data/embeddings" ] && [ "$(ls -A /app/data/embeddings)" ]; then
  python /app/prerequisites/load_to_minio.py
else
  echo "⚠️  No embedding files found in /app/data/embeddings - skipping"
fi

echo "🎉 Data loading completed!"