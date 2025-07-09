#!/usr/bin/env python3
"""
MinIO setup script for MLOps project.
Creates all necessary buckets for the Quora moderation system.
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_USERNAME = os.getenv("MINIO_USERNAME", "minio")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "minio123")

# Required buckets for the project
REQUIRED_BUCKETS = [
    "embeddings",    # Store word embeddings (GloVe, FastText, Paragram)
    "models",        # Store trained ML models
    "data",          # Store processed datasets
    "artifacts"      # Store other ML artifacts
]

def create_minio_client():
    """Create and return MinIO client."""
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_USERNAME,
            aws_secret_access_key=MINIO_PASSWORD,
            region_name="us-east-1"  # MinIO requires a region
        )
        
        # Test connection
        s3_client.list_buckets()
        print(f"✅ Connected to MinIO at {MINIO_ENDPOINT}")
        return s3_client
        
    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")
        print(f"   Make sure MinIO is running at {MINIO_ENDPOINT}")
        print(f"   Username: {MINIO_USERNAME}")
        sys.exit(1)

def create_buckets(s3_client):
    """Create all required buckets."""
    print("\n🪣 Creating MinIO buckets...")
    
    # Get existing buckets
    try:
        response = s3_client.list_buckets()
        existing_buckets = {bucket['Name'] for bucket in response['Buckets']}
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")
        existing_buckets = set()
    
    created_count = 0
    skipped_count = 0
    
    for bucket_name in REQUIRED_BUCKETS:
        if bucket_name in existing_buckets:
            print(f"   ⏭️  Bucket '{bucket_name}' already exists")
            skipped_count += 1
        else:
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"   ✅ Created bucket: {bucket_name}")
                created_count += 1
            except ClientError as e:
                print(f"   ❌ Failed to create bucket '{bucket_name}': {e}")
    
    print(f"\n📊 Bucket creation summary:")
    print(f"   Created: {created_count}")
    print(f"   Skipped (existing): {skipped_count}")
    print(f"   Total required: {len(REQUIRED_BUCKETS)}")

def set_bucket_policies(s3_client):
    """Set appropriate bucket policies for public read access where needed."""
    print("\n🔒 Setting bucket policies...")
    
    # Public read policy for embeddings bucket (if needed for external access)
    public_read_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::embeddings/*"
            }
        ]
    }
    
    try:
        # Note: This is optional - only set if you need public access
        # s3_client.put_bucket_policy(
        #     Bucket="embeddings",
        #     Policy=json.dumps(public_read_policy)
        # )
        # print("   ✅ Set public read policy for embeddings bucket")
        print("   ℹ️  Bucket policies skipped (using default private access)")
    except Exception as e:
        print(f"   ⚠️  Could not set bucket policies: {e}")

def verify_setup(s3_client):
    """Verify that all buckets were created correctly."""
    print("\n📋 MinIO Setup Verification:")
    print(f"   Endpoint: {MINIO_ENDPOINT}")
    print(f"   Access Key: {MINIO_USERNAME}")
    
    try:
        response = s3_client.list_buckets()
        existing_buckets = {bucket['Name'] for bucket in response['Buckets']}
        
        print(f"\n📊 Buckets status:")
        all_good = True
        
        for bucket_name in REQUIRED_BUCKETS:
            if bucket_name in existing_buckets:
                # Get bucket info
                try:
                    objects_response = s3_client.list_objects_v2(Bucket=bucket_name)
                    object_count = objects_response.get('KeyCount', 0)
                    print(f"   ✅ {bucket_name} (objects: {object_count})")
                except Exception:
                    print(f"   ✅ {bucket_name} (objects: 0)")
            else:
                print(f"   ❌ {bucket_name} (missing)")
                all_good = False
        
        # List any extra buckets
        extra_buckets = existing_buckets - set(REQUIRED_BUCKETS)
        if extra_buckets:
            print(f"\n📦 Additional buckets found:")
            for bucket in extra_buckets:
                print(f"   📁 {bucket}")
        
        if all_good:
            print(f"\n🎉 All required buckets are present!")
        else:
            print(f"\n⚠️  Some buckets are missing!")
            
    except Exception as e:
        print(f"❌ Error verifying setup: {e}")

def create_sample_structure(s3_client):
    """Create sample folder structure in buckets."""
    print("\n📁 Creating sample folder structure...")
    
    sample_objects = [
        ("embeddings", "glove/README.txt", "Place GloVe embedding files here"),
        ("embeddings", "fasttext/README.txt", "Place FastText embedding files here"),
        ("embeddings", "paragram/README.txt", "Place Paragram embedding files here"),
        ("models", "glove_fasttext/README.txt", "GloVe + FastText models stored here"),
        ("models", "glove_paragram/README.txt", "GloVe + Paragram models stored here"),
        ("data", "processed/README.txt", "Processed datasets stored here"),
        ("artifacts", "logs/README.txt", "Training logs and artifacts stored here")
    ]
    
    for bucket_name, key, content in sample_objects:
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='text/plain'
            )
            print(f"   📄 Created: s3://{bucket_name}/{key}")
        except Exception as e:
            print(f"   ❌ Failed to create {bucket_name}/{key}: {e}")

def main():
    """Main setup function."""
    print("🚀 Starting MinIO Setup for MLOps Project...")
    print(f"   Target MinIO: {MINIO_ENDPOINT}")
    print(f"   Access Key: {MINIO_USERNAME}")
    print()
    
    # Step 1: Create MinIO client
    s3_client = create_minio_client()
    
    # Step 2: Create buckets
    create_buckets(s3_client)
    
    # Step 3: Set bucket policies (optional)
    set_bucket_policies(s3_client)
    
    # Step 4: Create sample folder structure
    create_sample_structure(s3_client)
    
    # Step 5: Verify setup
    verify_setup(s3_client)
    
    print("\n🎉 MinIO setup completed successfully!")
    print("\nNext steps:")
    print("1. Upload embeddings: python training/data/load_to_minio.py")
    print("2. Start training to populate models bucket")
    print("3. Access MinIO console at: http://localhost:9001")

if __name__ == "__main__":
    main()