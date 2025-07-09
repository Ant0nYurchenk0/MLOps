-- MLOps Database Initialization Script
-- This script creates all necessary tables for the Quora moderation system

-- Create questions table (training data)
CREATE TABLE IF NOT EXISTS questions (
    qid VARCHAR(50) PRIMARY KEY,
    question_text TEXT NOT NULL,
    target INTEGER NOT NULL CHECK (target IN (0, 1)),
    prediction FLOAT DEFAULT NULL,
    ready_to_use BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create questions_test table (test data)
CREATE TABLE IF NOT EXISTS questions_test (
    qid VARCHAR(50) PRIMARY KEY,
    question_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model_registry table (champion-challenger tracking)
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('glove_fasttext', 'glove_paragram')),
    mlflow_run_id VARCHAR(100) NOT NULL,
    minio_path VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('champion', 'challenger', 'retired')) DEFAULT 'challenger',
    performance_metric FLOAT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    promoted_at TIMESTAMP DEFAULT NULL,
    retired_at TIMESTAMP DEFAULT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_questions_ready_to_use 
ON questions(ready_to_use);

CREATE INDEX IF NOT EXISTS idx_model_registry_status_type 
ON model_registry(status, model_type);

CREATE INDEX IF NOT EXISTS idx_model_registry_created_at 
ON model_registry(created_at DESC);

-- Create unique constraint to ensure only one champion per model type
CREATE UNIQUE INDEX IF NOT EXISTS unique_champion_per_type 
ON model_registry(model_type) 
WHERE status = 'champion';