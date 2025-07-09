"""
Simple Retraining Microservice
"""

from fastapi import FastAPI
import uvicorn
import os
from datetime import datetime

app = FastAPI()

@app.get("/")
def health():
    return {"status": "healthy", "service": "retraining"}

@app.post("/train")
def start_training():
    """Start training and return result"""
    try:
        from train import main
        result = main()
        return {
            "status": "success", 
            "message": "Training completed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)