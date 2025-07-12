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
    import logging
    import traceback
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting training process...")
        from train import main
        result = main()
        logger.info("✅ Training completed successfully")
        return {
            "status": "success", 
            "message": "Training completed",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        logger.error(f"❌ Training failed: {error_msg}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error(f"❌ Full traceback:\n{error_traceback}")
        
        return {
            "status": "error",
            "message": error_msg,
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)