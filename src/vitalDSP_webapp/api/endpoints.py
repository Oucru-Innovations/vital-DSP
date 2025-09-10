# src/webapp/api/endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import psutil

router = APIRouter()


class SignalData(BaseModel):
    data: list


@router.get("/health")
async def health_check():
    """Health check endpoint for Render.com monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "vitalDSP Webapp",
        "version": "1.0.0",
        "uptime": "running",
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "disk_usage": f"{psutil.disk_usage('/').percent}%",
    }


@router.post("/process-signal")
async def process_signal(data: SignalData):
    # Call a function from vitalDSP to process the signal
    processed_data = data.data
    return {"processed_data": processed_data}
