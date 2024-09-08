# src/webapp/api/endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SignalData(BaseModel):
    data: list


@router.post("/process-signal")
async def process_signal(data: SignalData):
    # Call a function from vitalDSP to process the signal
    processed_data = data.data
    return {"processed_data": processed_data}
