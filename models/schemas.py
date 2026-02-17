from pydantic import BaseModel
from typing import Optional

# base response model
class MessageResponse(BaseModel):
    """simple message response model"""
    message: str


# health check response model
class HealthResponse(BaseModel):
    """health check response model"""
    status: str

