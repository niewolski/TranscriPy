from fastapi import APIRouter
from models.schemas import MessageResponse

# create router for api endpoints
router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/", response_model=MessageResponse)
async def api_root():
    """api root endpoint"""
    return {"message": "transcripy api v1"}

