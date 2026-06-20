from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from services.cache_service import cache_service

router = APIRouter(tags=["Health"])

class HealthResponse(BaseModel):
    status: str
    cache_stats: Dict[str, int]

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Returns the health status of the application and in-memory cache statistics.
    """
    return {
        "status": "healthy",
        "cache_stats": cache_service.get_stats()
    }
