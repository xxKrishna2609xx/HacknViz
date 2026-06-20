from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.deepface_service import deepface_service
from utils.file_utils import save_temp_upload, remove_file
import logging

logger = logging.getLogger("hacknviz")
router = APIRouter(tags=["Biometrics"])

class ComparisonResult(BaseModel):
    image: str
    match_found: bool
    distance: Optional[float] = None
    confidence: float
    threshold: Optional[float] = None
    error: Optional[str] = None

class MatchResponse(BaseModel):
    match_found: bool
    total_compared: int
    best_match: Optional[ComparisonResult] = None
    all_comparisons: List[ComparisonResult]
    threshold: float

@router.post("/match", response_model=MatchResponse)
def match(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    """
    Match an uploaded face against all reference images in the database.
    Calculates embeddings and cosine distance using VGG-Face.
    """
    # Save upload to a unique temporary file
    temp_path = save_temp_upload(image)
    
    # Register the file for removal after sending response
    background_tasks.add_task(remove_file, temp_path)
    
    try:
        results = deepface_service.match_face(str(temp_path))
        return results
    except FileNotFoundError as e:
        logger.error(f"Reference folder error: {e}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": str(e),
                "match_found": False,
                "total_compared": 0
            }
        )
    except Exception as e:
        logger.error(f"Face matching process failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
