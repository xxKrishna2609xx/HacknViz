from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from routes.match import ComparisonResult
from services.deepface_service import deepface_service
from services.gemini_service import gemini_service
from utils.file_utils import save_temp_upload, remove_file
import logging

logger = logging.getLogger("hacknviz")
router = APIRouter(tags=["AI Analysis"])

class ComprehensiveResponse(BaseModel):
    face_match: Optional[bool] = None
    face_match_error: Optional[str] = None
    best_match: Optional[ComparisonResult] = None
    all_comparisons: Optional[List[ComparisonResult]] = None
    face_match_threshold: Optional[float] = None
    ai_analysis: Optional[str] = None
    status: str
    total_compared: int

@router.post("/comprehensive", response_model=ComprehensiveResponse)
def comprehensive(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    """
    Perform a combined face-matching and AI-powered visual reporting in a single request.
    """
    temp_path = save_temp_upload(image)
    background_tasks.add_task(remove_file, temp_path)
    
    # Initialize response structure
    results = {
        "status": "success",
        "total_compared": 0
    }
    
    # 1. Perform Face Matching
    try:
        match_results = deepface_service.match_face(str(temp_path))
        results["face_match"] = match_results["match_found"]
        results["best_match"] = match_results["best_match"]
        results["all_comparisons"] = match_results["all_comparisons"]
        results["face_match_threshold"] = match_results["threshold"]
        results["total_compared"] = match_results["total_compared"]
    except Exception as e:
        logger.error(f"Comprehensive face matching failed: {e}")
        results["face_match"] = False
        results["face_match_error"] = f"Face matching failed: {str(e)}"
        
    # 2. Perform AI Analysis
    try:
        report = gemini_service.comprehensive_analysis(str(temp_path))
        results["ai_analysis"] = report
    except Exception as e:
        logger.error(f"Comprehensive AI analysis failed: {e}")
        results["ai_analysis"] = f"AI analysis failed: {str(e)}"
        
    return results
