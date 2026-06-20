from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from services.gemini_service import gemini_service
from utils.file_utils import save_temp_upload, remove_file
import logging

logger = logging.getLogger("hacknviz")
router = APIRouter(tags=["AI Analysis"])

class AnalysisResponse(BaseModel):
    analysis: str
    status: str

@router.post("/analyze", response_model=AnalysisResponse)
def analyze(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    """
    Generate an AI-powered visual analysis report of the uploaded image using Gemini.
    """
    temp_path = save_temp_upload(image)
    background_tasks.add_task(remove_file, temp_path)
    
    try:
        report = gemini_service.analyze_image(str(temp_path))
        return {
            "analysis": report,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
