from fastapi import APIRouter, Request, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from config import settings
from services.gemini_service import gemini_service
from utils.file_utils import save_temp_upload, remove_file
import os
import logging
import asyncio

logger = logging.getLogger("hacknviz")
router = APIRouter(tags=["AI Search"])

class SearchResult(BaseModel):
    image: str
    score: int
    reason: str

class SearchResponse(BaseModel):
    status: str
    search_query: str
    results: List[SearchResult]

@router.post("/search", response_model=SearchResponse)
async def search(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Search database records matching a description.
    Supports text query in JSON body or image query in multipart form data.
    """
    query: Optional[str] = None
    uploaded_file: Optional[UploadFile] = None
    
    content_type = request.headers.get("content-type", "")
    
    # Parse input based on Content-Type header
    if "application/json" in content_type:
        try:
            body = await request.json()
            query = body.get("query")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON body format.")
    else:
        # Fallback to form parsing (multipart/form-data)
        try:
            form_data = await request.form()
            query = form_data.get("query")
            uploaded_file = form_data.get("image")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid request parameters.")

    # If image file is provided, analyze it using Gemini to generate a text query description
    if uploaded_file and uploaded_file.filename != '':
        # We need to save the uploaded file temporarily to process it
        temp_path = save_temp_upload(uploaded_file)
        background_tasks.add_task(remove_file, temp_path)
        
        try:
            # CPU/network-bound operations offloaded to run_in_threadpool
            loop = asyncio.get_running_loop()
            query = await loop.run_in_executor(
                None, 
                gemini_service.get_image_description, 
                str(temp_path)
            )
            
            if not query:
                raise HTTPException(status_code=500, detail="Failed to describe the query image.")
        except Exception as e:
            logger.error(f"Failed to analyze query image: {e}")
            raise HTTPException(status_code=500, detail=f"Query image description failed: {str(e)}")

    if not query:
        raise HTTPException(status_code=400, detail="No search query or image provided.")

    # Get database records and descriptions
    uploads_dir = str(settings.UPLOAD_FOLDER)
    if not os.path.exists(uploads_dir):
        raise HTTPException(status_code=404, detail="Reference database folder not found.")

    reference_images = [f for f in os.listdir(uploads_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not reference_images:
        raise HTTPException(status_code=404, detail="No reference images found in the database.")

    try:
        # Generate descriptions for all database images
        records = []
        loop = asyncio.get_running_loop()
        
        # We process this synchronously in our executor to avoid blocking the async event loop
        def prepare_records():
            prepared = []
            for ref_image in reference_images:
                path = os.path.join(uploads_dir, ref_image)
                description = gemini_service.get_image_description(path)
                if description:
                    prepared.append({
                        "image": ref_image,
                        "description": description
                    })
            return prepared

        records = await loop.run_in_executor(None, prepare_records)
        
        # Perform semantic evaluation with Gemini
        matches = await loop.run_in_executor(
            None, 
            gemini_service.match_descriptions_to_query, 
            query, 
            records
        )
        
        # Format and sort results by match score (descending)
        formatted_matches = []
        for m in matches:
            try:
                score = int(m.get("score", 0))
            except:
                score = 0
            formatted_matches.append({
                "image": m.get("image", ""),
                "score": score,
                "reason": m.get("reason", "No evaluation reason provided.")
            })
            
        formatted_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return {
            "status": "success",
            "search_query": query,
            "results": formatted_matches
        }
        
    except Exception as e:
        logger.error(f"Appearance search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")
