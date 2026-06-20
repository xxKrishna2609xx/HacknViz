import os
import uuid
import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException
from config import settings

logger = logging.getLogger("hacknviz")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_temp_upload(upload_file: UploadFile) -> Path:
    """
    Saves an uploaded file to the temporary directory with a unique UUID name
    to prevent file name collisions.
    """
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
        
    if not is_allowed_file(upload_file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    ext = upload_file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{ext}"
    filepath = settings.TEMP_FOLDER / unique_filename
    
    try:
        with open(filepath, "wb") as buffer:
            # Read contents in chunks to handle larger files efficiently
            while content := upload_file.file.read(1024 * 1024):
                buffer.write(content)
        return filepath
    except Exception as e:
        logger.error(f"Error saving temp upload: {e}")
        # Clean up in case of failure
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

def save_bytes_to_temp(image_bytes: bytes, original_filename: str = "capture.jpg") -> Path:
    """
    Saves binary image bytes (e.g. from webcam capture) to the temporary directory.
    """
    if not is_allowed_file(original_filename):
        raise HTTPException(status_code=400, detail="Invalid captured file extension.")
        
    ext = original_filename.rsplit('.', 1)[1].lower()
    unique_filename = f"capture_{uuid.uuid4()}.{ext}"
    filepath = settings.TEMP_FOLDER / unique_filename
    
    try:
        with open(filepath, "wb") as buffer:
            buffer.write(image_bytes)
        return filepath
    except Exception as e:
        logger.error(f"Error saving captured bytes: {e}")
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Failed to save captured image: {str(e)}")

def remove_file(filepath: str | Path) -> None:
    """Utility function to safely remove a file (ideal for FastAPI BackgroundTasks)."""
    try:
        path = Path(filepath)
        if path.exists():
            os.remove(path)
            logger.debug(f"Successfully cleaned up file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete file {filepath}: {e}")
