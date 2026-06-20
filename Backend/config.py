import os
from pydantic_settings import BaseSettings
from pathlib import Path

# Base Directory of the Backend
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    
    # Folders inside Backend
    UPLOAD_FOLDER: Path = BASE_DIR / "uploads"
    TEMP_FOLDER: Path = BASE_DIR / "temp_uploads"
    
    # Decoupled Frontend Folders
    STATIC_DIR: Path = BASE_DIR.parent / "Frontend" / "static"
    TEMPLATES_DIR: Path = BASE_DIR.parent / "Frontend" / "templates"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    
    # Face matching thresholds
    MATCH_THRESHOLD: float = 0.6
    
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure critical folders exist
settings.UPLOAD_FOLDER.mkdir(exist_ok=True)
settings.TEMP_FOLDER.mkdir(exist_ok=True)
