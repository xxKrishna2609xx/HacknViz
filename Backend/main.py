import os
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routes import health, match, analyze, comprehensive, search

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hacknviz")

# Initialize FastAPI App
app = FastAPI(
    title="HacknViz API",
    description="FastAPI Backend for the AI-Powered Missing Person Finder",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate directory configurations before mounting
if not settings.STATIC_DIR.exists():
    logger.error(f"Static directory not found at {settings.STATIC_DIR}")
if not settings.TEMPLATES_DIR.exists():
    logger.error(f"Templates directory not found at {settings.TEMPLATES_DIR}")

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Register Routes / Routers
app.include_router(health.router)
app.include_router(match.router)
app.include_router(analyze.router)
app.include_router(comprehensive.router)
app.include_router(search.router)

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def home(request: Request):
    """
    Serves the main Missing Person Finder dashboard template.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Global HTTP Exception Logger / Format Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on request {request.url}: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

if __name__ == '__main__':
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG
    )
