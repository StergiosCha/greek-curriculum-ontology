from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
import logging

from app.api.routes import router
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Greek Curriculum Ontology Extractor",
    description="Extract ontologies from Greek curricula using multiple LLM providers and modes",
    version="1.0.0"
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend"""
    return FileResponse("app/static/frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Greek Curriculum Ontology Extractor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
