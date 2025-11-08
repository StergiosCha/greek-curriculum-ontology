from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.core.config import LLMProvider, ExtractionMode

# Request Models
class ExtractionRequest(BaseModel):
    curriculum_filename: str
    extraction_mode: ExtractionMode
    llm_provider: LLMProvider
    api_key: str
    model_name: Optional[str] = None

class ContradictionRequest(BaseModel):
    ontology_filenames: Optional[List[str]] = None  # If None, analyze all
    llm_provider: Optional[LLMProvider] = None
    api_key: Optional[str] = None

# Response Models
class ExtractionResponse(BaseModel):
    success: bool
    message: str
    extraction_mode: ExtractionMode
    llm_provider: LLMProvider
    output_filename: str
    output_path: str
    extracted_elements: Dict[str, int]
    total_elements: int

class ContradictionResponse(BaseModel):
    success: bool
    message: str
    internal_contradictions: Dict[str, List[Dict]]
    cross_contradictions: Dict[str, List[Dict]]
    total_contradictions: int
    llm_analysis: Optional[str] = None
    analyzed_files: List[str]

class ScrapingResponse(BaseModel):
    success: bool
    message: str
    downloaded_files: List[str]
    failed_count: int
    total_found: int

class CurriculumListResponse(BaseModel):
    curricula: List[Dict[str, Any]]
    total_count: int
