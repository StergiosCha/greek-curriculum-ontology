from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, PlainTextResponse
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import asyncio
import tempfile
import shutil

from pydantic import BaseModel

from app.api.models import (
    ExtractionRequest, ExtractionResponse,
    ContradictionRequest, ContradictionResponse,
    ScrapingResponse, CurriculumListResponse
)
from app.services.scraper import scrape_greek_curricula
from app.services.enhanced_curriculum_extractor import EnhancedCurriculumOntologyExtractor
from app.services.contradiction_detector import ContradictionDetector
from app.services.rag_contradiction_detector import create_rag_contradiction_detector
from app.services.llm_service import LLMProvider
from app.core.config import ExtractionMode, settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global services (could be improved with dependency injection)
ontology_extractor = EnhancedCurriculumOntologyExtractor()
contradiction_detector = ContradictionDetector()


@router.post("/scrape", response_model=ScrapingResponse)
async def scrape_curricula():
    """Scrape Greek curricula from ebooks.edu.gr"""
    try:
        logger.info("Starting curriculum scraping...")
        results = await scrape_greek_curricula()
        
        return ScrapingResponse(
            success=True,
            message=f"Successfully scraped {results['total_found']} curricula",
            downloaded_files=results['downloaded'],
            failed_count=results['failed'],
            total_found=results['total_found']
        )
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/curricula")
async def list_local_curricula():
    """List available curricula in data/curricula folder"""
    try:
        curricula_dir = Path("data/curricula")
        if not curricula_dir.exists():
            return []
        
        curricula = []
        for pdf_file in curricula_dir.glob("*.pdf"):
            # Classify curriculum type from filename
            name = pdf_file.name.lower()
            if 'δημοτικ' in name or 'primary' in name:
                level = 'primary'
            elif 'γυμνάσι' in name or 'secondary_lower' in name:
                level = 'secondary_lower'
            elif 'λύκει' in name or 'secondary_upper' in name:
                level = 'secondary_upper'
            else:
                level = 'unknown'
            
            curricula.append({
                'filename': pdf_file.name,
                'path': str(pdf_file),
                'size': pdf_file.stat().st_size,
                'size_mb': round(pdf_file.stat().st_size / (1024*1024), 2),
                'education_level': level,
                'subject': 'greek_language' if any(term in name for term in ['ελληνικ', 'γλώσσα']) else 'other'
            })
        
        return curricula
        
    except Exception as e:
        logger.error(f"Error listing curricula: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract")
async def extract_ontology_upload(
    files: List[UploadFile] = File(...),
    extraction_mode: str = Form(...),
    llm_provider: str = Form(...),
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None)
):
    """Extract ontology from uploaded curriculum files"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        extraction_mode_enum = ExtractionMode(extraction_mode)
        llm_provider_enum = LLMProvider(llm_provider)

        if api_key or llm_provider_enum == LLMProvider.OLLAMA:
            ontology_extractor.setup_llm(llm_provider_enum, api_key or "", model_name)
        else:
            raise HTTPException(status_code=400, detail="API key required for non-Ollama providers")
        
        results = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = Path(temp_file.name)
            try:
                result = ontology_extractor.extract_from_pdf(temp_path, extraction_mode_enum, llm_provider_enum)
                output_dir = Path("data/outputs"); output_dir.mkdir(parents=True, exist_ok=True)
                output_filename = f"{Path(file.filename).stem}_{extraction_mode}_{llm_provider}.ttl"
                output_path = output_dir / output_filename
                ontology_extractor.save_ontology(result, output_path)

                # Patch: add entities / concepts / relations 
                json_graph = result.get("json_graph", {})
                entities = [n for n in json_graph.get("nodes", []) if n.get("category") == "entity"]
                concepts = [n for n in json_graph.get("nodes", []) if n.get("category") == "concept"]
                relations = json_graph.get("links", [])
                
                result.update({
                    'success': True,
                    'message': f"Ontology extracted successfully from {file.filename}",
                    'output_filename': output_filename,
                    'output_path': str(output_path),
                    'total_elements': sum(result.get('extracted_elements', {}).values()),
                    'json_graph': json_graph,
                    'entities': entities,
                    'concepts': concepts,
                    'relations': relations
                })
                results.append(result)
            finally:
                temp_path.unlink(missing_ok=True)
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid PDF files processed")
        return results[0] if len(results) == 1 else {'success': True, 'message': f"Processed {len(results)} files", 'results': results}
    except Exception as e:
        logger.error(f"Upload extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-local")
async def extract_from_local_curriculum(
    local_curriculum: str = Form(...),
    extraction_mode: str = Form(...),
    llm_provider: str = Form(...),
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None)
):
    """Extract ontology from local curriculum file with research paper alignment"""
    try:
        logger.info(f"Starting enhanced local extraction:")
        logger.info(f"  - File: {local_curriculum}")
        logger.info(f"  - Mode: {extraction_mode}")
        logger.info(f"  - Provider: {llm_provider}")
        logger.info(f"  - API Key provided: {'Yes' if api_key else 'No'}")
        logger.info(f"  - Model: {model_name or 'Default'}")

        # Normalize enums to lowercase
        try:
            extraction_mode_enum = ExtractionMode(extraction_mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid extraction_mode: {extraction_mode}"
            )

        try:
            llm_provider_enum = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid llm_provider: {llm_provider}"
            )

        # Find curriculum file
        curricula_dir = Path("data/curricula")
        pdf_path = curricula_dir / local_curriculum

        if not pdf_path.exists():
            available_files = [f.name for f in curricula_dir.glob("*.pdf")]
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {local_curriculum}. Available: {available_files}"
            )

        # Setup LLM
        if api_key or llm_provider_enum.value == "ollama":
            ontology_extractor.setup_llm(llm_provider_enum, api_key or "", model_name)
        else:
            raise HTTPException(
                status_code=400,
                detail="API key required for non-Ollama providers"
            )

        # Extract ontology with research alignment
        result = ontology_extractor.extract_from_pdf(
            pdf_path,
            extraction_mode_enum,
            llm_provider_enum,
            research_aligned=True  # This enables the enhanced extraction
        )

        # Save result
        output_dir = Path("data/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{pdf_path.stem}_{extraction_mode}_{llm_provider}_enhanced.ttl"
        output_path = output_dir / output_filename
        
        # Save the research-compliant ontology
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.get('turtle_output', ''))

        return {
            "success": True,
            "message": f"Enhanced extraction completed from {local_curriculum}",
            "filename": local_curriculum,
            "extraction_mode": extraction_mode,
            "llm_provider": llm_provider,
            "output_filename": output_filename,
            "turtle_output": result.get("turtle_output", ""),
            "extracted_elements": result.get("extracted_elements", {}),
            "total_elements": sum(result.get("extracted_elements", {}).values()),
            "json_graph": result.get("json_graph", {}),
            "entities": [n for n in result.get("json_graph", {}).get("nodes", []) if n.get("category") == "entity"],
            "concepts": [n for n in result.get("json_graph", {}).get("nodes", []) if n.get("category") == "concept"],
            "relations": result.get("json_graph", {}).get("links", []),
            
            # Enhanced research alignment results
            "research_aligned": result.get("research_aligned", True),
            "validation_results": result.get("validation_results", {}),
            "competency_scores": result.get("validation_results", {}).get("competency_scores", {}),
            "axiom_violations": result.get("validation_results", {}).get("axiom_violations", []),
            "overall_quality_score": result.get("validation_results", {}).get("overall_score", 0.0)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced extraction failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Enhanced extraction failed: {str(e)}")

# Rest of your routes continue here...
@router.post("/analyze-internal-contradictions")
async def analyze_internal_contradictions(
    curriculum: str = Form(...),
    llm_provider: str = Form(...),
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None)
):
    """Analyze internal contradictions within a single curriculum ontology"""
    try:
        logger.info(f"Starting internal contradiction analysis for: {curriculum}")
        
        # Validate LLM provider
        try:
            llm_provider_enum = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid llm_provider: {llm_provider}")
        
        # Setup LLM
        if api_key or llm_provider_enum.value == "ollama":
            contradiction_detector.setup_llm(llm_provider_enum, api_key or "")
        else:
            raise HTTPException(status_code=400, detail="API key required for non-Ollama providers")
        
        # Find the latest ontology file for this curriculum
        outputs_dir = Path("data/outputs")
        ontology_files = list(outputs_dir.glob(f"{Path(curriculum).stem}_*.ttl"))
        
        if not ontology_files:
            # Try to extract ontology first
            curricula_dir = Path("data/curricula")
            pdf_path = curricula_dir / curriculum
            
            if not pdf_path.exists():
                raise HTTPException(status_code=404, detail=f"Curriculum file not found: {curriculum}")
            
            raise HTTPException(
                status_code=400, 
                detail=f"No ontology found for {curriculum}. Please extract ontology first."
            )
        
        # Use the most recent ontology file
        ontology_path = max(ontology_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using ontology file: {ontology_path.name}")
        
        # Analyze contradictions
        result = contradiction_detector.detect_internal_contradictions(ontology_path, llm_provider_enum)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal contradiction analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-cross-contradictions")
async def analyze_cross_contradictions(
    curricula: List[str] = Form(...),
    llm_provider: str = Form(...),
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None)
):
    """Analyze contradictions between multiple curricula"""
    try:
        logger.info(f"Starting cross-curriculum analysis for: {curricula}")
        
        if len(curricula) < 2:
            raise HTTPException(status_code=400, detail="At least 2 curricula required for cross-analysis")
        
        # Validate LLM provider
        try:
            llm_provider_enum = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid llm_provider: {llm_provider}")
        
        # Setup LLM
        if api_key or llm_provider_enum.value == "ollama":
            contradiction_detector.setup_llm(llm_provider_enum, api_key or "")
        else:
            raise HTTPException(status_code=400, detail="API key required for non-Ollama providers")
        
        # Find ontology files for each curriculum
        outputs_dir = Path("data/outputs")
        ontology_paths = []
        
        for curriculum in curricula:
            curriculum_files = list(outputs_dir.glob(f"{Path(curriculum).stem}_*.ttl"))
            if not curriculum_files:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No ontology found for {curriculum}. Please extract ontology first."
                )
            # Use the most recent file
            latest_file = max(curriculum_files, key=lambda p: p.stat().st_mtime)
            ontology_paths.append(latest_file)
            logger.info(f"Using ontology: {latest_file.name} for {curriculum}")
        
        # Analyze cross-curriculum contradictions
        result = contradiction_detector.detect_cross_curriculum_contradictions(ontology_paths, llm_provider_enum)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cross-curriculum analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_latest_ttl_file():
    """Get the most recent TTL file from data/outputs"""
    outputs_dir = Path("data/outputs")
    
    if not outputs_dir.exists():
        return None
    
    ttl_files = list(outputs_dir.glob("*.ttl"))
    if not ttl_files:
        return None
    
    # Get the most recent TTL file
    latest_file = max(ttl_files, key=os.path.getctime)
    return latest_file

def load_ontology_graph(ttl_file_path: Path):
    """Load TTL file into RDF graph"""
    try:
        # Force fresh read
        from rdflib import Graph
        g = Graph()
        
        # Read file content first
        with open(ttl_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse from string instead of file
        g.parse(data=content, format="turtle")
        return g
    except Exception as e:
        logger.error(f"Failed to load TTL ontology {ttl_file_path}: {e}")
        return None

@router.post("/api/query-ontology")
async def query_ontology(query_request: dict):
    """Query the TTL ontology files"""
    
    query_type = query_request.get('query_type')
    params = query_request.get('params', {})
    
    # Load latest TTL file
    ttl_file = get_latest_ttl_file()
    if not ttl_file:
        return {'error': 'No TTL files found in data/outputs', 'results': []}
    
    g = load_ontology_graph(ttl_file)
    if not g:
        return {'error': f'Failed to load {ttl_file.name}', 'results': []}
    
    try:
        if query_type == "modules_by_topic":
            topic = params.get('topic', '').lower()
            
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT DISTINCT ?module ?title ?description ?topicDesc WHERE {{
                ?module a currkg:Module ;
                        currkg:hasTitle ?title ;
                        currkg:coversTopic ?topic .
                ?topic currkg:hasDescription ?topicDesc .
                OPTIONAL {{ ?module currkg:hasDescription ?description }}
                FILTER(CONTAINS(LCASE(?topicDesc), "{topic}"))
            }}
            """
            
            results = []
            for row in g.query(query):
                results.append({
                    'title': str(row[1]),
                    'description': str(row[2]) if row[2] else '',
                    'matching_topic': str(row[3])
                })
                
        elif query_type == "all_modules":
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?title ?description WHERE {
                ?module a currkg:Module ;
                        currkg:hasTitle ?title .
                OPTIONAL { ?module currkg:hasDescription ?description }
            }
            ORDER BY ?title
            """
            
            results = []
            for row in g.query(query):
                results.append({
                    'title': str(row[0]),
                    'description': str(row[1]) if row[1] else ''
                })
                
        elif query_type == "learning_path_steps":
            persona = params.get('persona', '')
            
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?step ?description ?module WHERE {{
                ?persona currkg:hasType ?personaType ;
                         currkg:determines ?path .
                ?path currkg:hasLearningStep ?step .
                ?step currkg:hasDescription ?description .
                OPTIONAL {{ ?step currkg:refersTo ?module }}
                FILTER(CONTAINS(LCASE(STR(?personaType)), "{persona.lower()}"))
            }}
            ORDER BY ?step
            """
            
            results = []
            for i, row in enumerate(g.query(query), 1):
                results.append({
                    'step_number': i,
                    'description': str(row[1]),
                    'refers_to_module': str(row[2]).split('_')[-1] if row[2] else 'N/A'
                })
                
        elif query_type == "cross_curricular":
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?title ?curriculum WHERE {
                ?module a currkg:Module ;
                        currkg:hasTitle ?title ;
                        currkg:alignsWithCurriculum ?curriculum .
            }
            """
            
            results = []
            for row in g.query(query):
                results.append({
                    'module': str(row[0]),
                    'connected_curriculum': str(row[1])
                })
                
        elif query_type == "topics_count":
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT (COUNT(DISTINCT ?topic) as ?count) WHERE {
                ?topic a currkg:Topic .
            }
            """
            
            result = list(g.query(query))[0]
            results = [{'total_topics': int(result[0])}]
            
        elif query_type == "sparql":
            sparql_query = params.get('query', '')
            if not sparql_query:
                return {'results': [], 'error': 'No SPARQL query provided'}
            
            query_results = g.query(sparql_query)
            results = []
            for row in query_results:
                results.append([str(item) for item in row])
                
        else:
            results = []
        
        return {
            'query_type': query_type,
            'results': results,
            'count': len(results),
            'source_file': ttl_file.name,
            'total_triples': len(g)
        }
        
    except Exception as e:
        return {'results': [], 'error': f'Query execution failed: {str(e)}'}

@router.get("/api/available-ontologies")
async def get_available_ontologies():
    """List all TTL files in data/outputs"""
    outputs_dir = Path("data/outputs")
    
    if not outputs_dir.exists():
        return {'ontologies': []}
    
    ontologies = []
    for ttl_file in outputs_dir.glob("*.ttl"):
        try:
            stat = ttl_file.stat()
            
            # Quick parse to get basic stats
            g = Graph()
            g.parse(ttl_file, format="turtle")
            
            ontologies.append({
                'filename': ttl_file.name,
                'created': stat.st_ctime,
                'size': stat.st_size,
                'triples_count': len(g)
            })
        except Exception:
            continue
    
    ontologies.sort(key=lambda x: x['created'], reverse=True)
    return {'ontologies': ontologies}

# Add these imports at the top
import os
from rdflib import Graph

# Replace your existing query functions with these properly integrated ones:

@router.post("/query-ontology")
async def query_ontology(query_request: dict):
    """Query the TTL ontology files"""
    
    query_type = query_request.get('query_type')
    params = query_request.get('params', {})
    
    # Load latest TTL file
    ttl_file = get_latest_ttl_file()
    if not ttl_file:
        return {'error': 'No TTL files found in data/outputs', 'results': []}
    
    g = load_ontology_graph(ttl_file)
    if not g:
        return {'error': f'Failed to load {ttl_file.name}', 'results': []}
    
    try:
        if query_type == "modules_by_topic":
            topic = params.get('topic', '').lower()
            
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT DISTINCT ?module ?title ?description ?topicDesc WHERE {{
                ?module a currkg:Module ;
                        currkg:hasTitle ?title ;
                        currkg:coversTopic ?topic .
                ?topic currkg:hasDescription ?topicDesc .
                OPTIONAL {{ ?module currkg:hasDescription ?description }}
                FILTER(CONTAINS(LCASE(?topicDesc), "{topic}"))
            }}
            """
            
            results = []
            for row in g.query(query):
                results.append({
                    'title': str(row[1]),
                    'description': str(row[2]) if row[2] else '',
                    'matching_topic': str(row[3])
                })
                
        elif query_type == "all_modules":
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?title ?description WHERE {
                ?module a currkg:Module ;
                        currkg:hasTitle ?title .
                OPTIONAL { ?module currkg:hasDescription ?description }
            }
            ORDER BY ?title
            """
            
            results = []
            for row in g.query(query):
                results.append({
                    'title': str(row[0]),
                    'description': str(row[1]) if row[1] else ''
                })
                
        elif query_type == "learning_path_steps":
            persona = params.get('persona', '')
            
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?step ?description ?module WHERE {{
                ?persona currkg:hasType ?personaType ;
                         currkg:determines ?path .
                ?path currkg:hasLearningStep ?step .
                ?step currkg:hasDescription ?description .
                OPTIONAL {{ ?step currkg:refersTo ?module }}
                FILTER(CONTAINS(LCASE(STR(?personaType)), "{persona.lower()}"))
            }}
            ORDER BY ?step
            """
            
            results = []
            for i, row in enumerate(g.query(query), 1):
                results.append({
                    'step_number': i,
                    'description': str(row[1]),
                    'refers_to_module': str(row[2]).split('_')[-1] if row[2] else 'N/A'
                })
                
        elif query_type == "cross_curricular":
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?title ?curriculum WHERE {
                ?module a currkg:Module ;
                        currkg:hasTitle ?title ;
                        currkg:alignsWithCurriculum ?curriculum .
            }
            """
            
            results = []
            for row in g.query(query):
                results.append({
                    'module': str(row[0]),
                    'connected_curriculum': str(row[1])
                })
                
        elif query_type == "topics_count":
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT (COUNT(DISTINCT ?topic) as ?count) WHERE {
                ?topic a currkg:Topic .
            }
            """
            
            result = list(g.query(query))[0]
            results = [{'total_topics': int(result[0])}]
            
        elif query_type == "sparql":
            sparql_query = params.get('query', '')
            if not sparql_query:
                return {'results': [], 'error': 'No SPARQL query provided'}
            
            query_results = g.query(sparql_query)
            results = []
            for row in query_results:
                results.append([str(item) for item in row])
                
        else:
            results = []
        
        return {
            'query_type': query_type,
            'results': results,
            'count': len(results),
            'source_file': ttl_file.name,
            'total_triples': len(g)
        }
        
    except Exception as e:
        return {'results': [], 'error': f'Query execution failed: {str(e)}'}

@router.get("/available-ontologies")
async def get_available_ontologies():
    """List all TTL files in data/outputs"""
    outputs_dir = Path("data/outputs")
    
    if not outputs_dir.exists():
        return {'ontologies': []}
    
    ontologies = []
    for ttl_file in outputs_dir.glob("*.ttl"):
        try:
            stat = ttl_file.stat()
            
            # Quick parse to get basic stats
            g = Graph()
            g.parse(ttl_file, format="turtle")
            
            ontologies.append({
                'filename': ttl_file.name,
                'created': stat.st_ctime,
                'size': stat.st_size,
                'triples_count': len(g)
            })
        except Exception:
            continue
    
    ontologies.sort(key=lambda x: x['created'], reverse=True)
    return {'ontologies': ontologies}

# Move these helper functions above the endpoints
def get_latest_ttl_file():
    """Get the most recent TTL file from data/outputs"""
    outputs_dir = Path("data/outputs")
    
    if not outputs_dir.exists():
        return None
    
    ttl_files = list(outputs_dir.glob("*.ttl"))
    if not ttl_files:
        return None
    
    # Get the most recent TTL file
    latest_file = max(ttl_files, key=os.path.getctime)
    return latest_file

def load_ontology_graph(ttl_file_path: Path):
    """Load TTL file into RDF graph"""
    try:
        g = Graph()
        g.parse(ttl_file_path, format="turtle")
        return g
    except Exception as e:
        logger.error(f"Failed to load TTL file {ttl_file_path}: {e}")
        return None


@router.post("/analyze-progression")
async def analyze_progression(
    curricula: List[str] = Form(...),
    llm_provider: str = Form(...),
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None)
):
    """Analyze learning progression coherence across curricula"""
    try:
        logger.info(f"Starting progression analysis for: {curricula}")
        
        if len(curricula) < 2:
            raise HTTPException(status_code=400, detail="At least 2 curricula required for progression analysis")
        
        # Validate LLM provider
        try:
            llm_provider_enum = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid llm_provider: {llm_provider}")
        
        # Setup LLM
        if api_key or llm_provider_enum.value == "ollama":
            contradiction_detector.setup_llm(llm_provider_enum, api_key or "")
        else:
            raise HTTPException(status_code=400, detail="API key required for non-Ollama providers")
        
        # Find ontology files for each curriculum
        outputs_dir = Path("data/outputs")
        ontology_paths = []
        
        for curriculum in curricula:
            curriculum_files = list(outputs_dir.glob(f"{Path(curriculum).stem}_*.ttl"))
            if not curriculum_files:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No ontology found for {curriculum}. Please extract ontology first."
                )
            # Use the most recent file
            latest_file = max(curriculum_files, key=lambda p: p.stat().st_mtime)
            ontology_paths.append(latest_file)
            logger.info(f"Using ontology: {latest_file.name} for {curriculum}")
        
        # Analyze progression coherence
        result = contradiction_detector.analyze_progression_coherence(ontology_paths, llm_provider_enum)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Progression analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-contradiction-report")
async def generate_contradiction_report(request: Dict[str, Any]):
    """Generate comprehensive contradiction report in Greek"""
    try:
        logger.info("Generating comprehensive contradiction report")
        
        # Extract provider info (should be available from the analysis results)
        llm_provider = request.get('llm_provider', 'openai')
        api_key = request.get('api_key', '')
        
        try:
            llm_provider_enum = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid llm_provider: {llm_provider}")
        
        # Setup LLM
        contradiction_detector.setup_llm(llm_provider_enum, api_key)
        
        # Generate comprehensive report
        report = contradiction_detector.generate_contradiction_report(
            internal_results=request.get('internal_results', {}),
            cross_results=request.get('cross_results', {}),
            progression_results=request.get('progression_results', {}),
            provider=llm_provider_enum
        )
        
        return PlainTextResponse(
            content=report,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=contradiction_report.txt"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-contradictions")
async def detect_contradictions(
    curriculum_files: List[str] = Form(...),
    analysis_type: str = Form(...),
    llm_provider: str = Form(...),
    detection_method: str = Form(default="basic"),  # NEW: basic or rag_enhanced
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None)
):
    """Unified endpoint for all contradiction detection types with RAG enhancement option"""
    try:
        logger.info(f"Starting {detection_method} {analysis_type} analysis for: {curriculum_files}")
        
        # Validate analysis type
        if analysis_type not in ["internal", "cross", "progression"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis_type: {analysis_type}. Must be 'internal', 'cross', or 'progression'"
            )
        
        # Validate detection method
        if detection_method not in ["basic", "rag_enhanced"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid detection_method: {detection_method}. Must be 'basic' or 'rag_enhanced'"
            )
        
        # Validate LLM provider
        try:
            llm_provider_enum = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid llm_provider: {llm_provider}")
        
        # Create appropriate detector based on method
        if detection_method == "rag_enhanced":
            from app.services.rag_contradiction_detector import create_rag_contradiction_detector
            detector = create_rag_contradiction_detector()
            logger.info("Using RAG-enhanced contradiction detector")
        else:
            detector = contradiction_detector  # Use your existing global detector
            logger.info("Using basic contradiction detector")
        
        # Setup LLM
        if api_key or llm_provider_enum.value == "ollama":
            detector.setup_llm(llm_provider_enum, api_key or "")
        else:
            raise HTTPException(status_code=400, detail="API key required for non-Ollama providers")
        
        # Route to appropriate analysis method
        if analysis_type == "internal":
            if len(curriculum_files) != 1:
                raise HTTPException(status_code=400, detail="Internal analysis requires exactly 1 curriculum file")
            
            # Find the latest ontology file
            outputs_dir = Path("data/outputs")
            ontology_files = list(outputs_dir.glob(f"{Path(curriculum_files[0]).stem}_*.ttl"))
            
            if not ontology_files:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No ontology found for {curriculum_files[0]}. Please extract ontology first."
                )
            
            ontology_path = max(ontology_files, key=lambda p: p.stat().st_mtime)
            result = detector.detect_internal_contradictions(ontology_path, llm_provider_enum)
            
        elif analysis_type == "cross":
            if len(curriculum_files) < 2:
                raise HTTPException(status_code=400, detail="Cross analysis requires at least 2 curriculum files")
            
            # Find ontology files for each curriculum
            outputs_dir = Path("data/outputs")
            ontology_paths = []
            
            for curriculum in curriculum_files:
                curriculum_files_found = list(outputs_dir.glob(f"{Path(curriculum).stem}_*.ttl"))
                if not curriculum_files_found:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"No ontology found for {curriculum}. Please extract ontology first."
                    )
                latest_file = max(curriculum_files_found, key=lambda p: p.stat().st_mtime)
                ontology_paths.append(latest_file)
            
            result = detector.detect_cross_curriculum_contradictions(ontology_paths, llm_provider_enum)
            
        elif analysis_type == "progression":
            if len(curriculum_files) < 2:
                raise HTTPException(status_code=400, detail="Progression analysis requires at least 2 curriculum files")
            
            # Find ontology files for each curriculum
            outputs_dir = Path("data/outputs")
            ontology_paths = []
            
            for curriculum in curriculum_files:
                curriculum_files_found = list(outputs_dir.glob(f"{Path(curriculum).stem}_*.ttl"))
                if not curriculum_files_found:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"No ontology found for {curriculum}. Please extract ontology first."
                    )
                latest_file = max(curriculum_files_found, key=lambda p: p.stat().st_mtime)
                ontology_paths.append(latest_file)
            
            result = detector.analyze_progression_coherence(ontology_paths, llm_provider_enum)
        
        # Add metadata to result
        result.update({
            "analysis_type": analysis_type,
            "detection_method": detection_method,
            "curricula_analyzed": curriculum_files,
            "llm_provider": llm_provider,
            "model_name": model_name
        })
        
        # Generate enhanced report for RAG method
        if detection_method == "rag_enhanced" and hasattr(detector, 'generate_rag_enhanced_report'):
            try:
                result["comprehensive_report"] = detector.generate_rag_enhanced_report(result, llm_provider_enum)
            except Exception as e:
                logger.warning(f"RAG report generation failed: {e}")
                result["comprehensive_report"] = "RAG report generation failed"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contradiction detection failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Add new endpoints for RAG status and method info
@router.get("/detection-methods")
async def get_detection_methods():
    """Get available contradiction detection methods"""
    return {
        "methods": [
            {
                "id": "basic",
                "name": "Basic Detection",
                "description": "Standard LLM-based contradiction detection",
                "features": [
                    "Internal contradictions",
                    "Cross-curriculum analysis", 
                    "Progression coherence",
                    "LLM-powered analysis"
                ]
            },
            {
                "id": "rag_enhanced",
                "name": "RAG-Enhanced Detection", 
                "description": "Enhanced detection using curriculum knowledge base",
                "features": [
                    "All basic features",
                    "RAG context from curriculum database",
                    "Pattern recognition from similar curricula",
                    "Best practices identification",
                    "Evidence-based recommendations",
                    "Cross-curricular pattern analysis"
                ]
            }
        ]
    }


@router.get("/rag-status")
async def get_rag_status():
    """Get status of RAG enhancement capabilities"""
    try:
        from app.services.rag_service import rag_service
        
        if not rag_service.curriculum_db:
            # Try to build database
            try:
                rag_service.build_curriculum_database(force_rebuild=False)
            except Exception as e:
                return {
                    "available": False,
                    "status": "unavailable",
                    "message": f"RAG database unavailable: {str(e)}",
                    "curricula_count": 0
                }
        
        return {
            "available": True,
            "status": "ready",
            "message": "RAG enhancement is available",
            "curricula_count": len(rag_service.curriculum_db),
            "curricula_files": list(rag_service.curriculum_db.keys()) if rag_service.curriculum_db else []
        }
        
    except ImportError:
        return {
            "available": False,
            "status": "not_installed", 
            "message": "RAG service not available",
            "curricula_count": 0
        }
    except Exception as e:
        return {
            "available": False,
            "status": "error",
            "message": f"RAG status check failed: {str(e)}",
            "curricula_count": 0
        }
@router.get("/debug/curricula")
async def debug_curricula():
    """Debug endpoint to see what curricula files are available"""
    try:
        curricula_dir = Path("data/curricula")
        
        info = {
            'directory_exists': curricula_dir.exists(),
            'directory_path': str(curricula_dir.absolute()),
            'files': []
        }
        
        if curricula_dir.exists():
            for file_path in curricula_dir.iterdir():
                info['files'].append({
                    'name': file_path.name,
                    'is_file': file_path.is_file(),
                    'is_pdf': file_path.suffix.lower() == '.pdf',
                    'size': file_path.stat().st_size if file_path.is_file() else 0
                })
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


@router.post("/extract-pydantic", response_model=ExtractionResponse)
async def extract_ontology_pydantic(request: ExtractionRequest):
    """Extract ontology from curriculum (Pydantic API)"""
    try:
        ontology_extractor.setup_llm(request.llm_provider, request.api_key, request.model_name)
        curricula_dir = Path("data/curricula")
        curriculum_path = curricula_dir / request.curriculum_filename
        if not curriculum_path.exists():
            raise HTTPException(status_code=404, detail="Curriculum file not found")

        result = ontology_extractor.extract_from_pdf(curriculum_path, request.extraction_mode, request.llm_provider)
        output_dir = Path("data/outputs"); output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{curriculum_path.stem}_{request.extraction_mode.value}_{request.llm_provider.value}.ttl"
        output_path = output_dir / output_filename
        ontology_extractor.save_ontology(result, output_path)

        json_graph = result.get("json_graph", {})
        entities = [n for n in json_graph.get("nodes", []) if n.get("category") == "entity"]
        concepts = [n for n in json_graph.get("nodes", []) if n.get("category") == "concept"]
        relations = json_graph.get("links", [])

        return {
            'success': True,
            'message': "Ontology extracted successfully",
            'extraction_mode': request.extraction_mode,
            'llm_provider': request.llm_provider,
            'output_filename': output_filename,
            'output_path': str(output_path),
            'extracted_elements': result['extracted_elements'],
            'total_elements': sum(result['extracted_elements'].values()),
            'json_graph': json_graph,
            'entities': entities,
            'concepts': concepts,
            'relations': relations
        }
    except Exception as e:
        logger.error(f"Pydantic extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/outputs")
async def list_output_files():
    """List generated ontology files"""
    try:
        outputs_dir = Path("data/outputs")
        if not outputs_dir.exists():
            return []
        
        files = []
        for file_path in outputs_dir.glob("*.ttl"):
            files.append({
                'filename': file_path.name,
                'size': file_path.stat().st_size,
                'size_mb': round(file_path.stat().st_size / (1024*1024), 2),
                'created': file_path.stat().st_mtime
            })
        
        return files
        
    except Exception as e:
        logger.error(f"Error listing output files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated ontology file"""
    outputs_dir = Path("data/outputs")
    file_path = outputs_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/turtle'
    )