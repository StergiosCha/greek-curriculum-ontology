"""
COMPLETE Enhanced Curriculum Ontology Extractor with FULL PROGRESSION TRACKING
FIXED VERSION - NO MORE ^b' ARTIFACTS
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import json
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from rdflib.namespace import NamespaceManager

# Import existing services
from app.services.llm_service import MultiLLMService, LLMProvider
from app.services.knowledge_enhancer import knowledge_enhancer
from app.services.rag_service import rag_service
from app.utils.file_handler import extract_text_from_pdf
from app.core.config import ExtractionMode
from app.services.focused_ontology_rag import FocusedOntologyRAGService

logger = logging.getLogger(__name__)

# Research paper aligned namespaces
CURRKG = Namespace("http://curriculum-kg.org/ontology/")
PROTOOKN = Namespace("http://proto-okn.net/")

# ============================================================
# NUCLEAR TEXT CLEANING FUNCTIONS - ADD AT TOP
# ============================================================

def nuclear_clean_text(text):
    """
    GUARANTEED clean text - removes ALL byte artifacts
    Call this on EVERY string before using it ANYWHERE
    """
    if text is None:
        return ""
    
    # Convert to string
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    text = str(text)
    
    # Remove ALL possible byte string patterns
    # Pattern 1: b'...'
    text = re.sub(r"b'([^']*)'", r'\1', text)
    text = re.sub(r'b"([^"]*)"', r'\1', text)
    
    # Pattern 2: ^b' anywhere
    text = re.sub(r'\^b[\'"]', '', text)
    text = re.sub(r'[\'"]?\^b[\'"]?', '', text)
    
    # Pattern 3: Just b' or b" at start
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:]
    if text.endswith("'") or text.endswith('"'):
        text = text[:-1]
    
    # Pattern 4: Embedded b' patterns
    text = text.replace("^b'", "")
    text = text.replace("'^b'", "")
    text = text.replace("' ^b'", "")
    text = text.replace('\^b\'', '')
    text = text.replace('"\^b\'', '')
    
    return text


def safe_escape_for_turtle(text):
    """
    Escape text for Turtle RDF - GUARANTEED SAFE
    """
    if not text:
        return ""
    
    # STEP 1: Clean ALL byte artifacts FIRST
    text = nuclear_clean_text(text)
    
    # STEP 2: Escape for RDF (order matters!)
    text = text.replace('\\', '\\\\')   # Backslashes first
    text = text.replace('"', '\\"')     # Quotes
    text = text.replace('\n', ' ')      # Newlines -> spaces (safer)
    text = text.replace('\r', '')       # Remove CR
    text = text.replace('\t', ' ')      # Tabs -> spaces
    
    # STEP 3: Final safety - remove any remaining artifacts
    if '^b' in text or "b'" in text:
        text = re.sub(r'\^b[\'"]?', '', text)
        text = re.sub(r'b[\'"]', '', text)
    
    # STEP 4: Limit length to prevent issues
    if len(text) > 1000:
        text = text[:997] + "..."
    
    return text


def deep_clean_parsed_data(data):
    """Recursively clean all strings in parsed JSON data"""
    if isinstance(data, dict):
        return {k: deep_clean_parsed_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_clean_parsed_data(item) for item in data]
    elif isinstance(data, str):
        return nuclear_clean_text(data)
    else:
        return data


# ============================================================
# ENUMS AND DATACLASSES
# ============================================================

class PersonaType(Enum):
    """Personas from research paper"""
    DEVELOPER = "developer"
    INSTRUCTOR = "instructor" 
    ANALYST = "analyst"
    EXECUTIVE = "executive"
    GRADUATE_STUDENT = "graduate_student"

@dataclass
class ResearchAlignedExtraction:
    """Extraction result following research paper structure"""
    curriculum: Dict[str, Any]
    learning_paths: List[Dict[str, Any]]
    modules: List[Dict[str, Any]]
    personas: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    competency_validation: Dict[str, float]
    axiom_compliance: List[str]

@dataclass
class EnhancedResearchExtraction:
    """EXTENDED extraction with Greek curriculum specifics and progression"""
    curriculum: Dict[str, Any]
    learning_paths: List[Dict[str, Any]]
    modules: List[Dict[str, Any]]
    personas: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    competency_validation: Dict[str, float]
    axiom_compliance: List[str]
    
    # Greek curriculum specific fields with progression
    learning_outcomes: List[Dict[str, Any]]
    assessment_strategies: List[Dict[str, Any]]
    teaching_strategies: List[Dict[str, Any]]
    time_allocations: Dict[str, Any]
    thematic_cycles: List[str]
    pedagogical_framework: Dict[str, Any]


# ============================================================
# MAIN EXTRACTOR CLASSES
# ============================================================

class ResearchPaperAlignedExtractor:
    """Extractor aligned with Christou et al. research paper structure"""
    
    def __init__(self):
        self.llm_service = MultiLLMService()
        self.ontology_rag = FocusedOntologyRAGService()
        self._rag_initialized = False
        self._initialize_services()
        self._load_existing_ontologies()
        
    def _initialize_services(self):
        """Initialize RAG and knowledge enhancement services"""
        if self._rag_initialized:
            logger.info("RAG service already initialized, skipping")
            return
            
        try:
            rag_service.initialize()
            rag_service.build_curriculum_database()
            self._rag_initialized = True
            logger.info("RAG service initialized")
        except Exception as e:
            logger.warning(f"RAG service initialization failed: {e}")
    
    def _load_existing_ontologies(self):
        """Load existing ontologies for RAG including CEDS"""
        ontology_dir = Path("data/outputs")
        self.ontology_rag.initialize_all_sources(ontology_dir)
        logger.info(f"Loaded {len(self.ontology_rag.loaded_ontologies)} ontologies for RAG")
   
    def setup_llm(self, provider: LLMProvider, api_key: str, model_name: Optional[str] = None):
        """Setup LLM provider"""
        self.llm_service.add_service(provider, api_key, model_name)
    
    def extract_research_aligned_ontology(self, 
                                        pdf_path: Path, 
                                        mode: ExtractionMode, 
                                        provider: LLMProvider,
                                        target_personas: List[PersonaType] = None) -> ResearchAlignedExtraction:
        """Extract modules FIRST, then pass to persona generation"""
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError(f"Could not extract text from {pdf_path}")
        
        logger.info(f"Processing {pdf_path.name} with research-aligned extraction")
        
        # Step 1: Extract curriculum structure
        curriculum_data = self._extract_curriculum_structure(text, provider)
        
        # Step 2: Extract modules FIRST
        modules = self._extract_modules_with_rag(text, provider)
        
        # Step 3: Generate persona paths using actual extracted modules
        target_personas = target_personas or [PersonaType.INSTRUCTOR, PersonaType.DEVELOPER]
        learning_paths = self._generate_persona_learning_paths(text, target_personas, provider, modules)
        
        # Step 4: Extract educational events
        events = self._extract_educational_events(text, provider)
        
        # Step 5: Enhance with external knowledge
        if mode in [ExtractionMode.LLM_ENHANCED, ExtractionMode.RAG_ENHANCED]:
            curriculum_data = self._enhance_with_external_knowledge(curriculum_data)
            modules = [self._enhance_module(m) for m in modules]
        
        # Step 6: Validate competency questions
        competency_scores = self._validate_competency_questions(
            curriculum_data, learning_paths, modules
        )
        
        # Step 7: Check axiom compliance
        axiom_violations = self._check_axiom_compliance(
            curriculum_data, learning_paths, modules
        )
        
        return ResearchAlignedExtraction(
            curriculum=curriculum_data,
            learning_paths=learning_paths,
            modules=modules,
            personas=[],
            events=events,
            competency_validation=competency_scores,
            axiom_compliance=axiom_violations
        )

    def _check_axiom_compliance(self, 
                            curriculum: Dict,
                            learning_paths: List[Dict],
                            modules: List[Dict]) -> List[str]:
        """Validate module references in learning steps"""
        
        violations = []
        
        # Get actual module titles for validation
        actual_module_titles = [m.get('title', '') for m in modules]
        
        # Axiom 1: Every Curriculum has a title
        if not curriculum.get('title'):
            violations.append("Curriculum missing hasTitle")
        
        # Axiom 2: Every Curriculum has at least one Module
        if not curriculum.get('modules') and not modules:
            violations.append("Curriculum missing hasModule")
        
        # Axiom 3: Every LearningPath is scoped by Curriculum
        for lp in learning_paths:
            if not lp.get('scoped_by_curriculum'):
                violations.append(f"LearningPath {lp.get('path_title', 'unknown')} missing scopedBy")
        
        # Axiom 4: Validate learning steps reference actual modules
        for lp in learning_paths:
            for step in lp.get('learning_steps', []):
                referred_module = step.get('refers_to_module')
                if not referred_module:
                    violations.append(f"LearningStep missing refersTo Module")
                elif referred_module not in actual_module_titles:
                    violations.append(f"LearningStep refers to non-existent module: {referred_module}")
        
        # Axiom 5: Every Module covers at least one Topic
        for module in modules:
            if not module.get('covers_topics'):
                violations.append(f"Module {module.get('title', 'unknown')} missing coversTopic")
        
        return violations

    def _extract_curriculum_structure(self, text: str, provider: LLMProvider) -> Dict[str, Any]:
        """Extract curriculum following research paper's Curriculum class definition"""
        
        prompt = f"""Extract curriculum structure from educational text following formal ontology requirements:

TEXT: {text}

REQUIREMENTS FROM RESEARCH PAPER:
- Every Curriculum MUST have a title (hasTitle)
- Every Curriculum MUST have at least one Module (hasModule)
- Curriculum should have organizational metadata

Extract as JSON (RESPOND ONLY WITH VALID JSON):
{{
"title": "Actual curriculum title from text",
"description": "Curriculum description", 
"education_level": "primary|secondary_lower|secondary_upper",
"subject_area": "greek_language|mathematics|science|etc",
"target_personas": ["instructor", "developer", "analyst"],
"modules": ["module1", "module2"],
"metadata": {{
    "source": "Full document analyzed",
    "extraction_date": "{datetime.now().isoformat()}",
    "axiom_compliant": true,
    "full_text_length": {len(text)}
}}
}}

CRITICAL: Return ONLY valid JSON. No explanations or additional text."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            return self._parse_json_response(response, fallback_data={
                "title": "Greek Curriculum",
                "description": "Curriculum extracted from educational material",
                "education_level": "primary",
                "subject_area": "greek_language",
                "target_personas": ["instructor", "analyst"],
                "modules": ["default_module"]
            })
        except Exception as e:
            logger.error(f"Curriculum structure extraction failed: {e}")
            return {
                "title": "Unknown Curriculum", 
                "modules": ["default_module"],
                "description": "Curriculum extracted from educational material"
            }
    
    def _generate_persona_learning_paths(self, 
                                    text: str, 
                                    personas: List[PersonaType],
                                    provider: LLMProvider,
                                    extracted_modules: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate learning paths ensuring steps reference actual modules"""
        
        learning_paths = []
        
        if not extracted_modules:
            logger.warning("No modules provided - will create basic learning path")
            extracted_modules = [{"title": "Basic Module", "description": "Default module"}]
        
        actual_module_titles = [m.get('title', f'Module_{i+1}') for i, m in enumerate(extracted_modules)]
        
        for persona in personas:
            try:
                existing_patterns = self.ontology_rag.get_learning_path_patterns(persona.value)
            except:
                existing_patterns = []
            
            rag_context = "EXISTING LEARNING PATH PATTERNS:\n"
            for pattern in existing_patterns[:3]:
                rag_context += f"From {pattern.get('source_curriculum', 'Unknown')}:\n"
                for step in pattern.get('steps', []):
                    rag_context += f"  - {step.get('description', 'No description')}\n"
                rag_context += "\n"
            
            module_list = "\n".join([f"- {title}" for title in actual_module_titles])
            
            prompt = f"""Create learning path for {persona.value} persona using curriculum content:

CURRENT CURRICULUM: {text[:2000]}

AVAILABLE MODULES (use EXACTLY these titles):
{module_list}

{rag_context}

PERSONA REQUIREMENTS:
- {persona.value.title()}: {self._get_persona_description(persona)}

CRITICAL REQUIREMENTS:
- Each learning step MUST have "refers_to_module" field 
- Use module titles EXACTLY as listed in AVAILABLE MODULES above
- Do NOT reference modules that aren't in the list above
- Maximum {len(actual_module_titles)} steps (one per available module)

Generate learning path as JSON (RESPOND ONLY WITH VALID JSON):
{{
"persona_type": "{persona.value}",
"path_title": "Learning path title",
"scoped_by_curriculum": "curriculum_title", 
"learning_steps": [
    {{
    "step_number": 1,
    "is_first": true,
    "title": "Step title",
    "description": "What the learner does",
    "refers_to_module": "EXACT_MODULE_TITLE_FROM_AVAILABLE_MODULES_LIST",
    "estimated_duration": "30 minutes"
    }}
],
"cross_curricular_connections": [],
"rag_enhanced": {len(existing_patterns) > 0}
}}

CRITICAL: refers_to_module MUST be EXACTLY from AVAILABLE MODULES list."""

            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                path_data = self._parse_json_response(response, fallback_data={
                    "persona_type": persona.value,
                    "path_title": f"Learning Path for {persona.value.title()}",
                    "learning_steps": [{
                        "step_number": 1,
                        "title": "Introduction",
                        "description": f"Basic introduction for {persona.value}",
                        "is_first": True,
                        "refers_to_module": actual_module_titles[0] if actual_module_titles else "Unknown"
                    }]
                })
                
                # Validate and filter steps
                if path_data and 'learning_steps' in path_data:
                    valid_steps = []
                    for step in path_data['learning_steps']:
                        referred_module = step.get('refers_to_module')
                        if referred_module in actual_module_titles:
                            valid_steps.append(step)
                        else:
                            logger.warning(f"Filtered out step referencing non-existent module: {referred_module}")
                    
                    if valid_steps:
                        path_data['learning_steps'] = valid_steps
                        learning_paths.append(path_data)
                        logger.info(f"✓ Created learning path for {persona.value} with {len(valid_steps)} steps")
                    else:
                        logger.error(f"✗ Learning path for {persona.value} has NO valid steps - creating minimal fallback")
                        fallback_path = {
                            "persona_type": persona.value,
                            "path_title": f"Basic Learning Path for {persona.value.title()}",
                            "scoped_by_curriculum": "curriculum",
                            "learning_steps": [{
                                "step_number": 1,
                                "is_first": True,
                                "title": "Introduction",
                                "description": f"Explore the curriculum content as {persona.value}",
                                "refers_to_module": actual_module_titles[0],
                                "estimated_duration": "30 minutes"
                            }]
                        }
                        learning_paths.append(fallback_path)
                        
            except Exception as e:
                logger.error(f"Learning path generation failed for {persona}: {e}")
        
        return learning_paths

    def _detect_expected_module_count(self, text: str) -> int:
        """Heuristically count modules/units in document"""
        pattern = re.compile(r'(?mi)^\s*\d{1,2}\s*η\s*ενότητα\b')
        return len(pattern.findall(text))

    def _split_by_grade_sections(self, text: str) -> Dict[str, str]:
        """Split text by grade sections"""
        a_pat = re.compile(r'(?is)(τάξη\s*α[΄\']?\s*γυμνασίου|α[΄\']?\s*τάξη\s*γυμνασίου)')
        b_pat = re.compile(r'(?is)(τάξη\s*β[΄\']?\s*γυμνασίου|β[΄\']?\s*τάξη\s*γυμνασίου)')

        a_m = a_pat.search(text)
        b_m = b_pat.search(text)

        sections = {"FULL": text}

        if a_m and b_m:
            a_start, b_start = a_m.start(), b_m.start()
            if a_start < b_start:
                sections["A_GYM"] = text[a_start:b_start]
                sections["B_GYM"] = text[b_start:]
            else:
                sections["B_GYM"] = text[b_start:a_start]
                sections["A_GYM"] = text[a_start:]
        elif a_m:
            sections["A_GYM"] = text[a_m.start():]
        elif b_m:
            sections["B_GYM"] = text[b_m.start():]

        return sections

    def _regex_extract_modules(self, text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Deterministic fallback that scrapes titles & hours"""
        mod_re = re.compile(r'(?mi)^\s*(\d{1,2})\s*η\s*ενότητα\s*\n+([^\n]+)', re.UNICODE)
        hrs1 = re.compile(r'(?mi)\bδιάρκεια\s+(\d+)\s*ώρ')
        hrs2 = re.compile(r'(?mi)\bώρες?\s+(\d+)\b')

        modules = []
        for m in mod_re.finditer(text):
            title = m.group(2).strip()
            win_start = m.end()
            window = text[win_start:win_start + 1200]
            hours = ""
            h = hrs1.search(window) or hrs2.search(window)
            if h:
                hours = h.group(1)

            grade = self._infer_grade_from_sections(title, sections)

            modules.append({
                "title": title,
                "description": "",
                "covers_topics": [],
                "level": "intermediate",
                "category": "survey", 
                "grade_level": grade or "unknown",
                "estimated_hours": hours,
                "learning_objectives": [],
                "assessment_methods": []
            })
        return modules

    def _detect_education_level(self, text: str) -> str:
        """Detect the education level from curriculum text"""
        text_lower = text.lower()
        
        primary_indicators = [
            "δημοτικ", "πρωτοβάθμι", "νηπιαγωγ", 
            "α΄ δημοτικού", "β΄ δημοτικού", "γ΄ δημοτικού",
            "δ΄ δημοτικού", "ε΄ δημοτικού", "στ΄ δημοτικού",
            "πρώτη τάξη", "δεύτερη τάξη", "τρίτη τάξη",
            "τετάρτη τάξη", "πέμπτη τάξη", "έκτη τάξη"
        ]
        
        upper_secondary_indicators = [
            "λύκει", "λυκειακ",
            "α΄ λυκείου", "β΄ λυκείου", "γ΄ λυκείου",
            "λυκείου α΄", "λυκείου β΄", "λυκείου γ΄"
        ]
        
        secondary_indicators = [
            "γυμνάσι", "δευτεροβάθμι",
            "α΄ γυμνασίου", "β΄ γυμνασίου", "γ΄ γυμνασίου",
            "γυμνασίου α΄", "γυμνασίου β΄", "γυμνασίου γ΄"
        ]
        
        if any(indicator in text_lower for indicator in primary_indicators):
            return "primary"
        elif any(indicator in text_lower for indicator in upper_secondary_indicators):
            return "upper_secondary"  
        elif any(indicator in text_lower for indicator in secondary_indicators):
            return "secondary"
        else:
            return "unknown"

    def _llm_extract_modules_once(self, text: str, provider: LLMProvider, expected_count: Optional[int] = None,
                                already_have_titles: Optional[List[str]] = None, section_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """FORCE LLM to extract topics for every module - IMPROVED WITH ORDER PRESERVATION"""
        
        already_have_titles = already_have_titles or []
        
        constraint = ""
        if expected_count:
            constraint += f"\nEXPECTED_TOTAL_MODULES: {expected_count} - Try to find ALL of them"
        if already_have_titles:
            safe_titles = [t.strip() for t in already_have_titles if t]
            constraint += f"\nALREADY_EXTRACTED_TITLES: {json.dumps(safe_titles, ensure_ascii=False)}"
            constraint += "\nReturn ONLY modules whose titles are NOT in ALREADY_EXTRACTED_TITLES."
        if section_hint:
            constraint += f"\nSECTION: {section_hint} (extract modules ONLY from this section)."

        education_level = self._detect_education_level(text)
        
        if education_level == "primary":
            grade_options = "A Dimotikou|B Dimotikou|C Dimotikou|D Dimotikou|E Dimotikou|ST Dimotikou|All Dimotiko|unknown"
        elif education_level == "secondary":
            grade_options = "A Gymnasiou|B Gymnasiou|C Gymnasiou|All Gimnasio|unknown"
        elif education_level == "upper_secondary":
            grade_options = "A Lykeiou|B Lykeiou|C Lykeiou|All Likio|unknown"
        else:
            grade_options = "unknown"

        # Use more text for better extraction
        text_sample = text[:15000] if len(text) > 15000 else text

        modules_prompt = f"""
Extract ALL educational modules/units from this curriculum. Be COMPREHENSIVE and THOROUGH.

CRITICAL: Extract modules in the EXACT ORDER they appear in the document. Preserve the document's pedagogical sequence.

Look for:
- Numbered units (1η ενότητα, 2η ενότητα, etc.)
- Thematic units (θεματικές ενότητες)
- Learning modules (διδακτικές ενότητες)
- Chapter titles
- Major topic sections
- Table of contents entries

For EACH module extract:
1. Title (exact from text)
2. Description (what it's about)
3. Topics covered (AT LEAST 2-5 topics per module)
4. Learning objectives (what students will learn)
5. Grade level (ONLY if EXPLICITLY mentioned for that specific module)
6. Duration (if mentioned)

GRADE LEVEL ASSIGNMENT RULES:
- ONLY assign a specific grade if the module EXPLICITLY mentions it
- If module says "Α΄ τάξη" or "για την Α΄ τάξη" → use "A Dimotikou"
- If module says "Α΄ και Β΄ τάξη" → use "A Dimotikou" (first mentioned)
- If module is GENERAL/THEORETICAL (e.g., "Θεωρητικό πλαίσιο", "Στρατηγικές", "Αξιολόγηση") → use "unknown"
- If module title contains "Αναλυτικό Πρόγραμμα" or "για όλες τις τάξεις" → use "unknown"
- If unsure → use "unknown" (better than guessing wrong)

REQUIREMENTS:
- Return a JSON ARRAY starting with '[' and ending with ']'
- Include ALL modules found, not just a sample
- PRESERVE THE DOCUMENT ORDER - first module in document = first in array
- EVERY module MUST have "covers_topics" with 2-5 actual topics from the text
- Extract the REAL subject matter for each module
- Don't leave covers_topics empty
- Use correct grade format: {grade_options}
- DON'T guess grades - use "unknown" if not explicitly stated

{constraint}

TEXT:
{text_sample}

OUTPUT FORMAT (in document order):
[
  {{
    "title": "Module title from text",
    "description": "What the module covers",
    "covers_topics": ["topic 1", "topic 2", "topic 3"],
    "level": "beginner|intermediate|advanced",
    "category": "foundation|survey|methodology|standard",
    "grade_level": "ONLY if explicitly mentioned, otherwise unknown",
    "estimated_hours": "number or empty string",
    "learning_objectives": ["objective 1", "objective 2"],
    "assessment_methods": ["method 1", "method 2"],
    "document_order": 1
  }}
]

CRITICAL RULES:
1. Extract EVERY module - if you see 10 modules, extract all 10
2. covers_topics MUST have real topics from the text
3. If expected count is {expected_count}, try to find that many
4. MAINTAIN DOCUMENT ORDER - this is pedagogically important
5. Don't reorder - first in document = first in output
6. BE CONSERVATIVE with grade assignment - use "unknown" when in doubt
7. Return [] only if NO modules exist

Example - Correct grade assignment:
✓ "1. Θεωρητικό πλαίσιο" → grade_level: "unknown" (general section)
✓ "2. Στρατηγικές ενθάρρυνσης" → grade_level: "unknown" (applies to all)
✓ "3. Δραστηριότητες για Α΄ τάξη" → grade_level: "A Dimotikou" (explicit)
✗ "3. Η Λογοτεχνία στο ΑΠΣ Δημοτικού" → grade_level: "All_Dimotiko" (WRONG - use "unknown")

Respond with ONLY valid JSON array, no markdown, no explanations."""
        
        resp = self.llm_service.generate_with_provider(provider, modules_prompt)
        parsed = self._parse_json_response(str(resp), fallback_data=[])
        
        logger.info(f"LLM extraction pass returned {len(parsed) if isinstance(parsed, list) else 0} modules")
        
        # Verify we got a list and log the order
        if isinstance(parsed, list) and len(parsed) > 0:
            logger.info(f"Extracted modules in order: {[m.get('title', 'Unknown')[:40] for m in parsed[:5]]}")
        
        return parsed if isinstance(parsed, list) else []

    def _infer_grade_from_sections(self, title: str, sections: Dict[str, str]) -> Optional[str]:
        """Check which section contains title and return appropriate grade format - FIXED"""
        
        full_text = sections.get("FULL", "").lower()
        title_lower = title.lower()
        education_level = self._detect_education_level(full_text)
        
        # IMPORTANT: If the module title itself doesn't contain grade indicators,
        # and it appears to be a general/theoretical section, return 'unknown'
        # to avoid false grade assignments
        
        general_module_indicators = [
            'θεωρητικό πλαίσιο', 'theoretical framework',
            'στρατηγικές', 'strategies',
            'αξιολόγηση', 'assessment', 'evaluation',
            'εισαγωγ', 'introduction',
            'παράδειγμα', 'example',
            'δράσεις', 'activities',
            'βιβλιογραφία', 'bibliography',
            'παράρτημα', 'appendix',
            'αναλυτικό πρόγραμμα', 'curriculum',
            'διδακτικ', 'instructional'
        ]
        
        # Check if this is a general/theoretical module
        if any(indicator in title_lower for indicator in general_module_indicators):
            logger.info(f"Module '{title[:40]}' appears to be general/theoretical - not assigning specific grade")
            return 'unknown'
        
        if education_level == "primary":
            grade_patterns = {
                r'α[΄\']?\s*τάξη|α[΄\']?\s*δημοτικ': "A Dimotikou",
                r'β[΄\']?\s*τάξη|β[΄\']?\s*δημοτικ': "B Dimotikou", 
                r'γ[΄\']?\s*τάξη|γ[΄\']?\s*δημοτικ': "C Dimotikou",
                r'δ[΄\']?\s*τάξη|δ[΄\']?\s*δημοτικ': "D Dimotikou",
                r'ε[΄\']?\s*τάξη|ε[΄\']?\s*δημοτικ': "E Dimotikou",
                r'στ[΄\']?\s*τάξη|στ[΄\']?\s*δημοτικ': "ST Dimotikou"
            }
        elif education_level == "secondary":
            grade_patterns = {
                r'α[΄\']?\s*γυμνασίου|α[΄\']?\s*τάξη\s*γυμνασίου': "A Gymnasiou",
                r'β[΄\']?\s*γυμνασίου|β[΄\']?\s*τάξη\s*γυμνασίου': "B Gymnasiou",
                r'γ[΄\']?\s*γυμνασίου|γ[΄\']?\s*τάξη\s*γυμνασίου': "C Gymnasiou"
            }
        elif education_level == "upper_secondary":
            grade_patterns = {
                r'α[΄\']?\s*λυκείου|α[΄\']?\s*τάξη\s*λυκείου': "A Lykeiou",
                r'β[΄\']?\s*λυκείου|β[΄\']?\s*τάξη\s*λυκείου': "B Lykeiou", 
                r'γ[΄\']?\s*λυκείου|γ[΄\']?\s*τάξη\s*λυκείου': "C Lykeiou"
            }
        else:
            return "unknown"
        
        # Only assign grade if we find EXPLICIT grade mention in the context
        for pattern, grade_name in grade_patterns.items():
            if re.search(pattern, title_lower, re.IGNORECASE):
                logger.info(f"Found explicit grade in title '{title[:40]}' -> {grade_name}")
                return grade_name
        
        # Check in document context around the module (more lenient)
        # Only if title itself doesn't indicate it's general
        return "unknown"

    def _dedup_modules_by_title(self, modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep first occurrence per casefolded title"""
        seen = set()
        deduped = []
        for m in modules:
            key = (m.get("title") or "").strip().casefold()
            if key and key not in seen:
                deduped.append(m)
                seen.add(key)
        return deduped

    def _complete_cross_curricular(self, modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced cross-curricular using ONLY TTL ontologies and CEDS"""
        
        for module in modules:
            if module.get('covers_topics'):
                all_connections = []
                
                for topic in module['covers_topics'][:2]:
                    try:
                        ttl_connections = self.ontology_rag.find_cross_curricular_connections(topic)
                        all_connections.extend(ttl_connections)
                        
                        ceds_connections = self.ontology_rag.query_ceds_for_curriculum_alignment(
                            module.get('title', ''), [topic]
                        )
                        all_connections.extend(ceds_connections)
                        
                    except Exception as e:
                        logger.warning(f"Focused cross-curricular query failed for {topic}: {e}")
                
                if all_connections:
                    module['cross_curricular_connections'] = all_connections[:3]
                    module['rag_source'] = 'ttl_and_ceds_only'
        
        return modules

    def _extract_modules_with_rag(self, text: str, provider: LLMProvider) -> List[Dict[str, Any]]:
        """Read whole document, guarantee completeness AND preserve order"""
        
        sections = self._split_by_grade_sections(text)
        expected = self._detect_expected_module_count(text)

        # Pass 1: whole text - PRESERVE ORIGINAL ORDER
        modules = self._llm_extract_modules_once(text, provider, expected_count=expected)
        modules = modules if isinstance(modules, list) else []
        
        # Log original order
        logger.info(f"Initial extraction order: {[m.get('title', 'Unknown')[:30] for m in modules[:5]]}")
        
        # Deduplicate but PRESERVE ORDER (don't use set)
        seen_titles = []
        deduped = []
        for m in modules:
            title = (m.get("title") or "").strip().casefold()
            if title and title not in seen_titles:
                deduped.append(m)
                seen_titles.append(title)
        modules = deduped

        # Quick grade inference
        for m in modules:
            if not m.get("grade_level") or m.get("grade_level") == "unknown":
                gl = self._infer_grade_from_sections(m.get("title", ""), sections)
                if gl:
                    m["grade_level"] = gl

        # Pass 2: if incomplete, try per section
        have_titles = [m.get("title", "") for m in modules]
        if expected and 0 < len(modules) < expected:
            if "A_GYM" in sections:
                extra_a = self._llm_extract_modules_once(sections["A_GYM"], provider,
                                                        expected_count=None,
                                                        already_have_titles=have_titles,
                                                        section_hint="A Gymnasio")
                modules += extra_a
                # Deduplicate preserving order
                seen_titles = []
                deduped = []
                for m in modules:
                    title = (m.get("title") or "").strip().casefold()
                    if title and title not in seen_titles:
                        deduped.append(m)
                        seen_titles.append(title)
                modules = deduped
                have_titles = [m.get("title", "") for m in modules]
            
            if expected and len(modules) < expected and "B_GYM" in sections:
                extra_b = self._llm_extract_modules_once(sections["B_GYM"], provider,
                                                        expected_count=None,
                                                        already_have_titles=have_titles,
                                                        section_hint="B Gymnasio")
                modules += extra_b
                # Deduplicate preserving order
                seen_titles = []
                deduped = []
                for m in modules:
                    title = (m.get("title") or "").strip().casefold()
                    if title and title not in seen_titles:
                        deduped.append(m)
                        seen_titles.append(title)
                modules = deduped
                have_titles = [m.get("title", "") for m in modules]

        # Infer grade where still missing
        for m in modules:
            if not m.get("grade_level") or m.get("grade_level") == "unknown":
                gl = self._infer_grade_from_sections(m.get("title", ""), sections)
                if gl:
                    m["grade_level"] = gl

        # Pass 3: regex fallback if still short
        if expected and len(modules) < expected:
            regex_mods = self._regex_extract_modules(text, sections)
            titles = {t.strip().casefold() for t in have_titles}
            regex_mods = [rm for rm in regex_mods if (rm.get("title","").strip().casefold() not in titles)]
            modules += regex_mods
            # Final deduplicate preserving order
            seen_titles = []
            deduped = []
            for m in modules:
                title = (m.get("title") or "").strip().casefold()
                if title and title not in seen_titles:
                    deduped.append(m)
                    seen_titles.append(title)
            modules = deduped

        # Log final order
        logger.info(f"Final module order: {[m.get('title', 'Unknown')[:30] for m in modules[:5]]}")

        # Final cross-curricular enrichment
        modules = self._complete_cross_curricular(modules)

        return modules

    def _extract_educational_events(self, text: str, provider: LLMProvider) -> List[Dict[str, Any]]:
        """Extract educational events following research paper Event class"""
        
        prompt = f"""Extract educational events from curriculum text:

TEXT: {text[:3000]}

Look for: workshops, seminars, presentations, conferences, assessments, activities

Extract as JSON (RESPOND ONLY WITH VALID JSON):
[
{{
    "title": "Event title from text",
    "type": "workshop|seminar|presentation|assessment|conference|activity",
    "description": "Event description",
    "provides_media": ["media_type1", "media_type2"],
    "has_sub_events": ["subevent1", "subevent2"],
    "duration": "extracted or estimated duration",
    "participants": "target participants"
}}
]

CRITICAL: 
- Return ONLY valid JSON array
- If NO events found in text, return empty array: []
- Do NOT explain or add text outside JSON"""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            if response and not response.strip().startswith('[') and not response.strip().startswith('{'):
                logger.info("No events found in text - using empty array")
                return []
            
            events = self._parse_json_response(response, fallback_data=[])
            return events if isinstance(events, list) else []
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return []

    def _enhance_with_external_knowledge(self, curriculum_data: Dict) -> Dict:
        """Enhance curriculum with external knowledge"""
        
        if not curriculum_data.get('title'):
            return curriculum_data
        
        try:
            enhancement = knowledge_enhancer.enhance_learning_objective(
                curriculum_data['title'], 
                curriculum_data.get('subject_area', 'general')
            )
            
            curriculum_data['external_enhancement'] = {
                'bloom_taxonomy': enhancement.get('bloom_taxonomy', {}),
                'cefr_alignment': enhancement.get('cefr_alignment', {}),
                'international_standards': enhancement.get('international_standards', []),
                'pedagogical_approaches': enhancement.get('pedagogical_approaches', [])
            }
            
        except Exception as e:
            logger.warning(f"External knowledge enhancement failed: {e}")
        
        return curriculum_data
    
    def _enhance_module(self, module: Dict) -> Dict:
        """Enhance individual module with external knowledge"""
        
        if not module.get('learning_objectives'):
            return module
        
        try:
            objective = module['learning_objectives'][0]
            enhancement = knowledge_enhancer.enhance_learning_objective(objective)
            
            module['knowledge_enhancement'] = enhancement
            
        except Exception as e:
            logger.warning(f"Module enhancement failed: {e}")
        
        return module
    
    def _validate_competency_questions(self, 
                                     curriculum: Dict,
                                     learning_paths: List[Dict],
                                     modules: List[Dict]) -> Dict[str, float]:
        """Validate extraction against research paper's competency questions"""
        
        scores = {}
        
        # CQ1: Which persona is associated with which learning path?
        if learning_paths and all('persona_type' in lp for lp in learning_paths):
            scores['persona_learning_paths'] = 1.0
        else:
            scores['persona_learning_paths'] = 0.0
        
        # CQ2: What are all the materials that explain specific topics?
        topic_coverage = 0.0
        if modules:
            modules_with_topics = sum(1 for m in modules if m.get('covers_topics'))
            topic_coverage = modules_with_topics / len(modules) if modules else 0
        scores['topic_coverage'] = topic_coverage
        
        # CQ3: What learning objectives by grade level?
        grade_coverage = 0.0
        if modules:
            modules_with_grades = sum(1 for m in modules if m.get('grade_level'))
            grade_coverage = modules_with_grades / len(modules) if modules else 0
        scores['grade_level_coverage'] = grade_coverage
        
        # CQ4: Cross-curricular connections
        cross_connections = 0.0
        if modules:
            modules_with_connections = sum(1 for m in modules if m.get('cross_curricular_connections'))
            cross_connections = modules_with_connections / len(modules) if modules else 0
        scores['cross_curricular_connections'] = cross_connections
        
        return scores
    
    def _get_persona_description(self, persona: PersonaType) -> str:
        """Get persona description from research paper"""
        descriptions = {
            PersonaType.DEVELOPER: "Hands-on technical implementation, coding exercises, practical projects",
            PersonaType.INSTRUCTOR: "Teaching methodology, lesson planning, assessment strategies", 
            PersonaType.ANALYST: "Data analysis, research methods, evaluation techniques",
            PersonaType.EXECUTIVE: "High-level overview, strategic implications, decision-making support",
            PersonaType.GRADUATE_STUDENT: "Academic depth, research orientation, theoretical foundations"
        }
        return descriptions.get(persona, "General educational path")
    
    def _parse_json_response(self, response: str, fallback_data: Any = None) -> Any:
        """Parse LLM JSON response with NUCLEAR cleaning"""
        if not response or not response.strip():
            logger.warning("Empty response received")
            return fallback_data
        
        try:
            # CRITICAL: Clean the response FIRST
            response = nuclear_clean_text(response)
            
            cleaned = response.strip()
            
            # Remove markdown
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            elif cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if len(lines) > 2:
                    cleaned = '\n'.join(lines[1:-1])
            
            # Extract JSON
            json_match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            
            # Parse
            parsed = json.loads(cleaned)
            
            # CRITICAL: Clean all strings in parsed data
            parsed = deep_clean_parsed_data(parsed)
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response[:200]}...")
            
            if fallback_data is not None:
                logger.info(f"Using fallback data: {type(fallback_data)}")
                return fallback_data
            
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return fallback_data
    
    def generate_research_compliant_turtle(self, extraction: ResearchAlignedExtraction) -> str:
        """Generate Turtle RDF - GUARANTEED CLEAN VERSION"""
        
        turtle_lines = [
            "@prefix currkg: <http://curriculum-kg.org/ontology/> .",
            "@prefix protookn: <http://proto-okn.net/> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
            "# Research Paper Compliant Curriculum Ontology",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Axiom violations: {len(extraction.axiom_compliance)}",
            f"# Competency score: {sum(extraction.competency_validation.values()) / len(extraction.competency_validation) if extraction.competency_validation else 0:.2f}",
            ""
        ]
        
        # Create module title to index mapping
        module_title_to_index = {}
        for i, module in enumerate(extraction.modules, 1):
            module_title = safe_escape_for_turtle(module.get('title', f'Module_{i}'))
            module_title_to_index[module_title] = i
        
        # Generate Curriculum
        curriculum = extraction.curriculum
        curriculum_uri = f"currkg:Curriculum_{self._safe_uri_name(curriculum.get('title', 'Unknown'))}"
        
        turtle_lines.extend([
            f"{curriculum_uri} a currkg:Curriculum ;",
            f'    currkg:hasTitle "{safe_escape_for_turtle(curriculum.get("title", "Unknown Curriculum"))}" ;'
        ])
        
        if curriculum.get('description'):
            turtle_lines.append(f'    currkg:hasDescription "{safe_escape_for_turtle(curriculum["description"])}" ;')
        
        # Add modules to curriculum
        for i, module in enumerate(extraction.modules, 1):
            module_uri = f"currkg:Module_{i}"
            turtle_lines.append(f"    currkg:hasModule {module_uri} ;")
        
        turtle_lines[-1] = turtle_lines[-1].rstrip(' ;') + ' .'
        turtle_lines.append("")
        
        # Generate Modules
        for i, module in enumerate(extraction.modules, 1):
            module_uri = f"currkg:Module_{i}"
            
            turtle_lines.extend([
                f"{module_uri} a currkg:Module ;",
                f'    currkg:hasTitle "{safe_escape_for_turtle(module.get("title", f"Module {i}"))}" ;'
            ])
            
            if module.get('description'):
                turtle_lines.append(f'    currkg:hasDescription "{safe_escape_for_turtle(module["description"])}" ;')
            
            # Cover topics
            for j, topic in enumerate(module.get('covers_topics', [])[:10], 1):
                topic_uri = f"currkg:Topic_{i}_{j}"
                turtle_lines.append(f"    currkg:coversTopic {topic_uri} ;")
            
            level = module.get('level')
            if level and level != 'unknown':
                turtle_lines.append(f"    currkg:hasLevel currkg:{safe_escape_for_turtle(level).title()}Level ;")
            
            category = module.get('category')
            if category and category != 'unknown':
                turtle_lines.append(f"    currkg:belongsTo currkg:{safe_escape_for_turtle(category).title()}Category ;")
            
            grade_level = module.get('grade_level')
            if grade_level and grade_level != 'unknown':
                safe_grade = safe_escape_for_turtle(grade_level).replace(' ', '_')
                turtle_lines.append(f"    currkg:hasGradeLevel currkg:{safe_grade} ;")
            
            if module.get('cross_curricular_connections'):
                for conn in module['cross_curricular_connections'][:3]:
                    if isinstance(conn, dict) and conn.get('source_curriculum'):
                        turtle_lines.append(f'    currkg:alignsWithCurriculum "{safe_escape_for_turtle(conn["source_curriculum"])}" ;')
            
            turtle_lines[-1] = turtle_lines[-1].rstrip(' ;') + ' .'
            turtle_lines.append("")
        
        # Generate Learning Paths
        for i, learning_path in enumerate(extraction.learning_paths, 1):
            path_uri = f"currkg:LearningPath_{i}"
            persona_uri = f"currkg:Persona_{self._safe_uri_name(learning_path.get('persona_type', 'Unknown'))}"
            
            turtle_lines.extend([
                f"{path_uri} a currkg:LearningPath ;",
                f'    currkg:hasTitle "{safe_escape_for_turtle(learning_path.get("path_title", f"Learning Path {i}"))}" ;',
                f"    currkg:scopedBy {curriculum_uri} ;",
                f"    currkg:determines {persona_uri} ;"
            ])
            
            steps = learning_path.get('learning_steps', [])
            for j, step in enumerate(steps):
                step_uri = f"currkg:LearningStep_{i}_{j+1}"
                turtle_lines.append(f"    currkg:hasLearningStep {step_uri} ;")
            
            turtle_lines[-1] = turtle_lines[-1].rstrip(' ;') + ' .'
            turtle_lines.append("")
            
            # Generate individual learning steps
            for j, step in enumerate(steps):
                step_uri = f"currkg:LearningStep_{i}_{j+1}"
                
                turtle_lines.extend([
                    f"{step_uri} a currkg:LearningStep"
                ])
                
                if step.get('is_first'):
                    turtle_lines.append(f"    , currkg:FirstLearningStep")
                
                if j == len(steps) - 1:
                    turtle_lines.append(f"    , currkg:LastLearningStep")
                
                turtle_lines.extend([
                    "    ;",
                    f'    currkg:hasDescription "{safe_escape_for_turtle(step.get("description", f"Step {j+1}"))}" ;'
                ])
                
                # Only add refersTo triple if module actually exists
                referred_module = step.get('refers_to_module')
                if referred_module:
                    clean_referred = safe_escape_for_turtle(referred_module)
                    if clean_referred in module_title_to_index:
                        module_index = module_title_to_index[clean_referred]
                        turtle_lines.append(f"    currkg:refersTo currkg:Module_{module_index} ;")
                
                if j < len(steps) - 1:
                    next_step_uri = f"currkg:LearningStep_{i}_{j+2}"
                    turtle_lines.append(f"    currkg:hasNextLearningStep {next_step_uri} ;")
                
                turtle_lines[-1] = turtle_lines[-1].rstrip(' ;') + ' .'
                turtle_lines.append("")
        
        # Generate Personas
        personas_generated = set()
        for learning_path in extraction.learning_paths:
            persona_type = learning_path.get('persona_type', 'Unknown')
            if persona_type not in personas_generated:
                persona_uri = f"currkg:Persona_{self._safe_uri_name(persona_type)}"
                
                turtle_lines.extend([
                    f"{persona_uri} a currkg:Persona ;",
                    f"    currkg:hasType currkg:{safe_escape_for_turtle(persona_type).title()}PersonaType ;",
                    f"    currkg:hasProfession currkg:{safe_escape_for_turtle(persona_type).title()}Profession ;",
                    "    ."
                ])
                turtle_lines.append("")
                personas_generated.add(persona_type)
        
        # Generate Topics
        for i, module in enumerate(extraction.modules, 1):
            for j, topic in enumerate(module.get('covers_topics', [])[:10], 1):
                topic_uri = f"currkg:Topic_{i}_{j}"
                clean_topic = safe_escape_for_turtle(topic)
                
                turtle_lines.extend([
                    f"{topic_uri} a currkg:Topic ;",
                    f'    currkg:hasDescription "{clean_topic}" ;',
                    f'    currkg:asString "{clean_topic}" .'
                ])
                turtle_lines.append("")
        
        # Validation metadata
        turtle_lines.extend([
            "# Ontology Validation Metadata",
            "currkg:OntologyValidation a currkg:ValidationReport ;",
            f'    currkg:extractionDate "{datetime.now().isoformat()}"^^xsd:dateTime ;',
            f'    currkg:axiomViolations {len(extraction.axiom_compliance)} ;'
        ])
        
        for cq, score in extraction.competency_validation.items():
            turtle_lines.append(f'    currkg:competencyScore_{cq} {score:.3f} ;')
        
        turtle_lines[-1] = turtle_lines[-1].rstrip(' ;') + ' .'
        
        output = "\n".join(turtle_lines)
        
        # FINAL NUCLEAR CHECK
        if '^b' in output:
            logger.error("CRITICAL: Found ^b in final output after all cleaning!")
            output = re.sub(r'\^b[\'"]?', '', output)
            output = re.sub(r'b[\'"]', '', output)
        
        return output
    
    def _safe_uri_name(self, name: str) -> str:
        """Convert text to safe URI component"""
        if not name:
            return "Unknown"
        name = nuclear_clean_text(name)
        safe_name = re.sub(r'[^\w]', '_', str(name))
        safe_name = re.sub(r'_+', '_', safe_name)
        return safe_name.strip('_') or "Unknown"
    
    def save_research_compliant_ontology(self, extraction: ResearchAlignedExtraction, output_path: Path):
        """Save ontology with research paper metadata"""
        
        turtle_content = self.generate_research_compliant_turtle(extraction)
        
        # FINAL clean before write
        turtle_content = nuclear_clean_text(turtle_content)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(turtle_content)
        
        # Verify
        with open(output_path, 'r', encoding='utf-8') as f:
            verify = f.read()
            if '^b' in verify:
                logger.error("STILL HAS ^b AFTER WRITE!")
                fixed = nuclear_clean_text(verify)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(fixed)
                logger.info("Applied emergency fix to file")
            else:
                logger.info("✓ File clean after write - no ^b' found")
        
        validation_report = {
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "research_paper_aligned": True,
                "ontology_file": str(output_path)
            },
            "axiom_compliance": {
                "violations": extraction.axiom_compliance,
                "compliant": len(extraction.axiom_compliance) == 0
            },
            "competency_validation": extraction.competency_validation,
            "overall_quality_score": sum(extraction.competency_validation.values()) / len(extraction.competency_validation) if extraction.competency_validation else 0.0,
            "curriculum_stats": {
                "modules_count": len(extraction.modules),
                "learning_paths_count": len(extraction.learning_paths),
                "events_count": len(extraction.events),
                "cross_curricular_connections": sum(1 for m in extraction.modules if m.get('cross_curricular_connections'))
            }
        }
        
        report_path = output_path.with_suffix('.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved research-compliant ontology to {output_path}")
        logger.info(f"Overall quality score: {validation_report['overall_quality_score']:.3f}")
        
        return output_path

    def _convert_extraction_to_graph(self, extraction) -> Dict[str, Any]:
        """Convert extraction to complete graph format with ALL relationships"""
        
        nodes = []
        links = []
        node_map = {}
        
        # Add curriculum node
        curriculum_id = "curriculum_0"
        nodes.append({
            "id": curriculum_id,
            "label": extraction.curriculum.get('title', 'Curriculum'),
            "type": "Curriculum",
            "category": "entity",
            "color": "#FF6B6B"
        })
        node_map['curriculum'] = curriculum_id
        
        # Add modules with progression
        for i, module in enumerate(extraction.modules):
            module_id = f"module_{i}"
            node_map[f"module_{module.get('title', f'Module {i}')}"] = module_id
            
            nodes.append({
                "id": module_id,
                "label": module.get('title', f'Module {i}'),
                "type": "Module",
                "category": "entity",
                "color": "#4ECDC4",
                "gradeLevel": module.get('grade_level', 'unknown'),
                "progressionLevel": module.get('progression_level', 'unknown')
            })
            
            links.append({
                "source": curriculum_id,
                "target": module_id,
                "label": "hasModule",
                "type": "structure"
            })
            
            # Add prerequisite links
            for prereq_title in module.get('prerequisite_modules', []):
                prereq_id = node_map.get(f"module_{prereq_title}")
                if prereq_id:
                    links.append({
                        "source": prereq_id,
                        "target": module_id,
                        "label": "prerequisiteFor",
                        "type": "progression"
                    })
        
        # Add topics
        topic_counter = 0
        for i, module in enumerate(extraction.modules):
            module_id = f"module_{i}"
            for topic in module.get('covers_topics', [])[:5]:  # Limit to 5 topics per module
                topic_id = f"topic_{topic_counter}"
                topic_counter += 1
                
                nodes.append({
                    "id": topic_id,
                    "label": topic[:50],
                    "type": "Topic",
                    "category": "concept",
                    "color": "#FFEAA7",
                    "description": topic
                })
                
                links.append({
                    "source": module_id,
                    "target": topic_id,
                    "label": "coversTopic",
                    "type": "content"
                })
        
        # Add learning paths
        for i, path in enumerate(extraction.learning_paths):
            path_id = f"path_{i}"
            
            nodes.append({
                "id": path_id,
                "label": path.get('path_title', f'Learning Path {i}'),
                "type": "LearningPath",
                "category": "entity",
                "color": "#45B7D1"
            })
            
            links.append({
                "source": curriculum_id,
                "target": path_id,
                "label": "scopedBy",
                "type": "structure"
            })
            
            # Add steps
            for j, step in enumerate(path.get('learning_steps', [])):
                step_id = f"step_{i}_{j}"
                
                nodes.append({
                    "id": step_id,
                    "label": step.get('title', f'Step {j+1}'),
                    "type": "LearningStep",
                    "category": "entity",
                    "color": "#96CEB4"
                })
                
                links.append({
                    "source": path_id,
                    "target": step_id,
                    "label": "hasLearningStep",
                    "type": "structure"
                })
                
                # Link step to module
                referred_module = step.get('refers_to_module')
                if referred_module:
                    module_id = node_map.get(f"module_{referred_module}")
                    if module_id:
                        links.append({
                            "source": step_id,
                            "target": module_id,
                            "label": "refersTo",
                            "type": "reference"
                        })
                
                # Link to next step
                if j < len(path.get('learning_steps', [])) - 1:
                    next_step_id = f"step_{i}_{j+1}"
                    links.append({
                        "source": step_id,
                        "target": next_step_id,
                        "label": "hasNextStep",
                        "type": "sequence"
                    })
        
        # Add learning outcomes if available
        if hasattr(extraction, 'learning_outcomes'):
            for i, outcome in enumerate(extraction.learning_outcomes):
                outcome_id = f"outcome_{i}"
                
                nodes.append({
                    "id": outcome_id,
                    "label": f"Outcome {i+1}",
                    "type": "LearningOutcome",
                    "category": "entity",
                    "color": "#DFE6E9",
                    "progressionLevel": outcome.get('progression_level', 'unknown'),
                    "supportLevel": outcome.get('support_level', 'unknown')
                })
                
                # Add progression links between outcomes
                for related in outcome.get('related_outcomes', []):
                    related_idx = related.get('outcome_id')
                    if related_idx is not None:
                        related_outcome_id = f"outcome_{related_idx}"
                        links.append({
                            "source": outcome_id,
                            "target": related_outcome_id,
                            "label": "progressesTo",
                            "type": "progression"
                        })
        
        overview = {
            "total_nodes": len(nodes),
            "total_links": len(links),
            "entities": len([n for n in nodes if n.get('category') == 'entity']),
            "relations": len(links),
            "progression_links": len([l for l in links if l.get('type') == 'progression'])
        }
        
        return {
            "nodes": nodes,
            "links": links,
            "overview": overview
        }


# GREEK CURRICULUM EXTRACTOR WITH FULL PROGRESSION
class EnhancedGreekCurriculumExtractor(ResearchPaperAlignedExtractor):
    """Extended extractor with Greek curriculum specifics AND FULL progression tracking"""
    
    def _fuzzy_match_module(self, query_title: str, candidate_titles: List[str], threshold: float = 0.6) -> Optional[str]:
        """Find best matching module title using fuzzy string matching"""
        if not query_title or not candidate_titles:
            return None
            
        best_match = None
        best_ratio = 0.0
        
        query_lower = query_title.lower().strip()
        
        for candidate in candidate_titles:
            if not candidate:
                continue
            candidate_lower = candidate.lower().strip()
            
            if query_lower == candidate_lower:
                return candidate
            
            ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()
            
            if query_lower in candidate_lower or candidate_lower in query_lower:
                ratio = max(ratio, 0.7)
            
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = candidate
        
        if best_match:
            logger.info(f"Fuzzy matched '{query_title}' -> '{best_match}' (score: {best_ratio:.2f})")
        
        return best_match
    
    def _add_module_progression_tracking(self, modules: List[Dict]) -> List[Dict]:
        """
        Add progression tracking to modules by:
        1. Identifying similar modules across grade levels
        2. Tracking complexity increases
        3. Linking prerequisite relationships
        """
        
        enhanced_modules = []
        
        # Group modules by similar titles across grades
        module_groups = self._group_similar_modules(modules)
        
        for module in modules:
            enhanced_module = module.copy()
            
            # Determine progression level based on grade
            grade = module.get('grade_level', 'unknown')
            enhanced_module['progression_level'] = self._determine_module_progression(grade)
            
            # Find prerequisite modules (same topic, lower grade)
            prerequisites = self._find_prerequisite_modules(module, modules)
            enhanced_module['prerequisite_modules'] = prerequisites
            
            # Find follow-up modules (same topic, higher grade)
            follow_ups = self._find_followup_modules(module, modules)
            enhanced_module['followup_modules'] = follow_ups
            
            # Analyze complexity progression
            complexity_indicators = self._analyze_module_complexity(module)
            enhanced_module['complexity_indicators'] = complexity_indicators
            
            enhanced_modules.append(enhanced_module)
        
        logger.info(f"✓ Added progression tracking to {len(enhanced_modules)} modules")
        return enhanced_modules

    def _group_similar_modules(self, modules: List[Dict]) -> Dict[str, List[Dict]]:
        """Group modules with similar titles/topics across grade levels"""
        
        groups = {}
        
        for i, module in enumerate(modules):
            title = nuclear_clean_text(module.get('title', '')).lower()
            topics = [nuclear_clean_text(t).lower() for t in module.get('covers_topics', [])]
            
            # Find matching group
            matched = False
            for group_key, group_modules in groups.items():
                # Check if this module is similar to any in the group
                for existing in group_modules:
                    existing_title = nuclear_clean_text(existing.get('title', '')).lower()
                    existing_topics = [nuclear_clean_text(t).lower() for t in existing.get('covers_topics', [])]
                    
                    # Calculate similarity
                    title_similarity = SequenceMatcher(None, title, existing_title).ratio()
                    topic_overlap = len(set(topics) & set(existing_topics))
                    
                    if title_similarity > 0.6 or topic_overlap >= 2:
                        groups[group_key].append(module)
                        matched = True
                        break
                
                if matched:
                    break
            
            if not matched:
                # Create new group
                groups[f"group_{i}"] = [module]
        
        return groups

    def _determine_module_progression(self, grade_level: str) -> str:
        """Determine progression level based on grade"""
        
        grade_progression = {
            'A Dimotikou': 'introduction',
            'B Dimotikou': 'introduction',
            'C Dimotikou': 'foundation',
            'D Dimotikou': 'foundation',
            'E Dimotikou': 'development',
            'ST Dimotikou': 'development',
            'A Gymnasiou': 'consolidation',
            'B Gymnasiou': 'consolidation',
            'C Gymnasiou': 'mastery',
            'A Lykeiou': 'advanced',
            'B Lykeiou': 'advanced',
            'C Lykeiou': 'specialization'
        }
        
        return grade_progression.get(grade_level, 'unknown')

    def _find_prerequisite_modules(self, module: Dict, all_modules: List[Dict]) -> List[str]:
        """Find modules that should come before this one"""
        
        prerequisites = []
        current_grade = module.get('grade_level', 'unknown')
        current_topics = set(nuclear_clean_text(t).lower() for t in module.get('covers_topics', []))
        
        # Define grade order
        grade_order = ['A Dimotikou', 'B Dimotikou', 'C Dimotikou', 'D Dimotikou', 
                       'E Dimotikou', 'ST Dimotikou', 'A Gymnasiou', 'B Gymnasiou', 
                       'C Gymnasiou', 'A Lykeiou', 'B Lykeiou', 'C Lykeiou']
        
        try:
            current_index = grade_order.index(current_grade)
        except ValueError:
            return prerequisites
        
        # Look for similar modules in earlier grades
        for other in all_modules:
            other_grade = other.get('grade_level', 'unknown')
            try:
                other_index = grade_order.index(other_grade)
                if other_index < current_index:
                    other_topics = set(nuclear_clean_text(t).lower() for t in other.get('covers_topics', []))
                    overlap = len(current_topics & other_topics)
                    
                    if overlap >= 1:  # At least one common topic
                        prerequisites.append(nuclear_clean_text(other.get('title', 'Unknown')))
            except ValueError:
                continue
        
        return prerequisites[:3]  # Limit to top 3

    def _find_followup_modules(self, module: Dict, all_modules: List[Dict]) -> List[str]:
        """Find modules that should come after this one"""
        
        followups = []
        current_grade = module.get('grade_level', 'unknown')
        current_topics = set(nuclear_clean_text(t).lower() for t in module.get('covers_topics', []))
        
        grade_order = ['A Dimotikou', 'B Dimotikou', 'C Dimotikou', 'D Dimotikou', 
                       'E Dimotikou', 'ST Dimotikou', 'A Gymnasiou', 'B Gymnasiou', 
                       'C Gymnasiou', 'A Lykeiou', 'B Lykeiou', 'C Lykeiou']
        
        try:
            current_index = grade_order.index(current_grade)
        except ValueError:
            return followups
        
        for other in all_modules:
            other_grade = other.get('grade_level', 'unknown')
            try:
                other_index = grade_order.index(other_grade)
                if other_index > current_index:
                    other_topics = set(nuclear_clean_text(t).lower() for t in other.get('covers_topics', []))
                    overlap = len(current_topics & other_topics)
                    
                    if overlap >= 1:
                        followups.append(nuclear_clean_text(other.get('title', 'Unknown')))
            except ValueError:
                continue
        
        return followups[:3]

    def _analyze_module_complexity(self, module: Dict) -> Dict[str, Any]:
        """Analyze indicators of module complexity"""
        
        complexity = {
            'cognitive_level': 'unknown',
            'independence_level': 'unknown',
            'depth_indicators': []
        }
        
        # Analyze learning objectives for cognitive complexity
        objectives = module.get('learning_objectives', [])
        if objectives:
            # Look for Bloom's taxonomy verbs
            high_order_verbs = ['analyze', 'evaluate', 'create', 'synthesize', 'αναλύω', 
                               'αξιολογώ', 'δημιουργώ', 'συνθέτω']
            mid_order_verbs = ['apply', 'demonstrate', 'use', 'εφαρμόζω', 'χρησιμοποιώ']
            
            obj_text = ' '.join([nuclear_clean_text(o) for o in objectives]).lower()
            
            if any(verb in obj_text for verb in high_order_verbs):
                complexity['cognitive_level'] = 'high'
            elif any(verb in obj_text for verb in mid_order_verbs):
                complexity['cognitive_level'] = 'medium'
            else:
                complexity['cognitive_level'] = 'foundational'
        
        # Analyze description for independence indicators
        description = nuclear_clean_text(module.get('description', '')).lower()
        
        if any(phrase in description for phrase in ['αυτόνομα', 'ανεξάρτητα', 'independently']):
            complexity['independence_level'] = 'independent'
        elif any(phrase in description for phrase in ['με υποστήριξη', 'with support', 'guided']):
            complexity['independence_level'] = 'supported'
        else:
            complexity['independence_level'] = 'scaffolded'
        
        # Count topics as depth indicator
        topic_count = len(module.get('covers_topics', []))
        if topic_count >= 5:
            complexity['depth_indicators'].append('comprehensive_coverage')
        elif topic_count >= 3:
            complexity['depth_indicators'].append('moderate_coverage')
        else:
            complexity['depth_indicators'].append('focused_coverage')
        
        return complexity

    def _add_assessment_progression(self, strategies: List[Dict]) -> List[Dict]:
        """Add progression tracking to assessment strategies"""
        
        enhanced = []
        
        # Sort by grade level
        grade_order = ['A Dimotikou', 'B Dimotikou', 'C Dimotikou', 'D Dimotikou', 
                       'E Dimotikou', 'ST Dimotikou', 'A Gymnasiou', 'B Gymnasiou', 
                       'C Gymnasiou', 'A Lykeiou', 'B Lykeiou', 'C Lykeiou']
        
        for strategy in strategies:
            enhanced_strategy = strategy.copy()
            
            # Determine progression stage
            grades = strategy.get('grade_levels', [])
            if grades and grades[0] != 'unknown':
                try:
                    min_grade_idx = min(grade_order.index(g) for g in grades if g in grade_order)
                    
                    if min_grade_idx <= 1:  # A-B Dimotikou
                        enhanced_strategy['assessment_progression'] = 'teacher_led'
                    elif min_grade_idx <= 3:  # C-D Dimotikou
                        enhanced_strategy['assessment_progression'] = 'collaborative'
                    elif min_grade_idx <= 5:  # E-ST Dimotikou
                        enhanced_strategy['assessment_progression'] = 'guided_self_assessment'
                    else:  # Gymnasio+
                        enhanced_strategy['assessment_progression'] = 'autonomous_assessment'
                except ValueError:
                    enhanced_strategy['assessment_progression'] = 'unknown'
            
            # Add scaffolding progression
            complexity = strategy.get('complexity_level', 'unknown')
            enhanced_strategy['scaffolding_progression'] = {
                'simple': 'high_scaffolding',
                'moderate': 'medium_scaffolding',
                'complex': 'low_scaffolding'
            }.get(complexity, 'unknown')
            
            enhanced.append(enhanced_strategy)
        
        logger.info(f"✓ Added progression to {len(enhanced)} assessment strategies")
        return enhanced

    def _add_teaching_progression(self, strategies: List[Dict]) -> List[Dict]:
        """Add progression tracking to teaching strategies"""
        
        enhanced = []
        
        for strategy in strategies:
            enhanced_strategy = strategy.copy()
            
            # Map scaffolding to progression
            scaffolding = strategy.get('scaffolding_type', 'unknown')
            
            progression_map = {
                'high': {
                    'stage': 'teacher_directed',
                    'student_role': 'observer_follower',
                    'teacher_role': 'demonstrator_guide'
                },
                'moderate': {
                    'stage': 'collaborative',
                    'student_role': 'active_participant',
                    'teacher_role': 'facilitator_coach'
                },
                'low': {
                    'stage': 'student_centered',
                    'student_role': 'independent_learner',
                    'teacher_role': 'mentor_advisor'
                },
                'none': {
                    'stage': 'autonomous',
                    'student_role': 'self_directed',
                    'teacher_role': 'consultant'
                }
            }
            
            enhanced_strategy['teaching_progression'] = progression_map.get(
                scaffolding, 
                {'stage': 'unknown', 'student_role': 'unknown', 'teacher_role': 'unknown'}
            )
            
            enhanced.append(enhanced_strategy)
        
        logger.info(f"✓ Added progression to {len(enhanced)} teaching strategies")
        return enhanced
    
    def extract_greek_curriculum_ontology(self, 
                                         pdf_path: Path, 
                                         mode: ExtractionMode, 
                                         provider: LLMProvider,
                                         target_personas: List[PersonaType] = None) -> EnhancedResearchExtraction:
        """Extract with research alignment AND Greek curriculum specifics AND FULL progression"""
        
        # Get base extraction
        base_extraction = self.extract_research_aligned_ontology(
            pdf_path, mode, provider, target_personas
        )
        
        # Extract text for additional processing
        text = extract_text_from_pdf(pdf_path)
        
        logger.info("Extracting Greek curriculum-specific elements with FULL progression tracking...")
        
        # Extract Greek curriculum-specific elements WITH progression
        learning_outcomes = self._extract_learning_outcomes(text, provider, base_extraction.modules)
        assessment_strategies = self._extract_assessment_strategies(text, provider)
        teaching_strategies = self._extract_teaching_strategies(text, provider)
        time_allocations = self._extract_time_allocations(text, provider)
        thematic_cycles = self._extract_thematic_cycles(text, provider)
        pedagogical_framework = self._extract_pedagogical_framework(text, provider)
        
        # ADD PROGRESSION TO ALL ELEMENTS
        logger.info("Adding progression tracking to modules...")
        enhanced_modules = self._add_module_progression_tracking(base_extraction.modules)
        
        logger.info("Adding progression tracking to assessments...")
        enhanced_assessments = self._add_assessment_progression(assessment_strategies)
        
        logger.info("Adding progression tracking to teaching strategies...")
        enhanced_teaching = self._add_teaching_progression(teaching_strategies)
        
        # Enhance modules with outcomes
        final_modules = self._enhance_modules_with_outcomes(
            enhanced_modules, 
            learning_outcomes,
            time_allocations
        )
        
        return EnhancedResearchExtraction(
            curriculum=base_extraction.curriculum,
            learning_paths=base_extraction.learning_paths,
            modules=final_modules,
            personas=base_extraction.personas,
            events=base_extraction.events,
            competency_validation=base_extraction.competency_validation,
            axiom_compliance=base_extraction.axiom_compliance,
            learning_outcomes=learning_outcomes,
            assessment_strategies=enhanced_assessments,
            teaching_strategies=enhanced_teaching,
            time_allocations=time_allocations,
            thematic_cycles=thematic_cycles,
            pedagogical_framework=pedagogical_framework
        )

    def _extract_learning_outcomes(self, text: str, provider: LLMProvider, 
                                   modules: List[Dict]) -> List[Dict[str, Any]]:
        """Extract learning outcomes with PROGRESSION tracking - MAXIMUM EXTRACTION"""
        
        # Use FULL text, not just sample - like original
        prompt = f"""Extract ALL learning outcomes from this Greek curriculum document. Extract EVERYTHING you find.

FULL CURRICULUM TEXT:
{text}

Find ALL outcomes in the document. Look in EVERY section:
- Μαθησιακά αποτελέσματα sections
- Στόχοι (goals/objectives)
- Επιδιώξεις (pursuits/aims)  
- Any bulleted or numbered lists of what students learn
- Competency descriptions
- Skill descriptions

Extract EVERY SINGLE outcome, even if there are 50, 100, or 200 outcomes.

For each outcome:
1. Copy the EXACT Greek text
2. Identify grade levels (look for Α΄, Β΄, Γ΄, Δ΄, Ε΄, ΣΤ΄, Γυμνασίου, etc.)
3. Detect support level (teacher support keywords)
4. Classify skill type

OUTPUT JSON ARRAY with EVERY outcome:
[
  {{
    "outcome_text": "Exact Greek text - keep it complete",
    "grade_levels": ["A Dimotikou", "B Dimotikou"],
    "progression_level": "beginning|developing|proficient|advanced|unknown",
    "support_level": "high_support|moderate_support|independent|unknown",
    "bloom_level": "remember|understand|apply|analyze|evaluate|create",
    "skill_category": "listening|reading|speaking|writing|literature|language|general"
  }}
]

GRADE MAPPING (be flexible, look for these patterns):
- Α΄ και Β΄ / Α & Β / Α΄-Β΄ → ["A Dimotikou", "B Dimotikou"]
- Γ΄ και Δ΄ / Γ & Δ / Γ΄-Δ΄ → ["C Dimotikou", "D Dimotikou"]
- Ε΄ και ΣΤ΄ / Ε & ΣΤ / Ε΄-ΣΤ΄ → ["E Dimotikou", "ST Dimotikou"]
- ΣΤ΄ τάξη / 6η τάξη → ["ST Dimotikou"]
- Α΄ Γυμνασίου → ["A Gymnasiou"]
- Β΄ Γυμνασίου → ["B Gymnasiou"]
- Γ΄ Γυμνασίου → ["C Gymnasiou"]
- If no grade mentioned → ["unknown"]

SUPPORT LEVEL (look for these Greek phrases):
- "με την ενισχυμένη υποστήριξη", "με την καθοδήγηση" → high_support
- "με την διαμεσολάβηση", "με βοήθεια" → moderate_support  
- "αυτόνομα", "ανεξάρτητα", "μόνοι τους" → independent
- Not mentioned → unknown

PROGRESSION LEVEL:
- Α΄-Β΄ grades → beginning
- Γ΄-Δ΄ grades → developing
- Ε΄-ΣΤ΄ grades → proficient
- Γυμνάσιο → advanced
- Unknown grade → unknown

BLOOM LEVEL (based on Greek verbs):
- "αναγνωρίζω", "ονομάζω", "θυμάμαι" → remember
- "κατανοώ", "εξηγώ", "περιγράφω" → understand
- "εφαρμόζω", "χρησιμοποιώ", "λύνω" → apply
- "αναλύω", "συγκρίνω", "διακρίνω" → analyze
- "αξιολογώ", "κρίνω", "επιλέγω" → evaluate
- "δημιουργώ", "σχεδιάζω", "συνθέτω" → create
- Default → understand

CRITICAL INSTRUCTIONS:
1. DO NOT SKIP ANYTHING - extract every outcome you find
2. DO NOT limit yourself - if there are 100 outcomes, extract all 100
3. DO NOT summarize - extract each outcome separately
4. Keep Greek text EXACTLY as written
5. Be thorough - read the ENTIRE text
6. Return [] ONLY if absolutely no outcomes exist

Return ONLY the JSON array, nothing else."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            outcomes = self._parse_json_response(response, fallback_data=[])
            
            if not isinstance(outcomes, list):
                logger.error(f"LLM returned non-list: {type(outcomes)}")
                return []
            
            logger.info(f"Raw extraction: {len(outcomes)} learning outcomes")
            
            # Post-process: Link related outcomes across progression
            outcomes = self._link_progressive_outcomes(outcomes)
            
            # Validate and add defaults
            valid_outcomes = []
            for outcome in outcomes:
                if not isinstance(outcome, dict) or not outcome.get('outcome_text'):
                    continue
                
                # Clean the outcome text
                outcome['outcome_text'] = nuclear_clean_text(outcome['outcome_text'])
                
                outcome.setdefault('grade_levels', ['unknown'])
                outcome.setdefault('progression_level', 'unknown')
                outcome.setdefault('support_level', 'unknown')
                outcome.setdefault('bloom_level', 'understand')
                outcome.setdefault('skill_category', 'general')
                outcome.setdefault('related_module', 'General')
                outcome.setdefault('measurable', True)
                outcome.setdefault('related_outcomes', [])
                
                valid_outcomes.append(outcome)
            
            logger.info(f"✓ Extracted {len(valid_outcomes)} learning outcomes with progression tracking")
            
            # If we got very few outcomes, log a warning
            if len(valid_outcomes) < 5:
                logger.warning(f"Only extracted {len(valid_outcomes)} outcomes - this seems low. Document may not contain explicit outcomes.")
            
            return valid_outcomes
            
        except Exception as e:
            logger.error(f"Learning outcomes extraction failed: {e}")
            return []

    def _link_progressive_outcomes(self, outcomes: List[Dict]) -> List[Dict]:
        """Link outcomes that show progression of the same skill across grades"""
        
        for i, outcome in enumerate(outcomes):
            skill = outcome.get('skill_category', 'general')
            text = nuclear_clean_text(outcome.get('outcome_text', '')).lower()
            
            key_words = set([
                word for word in text.split()
                if len(word) > 4 and word not in {'είναι', 'έχει', 'μπορεί', 'πρέπει', 'γίνει'}
            ])
            
            if not key_words:
                continue
            
            for j, other in enumerate(outcomes):
                if i >= j:
                    continue
                
                if other.get('skill_category') != skill:
                    continue
                
                other_text = nuclear_clean_text(other.get('outcome_text', '')).lower()
                other_words = set([
                    word for word in other_text.split()
                    if len(word) > 4
                ])
                
                overlap = len(key_words & other_words)
                if overlap >= 2:
                    outcome.setdefault('related_outcomes', [])
                    other.setdefault('related_outcomes', [])
                    
                    outcome['related_outcomes'].append({
                        'outcome_id': j,
                        'grade_levels': other.get('grade_levels', []),
                        'progression_level': other.get('progression_level', 'unknown'),
                        'relationship': 'progression'
                    })
                    
                    other['related_outcomes'].append({
                        'outcome_id': i,
                        'grade_levels': outcome.get('grade_levels', []),
                        'progression_level': outcome.get('progression_level', 'unknown'),
                        'relationship': 'progression'
                    })
        
        return outcomes

    def _extract_assessment_strategies(self, text: str, provider: LLMProvider) -> List[Dict[str, Any]]:
        """Extract assessment strategies with PROGRESSION - IMPROVED"""
        
        text_sample = text[:15000] if len(text) > 15000 else text
        
        prompt = f"""Extract ALL assessment strategies from this Greek curriculum. Be COMPREHENSIVE.

FULL TEXT:
{text_sample}

Look for ANY mentions of:
- Assessment methods (μέθοδοι αξιολόγησης)
- Evaluation approaches (αξιολόγηση)
- Testing strategies (στρατηγικές ελέγχου)
- Self-assessment (αυτοαξιολόγηση)
- Peer assessment (ετεροαξιολόγηση)
- Portfolio assessment (φάκελος εργασιών)
- Formative assessment (διαμορφωτική)
- Summative assessment (τελική)
- Rubrics (κριτήρια αξιολόγησης)

Extract EVERY strategy mentioned, even if brief. Don't be selective.

OUTPUT - Return JSON array with ALL strategies found:
[
  {{
    "strategy_type": "formative|summative|diagnostic|self|peer|portfolio|rubric|observation",
    "greek_term": "Original Greek term from text",
    "description": "Description from text",
    "grade_levels": ["A Dimotikou", "B Dimotikou"],
    "frequency": "continuous|periodic|end_of_unit|unknown",
    "complexity_level": "simple|moderate|complex|unknown",
    "progression_notes": "How assessment evolves across grades (if mentioned)"
  }}
]

COMPLEXITY INDICATORS:
- simple: Basic yes/no, teacher-only, binary criteria
- moderate: Multiple criteria, checklists, beginning student involvement
- complex: Detailed rubrics, self & peer assessment, reflection, portfolios

GRADE DETECTION:
- Look for grade mentions near assessment descriptions
- If assessment applies to multiple grades, list all
- If unclear → ["unknown"]

CRITICAL:
1. Extract ALL strategies - aim for 10+ if document is comprehensive
2. Include even brief mentions
3. Preserve Greek terminology exactly
4. Return [] only if truly no assessment info exists
5. Do NOT limit yourself

Return ONLY valid JSON array."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            strategies = self._parse_json_response(response, fallback_data=[])
            
            if not isinstance(strategies, list):
                return []
            
            logger.info(f"Raw extraction: {len(strategies)} assessment strategies")
            
            for strategy in strategies:
                strategy.setdefault('tools', [])
                strategy.setdefault('applies_to', [])
                strategy.setdefault('grade_levels', [])
                strategy.setdefault('complexity_level', 'unknown')
                strategy.setdefault('progression_notes', '')
            
            logger.info(f"✓ Extracted {len(strategies)} assessment strategies with progression")
            return strategies
            
        except Exception as e:
            logger.error(f"Assessment extraction failed: {e}")
            return []

    def _extract_teaching_strategies(self, text: str, provider: LLMProvider) -> List[Dict[str, Any]]:
        """Extract teaching strategies with PROGRESSION - IMPROVED"""
        
        text_sample = text[:15000] if len(text) > 15000 else text
        
        prompt = f"""Extract ALL teaching strategies from this Greek curriculum. Be THOROUGH and COMPREHENSIVE.

FULL TEXT:
{text_sample}

Look for ANY mentions of:
- Teaching methods (διδακτικές μέθοδοι)
- Pedagogical approaches (παιδαγωγικές προσεγγίσεις)
- Instructional strategies (στρατηγικές διδασκαλίας)
- Learning activities (μαθησιακές δραστηριότητες)
- Cooperative learning (συνεργατική μάθηση)
- Project-based learning (μάθηση με projects)
- Differentiated instruction (διαφοροποιημένη διδασκαλία)
- Scaffolding techniques (υποστήριξη)
- Direct instruction (άμεση διδασκαλία)
- Inquiry-based learning (ερευνητική μάθηση)

Extract EVERY strategy mentioned. Include:
- General approaches
- Specific techniques
- Activity types
- Support mechanisms

OUTPUT - Return JSON array with ALL strategies:
[
  {{
    "strategy_name": "Name or description from document",
    "greek_term": "Original Greek term",
    "category": "cognitive|metacognitive|social|memory|affective|organizational|general",
    "description": "What the document says about it",
    "grade_levels": ["A Dimotikou", "B Dimotikou"],
    "progression_notes": "How strategy changes across grades (if mentioned)",
    "scaffolding_type": "high|moderate|low|none|unknown"
  }}
]

CATEGORY GUIDE:
- cognitive: Thinking, reasoning, problem-solving strategies
- metacognitive: Thinking about thinking, self-monitoring
- social: Group work, collaboration, peer learning
- memory: Memorization, recall techniques
- affective: Motivation, engagement strategies
- organizational: Planning, time management
- general: Doesn't fit other categories

SCAFFOLDING DETECTION:
- high: Heavy teacher guidance, modeling, step-by-step
- moderate: Teacher facilitation, partial support
- low: Minimal guidance, student autonomy
- none: Full independence
- unknown: Not specified

CRITICAL:
1. Extract EVERYTHING - aim for 15+ strategies if comprehensive document
2. Include both explicit strategies and implicit approaches
3. Don't skip brief mentions
4. Preserve Greek terms exactly
5. Return [] only if truly nothing found

Return ONLY valid JSON array."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            strategies = self._parse_json_response(response, fallback_data=[])
            
            if not isinstance(strategies, list):
                return []
            
            logger.info(f"Raw extraction: {len(strategies)} teaching strategies")
            
            for strategy in strategies:
                strategy.setdefault('techniques', [])
                strategy.setdefault('applies_to_grades', strategy.get('grade_levels', []))
                strategy.setdefault('progression_notes', '')
                strategy.setdefault('scaffolding_type', 'unknown')
            
            logger.info(f"✓ Extracted {len(strategies)} teaching strategies with progression")
            return strategies
            
        except Exception as e:
            logger.error(f"Teaching strategies extraction failed: {e}")
            return []

    def _extract_time_allocations(self, text: str, provider: LLMProvider) -> Dict[str, Any]:
        """Extract time allocations"""
        
        prompt = f"""Extract time allocation information from this curriculum document.

FULL TEXT:
{text[:10000]}

LOOK FOR:
- Weekly hours (εβδομαδιαίες ώρες)
- Hours per subject/grade (ώρες ανά μάθημα/τάξη)
- Time distribution
- Transition periods (περίοδοι μετάβασης)

OUTPUT - Return ONLY this JSON:
{{
"allocations": [
    {{
    "grade": "Grade identifier from text",
    "total_hours": <number>,
    "breakdown": {{"subject_name": <hours>}}
    }}
],
"transition_periods": [
    {{
    "name": "Period name",
    "weeks": <number>,
    "purpose": "Description"
    }}
],
"notes": "Any relevant notes"
}}

Return {{"allocations": [], "transition_periods": [], "notes": ""}} if none found.
ONLY JSON."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            result = self._parse_json_response(response, fallback_data={
                "allocations": [],
                "transition_periods": [],
                "notes": ""
            })
            
            weekly_hours = {}
            for allocation in result.get('allocations', []):
                grade = allocation.get('grade', 'unknown')
                weekly_hours[grade] = {
                    'total': allocation.get('total_hours', 0),
                    'subject_hours': allocation.get('breakdown', {})
                }
            
            transitions = {}
            for period in result.get('transition_periods', []):
                name = period.get('name', 'unknown')
                transitions[name] = {
                    'weeks': period.get('weeks', 0),
                    'purpose': period.get('purpose', '')
                }
            
            final = {
                'weekly_hours_by_grade': weekly_hours,
                'transition_periods': transitions,
                'notes': result.get('notes', '')
            }
            
            logger.info(f"✓ Extracted time allocations for {len(weekly_hours)} grades")
            return final
            
        except Exception as e:
            logger.error(f"Time allocation extraction failed: {e}")
            return {'weekly_hours_by_grade': {}, 'transition_periods': {}}

    def _extract_thematic_cycles(self, text: str, provider: LLMProvider) -> List[str]:
        """Extract thematic cycles"""
        
        prompt = f"""Extract thematic areas/cycles from this curriculum document.

FULL TEXT:
{text[:10000]}

LOOK FOR:
- Thematic cycles (θεματικοί κύκλοι)
- Thematic fields (θεματικά πεδία)
- Content areas (περιοχές περιεχομένου)
- Main themes/topics

OUTPUT - Return ONLY a JSON array of strings:
["theme 1 in Greek", "theme 2 in Greek", "theme 3 in Greek"]

Extract EXACT phrases from the document.
Return [] if none found.
ONLY JSON."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            cycles = self._parse_json_response(response, fallback_data=[])
            
            if not isinstance(cycles, list):
                return []
            
            unique_cycles = []
            seen = set()
            for cycle in cycles:
                if isinstance(cycle, str) and cycle.strip():
                    clean = nuclear_clean_text(cycle).strip()
                    if clean.lower() not in seen:
                        unique_cycles.append(clean)
                        seen.add(clean.lower())
            
            logger.info(f"✓ Extracted {len(unique_cycles)} thematic cycles")
            return unique_cycles
            
        except Exception as e:
            logger.error(f"Thematic cycles extraction failed: {e}")
            return []

    def _extract_pedagogical_framework(self, text: str, provider: LLMProvider) -> Dict[str, Any]:
        """Extract pedagogical framework"""
        
        prompt = f"""Extract the pedagogical framework from this curriculum document.

FULL TEXT:
{text[:10000]}

LOOK FOR:
- Teaching approaches/philosophies
- Learning principles
- Methodological frameworks
- Student-centered elements
- Collaborative learning mentions
- Differentiation strategies
- Technology integration

OUTPUT - Return ONLY this JSON:
{{
"main_approaches": ["approach 1", "approach 2"],
"key_principles": [
    {{"principle": "name", "description": "brief desc"}}
],
"student_centered": true|false,
"collaborative_learning": true|false,
"differentiation": true|false,
"technology_integration": "none|low|medium|high",
"cross_curricular": true|false
}}

Set booleans to true only if explicitly mentioned.
ONLY JSON."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            framework = self._parse_json_response(response, fallback_data={})
            
            framework.setdefault('main_approaches', [])
            framework.setdefault('key_principles', [])
            framework.setdefault('student_centered', False)
            framework.setdefault('collaborative_learning', False)
            framework.setdefault('differentiation', False)
            framework.setdefault('technology_integration', 'unknown')
            framework.setdefault('cross_curricular', False)
            framework.setdefault('digital_integration', {
                'level': framework.get('technology_integration', 'unknown'),
                'tools': []
            })
            
            logger.info(f"✓ Extracted pedagogical framework with {len(framework['main_approaches'])} approaches")
            return framework
            
        except Exception as e:
            logger.error(f"Pedagogical framework extraction failed: {e}")
            return {
                'main_approaches': [],
                'key_principles': [],
                'student_centered': False
            }

    def _enhance_modules_with_outcomes(self, 
                                    modules: List[Dict], 
                                    learning_outcomes: List[Dict],
                                    time_allocations: Dict) -> List[Dict]:
        """Enhance modules with outcomes using fuzzy matching"""
        
        enhanced = []
        module_titles = [m.get('title', '') for m in modules]
        
        for module in modules:
            enhanced_module = module.copy()
            
            module_title = module.get('title', '')
            related_outcomes = []
            
            for outcome in learning_outcomes:
                outcome_module = outcome.get('related_module', '')
                
                if outcome_module == module_title:
                    related_outcomes.append(outcome)
                elif outcome.get('fuzzy_matched') and outcome_module == module_title:
                    related_outcomes.append(outcome)
            
            if related_outcomes:
                enhanced_module['learning_outcomes_detailed'] = related_outcomes
                logger.info(f"✓ Linked {len(related_outcomes)} outcomes to module: {module_title}")
            
            grade = module.get('grade_level', '')
            if grade and time_allocations.get('weekly_hours_by_grade'):
                grade_data = time_allocations['weekly_hours_by_grade'].get(grade, {})
                if grade_data and grade_data.get('total'):
                    enhanced_module['allocated_weekly_hours'] = grade_data.get('total')
                    enhanced_module['subject_specific_hours'] = grade_data.get('subject_hours', {})
            
            enhanced.append(enhanced_module)
        
        return enhanced

    def generate_enhanced_greek_turtle(self, extraction: EnhancedResearchExtraction) -> str:
        """Generate Turtle RDF with research compliance AND Greek curriculum specifics AND progression"""
        
        base_turtle = self.generate_research_compliant_turtle(extraction)
        
        additional_triples = []
        
        additional_triples.append("\n# Greek Curriculum Learning Outcomes with Progression")
        for i, outcome in enumerate(extraction.learning_outcomes, 1):
            outcome_uri = f"currkg:LearningOutcome_{i}"
            additional_triples.extend([
                f"{outcome_uri} a currkg:LearningOutcome ;",
                f'    currkg:hasText "{safe_escape_for_turtle(outcome.get("outcome_text", ""))}" ;',
                f'    currkg:bloomLevel currkg:{safe_escape_for_turtle(outcome.get("bloom_level", "understand")).title()} ;',
                f'    currkg:skillCategory currkg:{safe_escape_for_turtle(outcome.get("skill_category", "general")).title()} ;',
                f'    currkg:progressionLevel currkg:{safe_escape_for_turtle(outcome.get("progression_level", "unknown")).title()} ;',
                f'    currkg:supportLevel currkg:{safe_escape_for_turtle(outcome.get("support_level", "unknown")).title()} ;'
            ])
            
            for grade in outcome.get('grade_levels', []):
                if grade != 'unknown':
                    safe_grade = safe_escape_for_turtle(grade).replace(' ', '_')
                    additional_triples.append(f"    currkg:applicableToGrade currkg:{safe_grade} ;")
            
            for related in outcome.get('related_outcomes', []):
                related_id = related.get('outcome_id')
                if related_id is not None:
                    related_uri = f"currkg:LearningOutcome_{related_id + 1}"
                    additional_triples.append(f"    currkg:progressesTo {related_uri} ;")
            
            additional_triples[-1] = additional_triples[-1].rstrip(' ;') + ' .'
            additional_triples.append("")
        
        additional_triples.append("# Module Progression Relationships")
        for i, module in enumerate(extraction.modules, 1):
            module_uri = f"currkg:Module_{i}"
            
            progression_level = module.get('progression_level', 'unknown')
            if progression_level != 'unknown':
                safe_prog = safe_escape_for_turtle(progression_level).title()
                additional_triples.append(f"{module_uri} currkg:hasProgressionLevel currkg:{safe_prog} .")
            
            for prereq_title in module.get('prerequisite_modules', []):
                for j, other_module in enumerate(extraction.modules, 1):
                    if nuclear_clean_text(other_module.get('title')) == nuclear_clean_text(prereq_title):
                        prereq_uri = f"currkg:Module_{j}"
                        additional_triples.append(f"{module_uri} currkg:hasPrerequisite {prereq_uri} .")
                        break
            
            complexity = module.get('complexity_indicators', {})
            if complexity.get('cognitive_level') != 'unknown':
                cog_level = safe_escape_for_turtle(complexity['cognitive_level']).title()
                additional_triples.append(f"{module_uri} currkg:cognitiveLevel currkg:{cog_level} .")
            if complexity.get('independence_level') != 'unknown':
                ind_level = safe_escape_for_turtle(complexity['independence_level']).title()
                additional_triples.append(f"{module_uri} currkg:independenceLevel currkg:{ind_level} .")
        
        additional_triples.append("")
        
        additional_triples.append("# Assessment Strategies with Complexity Progression")
        for i, strategy in enumerate(extraction.assessment_strategies, 1):
            strategy_uri = f"currkg:AssessmentStrategy_{i}"
            stype = safe_escape_for_turtle(strategy.get('strategy_type', 'formative')).title()
            complexity_level = safe_escape_for_turtle(strategy.get('complexity_level', 'unknown')).title()
            assessment_prog = safe_escape_for_turtle(strategy.get('assessment_progression', 'unknown')).title()
            
            additional_triples.extend([
                f"{strategy_uri} a currkg:AssessmentStrategy ;",
                f'    currkg:strategyType currkg:{stype} ;',
                f'    currkg:greekTerm "{safe_escape_for_turtle(strategy.get("greek_term", ""))}" ;',
                f'    currkg:complexityLevel currkg:{complexity_level} ;',
                f'    currkg:assessmentProgression currkg:{assessment_prog} ;'
            ])
            
            if strategy.get('progression_notes'):
                notes = safe_escape_for_turtle(strategy['progression_notes'])
                additional_triples.append(f'    currkg:progressionNotes "{notes}" ;')
            
            for grade in strategy.get('grade_levels', []):
                if grade != 'unknown':
                    safe_grade = safe_escape_for_turtle(grade).replace(' ', '_')
                    additional_triples.append(f"    currkg:applicableToGrade currkg:{safe_grade} ;")
            
            additional_triples[-1] = additional_triples[-1].rstrip(' ;') + ' .'
            additional_triples.append("")
        
        additional_triples.append("# Teaching Strategies with Scaffolding Progression")
        for i, strategy in enumerate(extraction.teaching_strategies, 1):
            strategy_uri = f"currkg:TeachingStrategy_{i}"
            teaching_prog = strategy.get('teaching_progression', {})
            scaffolding = safe_escape_for_turtle(strategy.get('scaffolding_type', 'unknown')).title()
            stage = safe_escape_for_turtle(teaching_prog.get('stage', 'unknown')).title()
            student_role = safe_escape_for_turtle(teaching_prog.get('student_role', 'unknown')).title()
            teacher_role = safe_escape_for_turtle(teaching_prog.get('teacher_role', 'unknown')).title()
            
            additional_triples.extend([
                f"{strategy_uri} a currkg:TeachingStrategy ;",
                f'    currkg:strategyName "{safe_escape_for_turtle(strategy.get("strategy_name", ""))}" ;',
                f'    currkg:scaffoldingType currkg:{scaffolding} ;',
                f'    currkg:teachingStage currkg:{stage} ;',
                f'    currkg:studentRole currkg:{student_role} ;',
                f'    currkg:teacherRole currkg:{teacher_role} ;'
            ])
            
            if strategy.get('progression_notes'):
                notes = safe_escape_for_turtle(strategy['progression_notes'])
                additional_triples.append(f'    currkg:progressionNotes "{notes}" ;')
            
            for grade in strategy.get('grade_levels', []):
                if grade != 'unknown':
                    safe_grade = safe_escape_for_turtle(grade).replace(' ', '_')
                    additional_triples.append(f"    currkg:applicableToGrade currkg:{safe_grade} ;")
            
            additional_triples[-1] = additional_triples[-1].rstrip(' ;') + ' .'
            additional_triples.append("")
        
        if extraction.time_allocations.get('weekly_hours_by_grade'):
            additional_triples.append("# Time Allocations per Grade")
            for grade, hours in extraction.time_allocations['weekly_hours_by_grade'].items():
                safe_grade = safe_escape_for_turtle(grade).replace(' ', '_')
                allocation_uri = f"currkg:TimeAllocation_{safe_grade}"
                additional_triples.extend([
                    f"{allocation_uri} a currkg:TimeAllocation ;",
                    f"    currkg:forGradeLevel currkg:{safe_grade} ;",
                    f"    currkg:totalWeeklyHours {hours.get('total', 0)} ;"
                ])
                
                if hours.get('subject_hours'):
                    for subject, subj_hours in hours['subject_hours'].items():
                        safe_subject = safe_escape_for_turtle(subject).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                        additional_triples.append(f"    currkg:{safe_subject}Hours {subj_hours} ;")
                
                additional_triples[-1] = additional_triples[-1].rstrip(' ;') + ' .'
                additional_triples.append("")
        
        if extraction.thematic_cycles:
            additional_triples.append("# Thematic Cycles")
            for i, cycle in enumerate(extraction.thematic_cycles, 1):
                cycle_uri = f"currkg:ThematicCycle_{i}"
                additional_triples.extend([
                    f"{cycle_uri} a currkg:ThematicCycle ;",
                    f'    currkg:title "{safe_escape_for_turtle(cycle)}" ;',
                    "    ."
                ])
                additional_triples.append("")
        
        if extraction.pedagogical_framework:
            additional_triples.append("# Pedagogical Framework")
            additional_triples.append("currkg:PedagogicalFramework a currkg:Framework ;")
            
            for approach in extraction.pedagogical_framework.get('main_approaches', []):
                safe_approach = safe_escape_for_turtle(approach).title().replace('_', '').replace(' ', '').replace('-', '')
                additional_triples.append(f"    currkg:usesApproach currkg:{safe_approach}Approach ;")
            
            if extraction.pedagogical_framework.get('student_centered'):
                additional_triples.append("    currkg:isStudentCentered true ;")
            
            if extraction.pedagogical_framework.get('collaborative_learning'):
                additional_triples.append("    currkg:supportsCollaborativeLearning true ;")
            
            additional_triples[-1] = additional_triples[-1].rstrip(' ;') + ' .'
            additional_triples.append("")
        
        module_progressions = sum(len(m.get('prerequisite_modules', [])) for m in extraction.modules)
        outcome_progressions = sum(len(o.get('related_outcomes', [])) for o in extraction.learning_outcomes)
        
        additional_triples.extend([
            "# Greek Curriculum Enhanced Metadata with FULL Progression",
            "currkg:GreekCurriculumMetadata a currkg:EnhancementMetadata ;",
            f'    currkg:extractionDate "{datetime.now().isoformat()}"^^xsd:dateTime ;',
            f'    currkg:learningOutcomesCount {len(extraction.learning_outcomes)} ;',
            f'    currkg:assessmentStrategiesCount {len(extraction.assessment_strategies)} ;',
            f'    currkg:teachingStrategiesCount {len(extraction.teaching_strategies)} ;',
            f'    currkg:moduleProgressionLinks {module_progressions} ;',
            f'    currkg:outcomeProgressionLinks {outcome_progressions} ;',
            f'    currkg:totalProgressionLinks {module_progressions + outcome_progressions} ;',
            f'    currkg:progressionTrackingEnabled true ;',
            "    ."
        ])
        
        output = base_turtle + "\n" + "\n".join(additional_triples)
        
        # FINAL NUCLEAR CHECK
        if '^b' in output:
            logger.error("CRITICAL: Found ^b in final output!")
            output = nuclear_clean_text(output)
        
        return output

    def save_enhanced_greek_ontology(self, extraction: EnhancedResearchExtraction, output_path: Path):
        """Save ontology with research compliance AND Greek curriculum AND FULL progression"""
        
        turtle_content = self.generate_enhanced_greek_turtle(extraction)
        
        # NUCLEAR CLEAN
        turtle_content = nuclear_clean_text(turtle_content)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write as pure UTF-8
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(turtle_content)
        
        # Verify
        with open(output_path, 'r', encoding='utf-8') as f:
            verify = f.read()
            if '^b' in verify:
                logger.error("STILL HAS ^b AFTER WRITE!")
                fixed = nuclear_clean_text(verify)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(fixed)
                logger.info("Applied emergency fix")
            else:
                logger.info("✓ File clean - no ^b' found")
        
        module_progressions = sum(len(m.get('prerequisite_modules', [])) for m in extraction.modules)
        outcome_progressions = sum(len(o.get('related_outcomes', [])) for o in extraction.learning_outcomes)
        total_progressions = module_progressions + outcome_progressions
        
        validation_report = {
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "research_paper_aligned": True,
                "greek_curriculum_enhanced": True,
                "full_progression_tracking": True,
                "ontology_file": str(output_path)
            },
            "curriculum_stats": {
                "modules_count": len(extraction.modules),
                "learning_paths_count": len(extraction.learning_paths),
                "learning_outcomes_count": len(extraction.learning_outcomes),
                "assessment_strategies_count": len(extraction.assessment_strategies),
                "teaching_strategies_count": len(extraction.teaching_strategies)
            },
            "progression_analysis": {
                "module_progressions": module_progressions,
                "outcome_progressions": outcome_progressions,
                "total_progression_links": total_progressions,
                "modules_with_prerequisites": sum(1 for m in extraction.modules if m.get('prerequisite_modules')),
                "modules_with_followups": sum(1 for m in extraction.modules if m.get('followup_modules')),
                "outcomes_with_progression": sum(
                    1 for o in extraction.learning_outcomes 
                    if o.get('related_outcomes')
                ),
                "support_levels": {
                    level: sum(
                        1 for o in extraction.learning_outcomes 
                        if o.get('support_level') == level
                    )
                    for level in ['high_support', 'moderate_support', 'independent', 'unknown']
                },
                "progression_levels": {
                    level: sum(
                        1 for o in extraction.learning_outcomes 
                        if o.get('progression_level') == level
                    )
                    for level in ['beginning', 'developing', 'proficient', 'advanced', 'unknown']
                },
                "module_progression_levels": {
                    level: sum(
                        1 for m in extraction.modules
                        if m.get('progression_level') == level
                    )
                    for level in ['introduction', 'foundation', 'development', 'consolidation', 'mastery', 'advanced', 'specialization', 'unknown']
                }
            }
        }
        
        report_path = output_path.with_suffix('.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved ENHANCED Greek curriculum ontology with FULL PROGRESSION to {output_path}")
        logger.info(f"  - Learning Outcomes: {len(extraction.learning_outcomes)}")
        logger.info(f"  - Module Progression Links: {module_progressions}")
        logger.info(f"  - Outcome Progression Links: {outcome_progressions}")
        logger.info(f"  - TOTAL Progression Links: {total_progressions}")
        logger.info(f"  - Assessment Strategies: {len(extraction.assessment_strategies)}")
        logger.info(f"  - Teaching Strategies: {len(extraction.teaching_strategies)}")
        
        return output_path


# MAIN EXTRACTOR CLASS FOR FASTAPI INTEGRATION
class EnhancedCurriculumOntologyExtractor:
    """Main extractor with FULL progression tracking - FIXED VERSION"""
    
    def __init__(self):
        self.greek_extractor = EnhancedGreekCurriculumExtractor()
        self.llm_service = self.greek_extractor.llm_service
    
    def setup_llm(self, provider, api_key: str, model_name: Optional[str] = None):
        """Setup LLM provider"""
        self.greek_extractor.setup_llm(provider, api_key, model_name)
    
    def extract_from_pdf(self, 
                        pdf_path: Path, 
                        mode, 
                        provider,
                        research_aligned: bool = True,
                        greek_curriculum: bool = True) -> Dict[str, Any]:
        """Extract with FULL progression tracking"""
        
        if research_aligned and greek_curriculum:
            extraction = self.greek_extractor.extract_greek_curriculum_ontology(
                pdf_path, mode, provider
            )
            
            turtle_output = self.greek_extractor.generate_enhanced_greek_turtle(extraction)
            json_graph = self.greek_extractor._convert_extraction_to_graph(extraction)
            
            module_progressions = sum(
                len(m.get('prerequisite_modules', [])) 
                for m in extraction.modules
            )
            outcome_progressions = sum(
                len(outcome.get('related_outcomes', [])) 
                for outcome in extraction.learning_outcomes
            )
            
            return {
                'mode': mode.value,
                'provider': provider.value,
                'filename': pdf_path.name,
                'research_aligned': True,
                'greek_curriculum_enhanced': True,
                'full_progression_tracking': True,
                'turtle_output': turtle_output,
                'json_graph': json_graph,
                'extraction_data': {
                    'curriculum': extraction.curriculum,
                    'modules': extraction.modules,
                    'learning_paths': extraction.learning_paths,
                    'learning_outcomes': extraction.learning_outcomes,
                    'assessment_strategies': extraction.assessment_strategies,
                    'teaching_strategies': extraction.teaching_strategies
                },
                'extracted_elements': {
                    'Curriculum': 1,
                    'Modules': len(extraction.modules),
                    'LearningPaths': len(extraction.learning_paths),
                    'LearningOutcomes': len(extraction.learning_outcomes),
                    'ProgressionLinks': module_progressions + outcome_progressions
                },
                'progression_analysis': {
                    'total_progression_links': module_progressions + outcome_progressions,
                    'module_progression_links': module_progressions,
                    'outcome_progression_links': outcome_progressions
                }
            }
        
        else:
            extraction = self.greek_extractor.extract_research_aligned_ontology(
                pdf_path, mode, provider
            )
            
            turtle_output = self.greek_extractor.generate_research_compliant_turtle(extraction)
            json_graph = self.greek_extractor._convert_extraction_to_graph(extraction)
            
            return {
                'mode': mode.value,
                'provider': provider.value,
                'filename': pdf_path.name,
                'research_aligned': True,
                'turtle_output': turtle_output,
                'json_graph': json_graph
            }