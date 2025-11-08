from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re
import json
from rdflib import Graph, Namespace, URIRef, Literal

from app.services.llm_service import MultiLLMService, LLMProvider
from app.services.rag_service import rag_service
from app.services.knowledge_enhancer import knowledge_enhancer
from app.utils.file_handler import extract_text_from_pdf
from app.core.config import ExtractionMode

logger = logging.getLogger(__name__)


class CurriculumOntologyExtractor:
    """Complete ontology extractor with enhanced graph visualization support."""

    def __init__(self):
        self.llm_service = MultiLLMService()
        self._initialize_services()

    def _initialize_services(self):
        """Initialize RAG and knowledge enhancement services"""
        try:
            rag_service.initialize()
            rag_service.build_curriculum_database()
            rag_service.debug_database_contents()  # Add this
            logger.info("RAG service initialized")
            logger.info("Knowledge enhancer ready")
        except Exception as e:
            logger.warning(f"Service initialization failed: {e}")

    def setup_llm(self, provider: LLMProvider, api_key: str, model_name: Optional[str] = None):
        """Setup LLM provider"""
        self.llm_service.add_service(provider, api_key, model_name)

    def extract_from_pdf(self, pdf_path: Path, mode: ExtractionMode, provider: LLMProvider) -> Dict[str, Any]:
        """Extract ontology from curriculum PDF with enhanced text extraction"""
        
        # Use enhanced text extraction that handles complex layouts
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError(f"Could not extract text from {pdf_path}")

        logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
        logger.info(f"Full text length: {len(text)} characters")
        logger.info(f"Text preview (first 1000 chars): {text[:1000]}...")
        
        # Check if text is too short or generic
        if len(text.strip()) < 200:
            logger.warning(f"Very short text extracted from {pdf_path.name}: {len(text)} characters")
        
        # Process text in chunks if it's large
        chunks = self._chunk_text(text, max_chunk_size=12000)
        logger.info(f"Split text into {len(chunks)} chunks for processing")

        # Choose extraction method based on mode (using correct enum values)
        if mode == ExtractionMode.LLM_ONLY:
            result = self._extract_llm_only(chunks, provider, pdf_path.name)
        elif mode == ExtractionMode.LLM_ENHANCED:
            result = self._extract_llm_enhanced(chunks, provider, pdf_path.name)
        elif mode == ExtractionMode.RAG_ONLY:
            result = self._extract_rag_only(chunks, provider, pdf_path.name)
        elif mode == ExtractionMode.RAG_ENHANCED:
            result = self._extract_rag_enhanced(chunks, provider, pdf_path.name)
        else:
            raise ValueError(f"Unsupported extraction mode: {mode}")

        # Enhance with graph data
        turtle_output = result.get('turtle_output', '')
        json_graph = self._convert_turtle_to_comprehensive_graph(turtle_output)
        
        result['json_graph'] = json_graph
        result['entities'] = [n for n in json_graph['nodes'] if n.get('category') == 'entity']
        result['concepts'] = [n for n in json_graph['nodes'] if n.get('category') == 'concept']
        result['relations'] = json_graph['links']

        return result

    def _chunk_text(self, text: str, max_chunk_size: int = 12000) -> List[str]:
        """Split text into overlapping chunks for better processing"""
        if len(text) <= max_chunk_size:
            return [text]
        
        words = text.split()
        chunks = []
        overlap_size = 1000  # 1000 word overlap between chunks
        
        for i in range(0, len(words), max_chunk_size - overlap_size):
            chunk = ' '.join(words[i:i + max_chunk_size])
            if len(chunk.strip()) > 100:  # Only add meaningful chunks
                chunks.append(chunk)
        
        return chunks

    def _extract_llm_only(self, chunks: List[str], provider: LLMProvider, filename: str) -> Dict[str, Any]:
        """Extract using LLM only with enhanced prompt for better graph data"""
        
        all_turtle_outputs = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} for {filename}")
            
            prompt = f"""Εξάγετε λεπτομερή στοιχεία εκπαιδευτικής οντολογίας από αυτό το ελληνικό κείμενο αναλυτικού προγράμματος:

ΚΕΙΜΕΝΟ:
{chunk}

ΕΡΓΑΣΙΑ: Εξάγετε ΠΡΑΓΜΑΤΙΚΑ εκπαιδευτικά στοιχεία που αναφέρονται στο κείμενο:

1. ΜΑΘΗΣΙΑΚΟΙ ΣΤΟΧΟΙ - Βρείτε πραγματικούς στόχους που αναφέρονται:
   - Χρησιμοποιήστε την ελληνική περιγραφή από το κείμενο
   - Προσδιορίστε τη βαθμίδα (Α' Δημοτικού, Β' Δημοτικού κλπ)
   - Εντοπίστε το γνωστικό επίπεδο

2. ΔΕΞΙΟΤΗΤΕΣ - Εξάγετε συγκεκριμένες δεξιότητες που περιγράφονται:
   - Χρησιμοποιήστε την ακριβή ελληνική ορολογία
   - Κατηγοριοποιήστε (τεχνολογικές, δημιουργικές, επικοινωνιακές)
   - Συνδέστε με θέματα

3. ΘΕΜΑΤΙΚΕΣ ΕΝΟΤΗΤΕΣ - Προσδιορίστε τα πραγματικά θέματα:
   - Χρησιμοποιήστε τους τίτλους από το κείμενο
   - Συνδέστε με δεξιότητες και στόχους

Μορφή RDF Turtle με ΠΡΑΓΜΑΤΙΚΟ περιεχόμενο:

```turtle
@prefix curriculum: <http://curriculum.edu.gr/2022/> .

# Πραγματικός στόχος από το κείμενο
curriculum:Στόχος_Χρήση_ΤΠΕ a curriculum:LearningObjective ;
    curriculum:hasDescription "Χρήση των ΤΠΕ για την ενίσχυση της μάθησης" ;
    curriculum:hasGradeLevel curriculum:Primary_Education ;
    curriculum:hasCognitiveLevel curriculum:Apply .

# Πραγματική δεξιότητα από το κείμενο  
curriculum:Δεξιότητα_Παρουσίαση a curriculum:Skill ;
    curriculum:hasDescription "Δημιουργία ψηφιακών παρουσιάσεων" ;
    curriculum:hasSkillType curriculum:Technology ;
    curriculum:supportsTopic curriculum:Θέμα_ΤΠΕ .

# Πραγματικό θέμα από το κείμενο
curriculum:Θέμα_ΤΠΕ a curriculum:Topic ;
    curriculum:hasDescription "Τεχνολογίες Πληροφορίας και Επικοινωνιών" ;
    curriculum:hasGradeLevel curriculum:Primary_Education .
```

ΠΡΟΣΟΧΗ: Χρησιμοποιήστε ΜΟΝΟ πραγματικό περιεχόμενο από το κείμενο, όχι γενικά παραδείγματα."""

            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                turtle_chunk = self._clean_turtle_output(response)
                if turtle_chunk.strip():
                    all_turtle_outputs.append(turtle_chunk)
                    logger.info(f"Extracted ontology from chunk {i+1}: {len(turtle_chunk)} characters")
                else:
                    logger.warning(f"No valid turtle output from chunk {i+1}")
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {e}")
                continue

        # Combine all turtle outputs
        combined_turtle = self._combine_turtle_outputs(all_turtle_outputs)

        return {
            'mode': 'llm_only',
            'provider': provider.value,
            'filename': filename,
            'turtle_output': combined_turtle,
            'extracted_elements': self._count_elements(combined_turtle),
            'chunks_processed': len(chunks),
            'chunks_successful': len(all_turtle_outputs)
        }

    def _extract_llm_enhanced(self, chunks: List[str], provider: LLMProvider, filename: str) -> Dict[str, Any]:
        """Extract using LLM + external knowledge with focus on real Greek content"""
        
        # Extract concepts from first chunk for enhancement
        first_chunk = chunks[0] if chunks else ""
        concepts = self._extract_key_concepts(first_chunk)
        enhanced_knowledge = {}
        
        for concept in concepts[:3]:
            try:
                enhancement = knowledge_enhancer.enhance_learning_objective(concept)
                enhanced_knowledge[concept] = enhancement
            except Exception as e:
                logger.warning(f"Knowledge enhancement failed for {concept}: {e}")

        knowledge_context = self._format_enhanced_knowledge(enhanced_knowledge)
        all_turtle_outputs = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing enhanced chunk {i+1}/{len(chunks)} for {filename}")
            
            prompt = f"""Εξάγετε εμπλουτισμένη εκπαιδευτική οντολογία από ελληνικό αναλυτικό πρόγραμμα με διεθνή πρότυπα:

ΚΕΙΜΕΝΟ ΑΝΑΛΥΤΙΚΟΥ ΠΡΟΓΡΑΜΜΑΤΟΣ:
{chunk}

ΔΙΕΘΝΕΙΣ ΓΝΩΣΕΙΣ:
{knowledge_context}

ΕΣΤΙΑΣΗ: Εξάγετε πραγματικό περιεχόμενο από το ελληνικό κείμενο, όχι γενικά παραδείγματα.

ΕΜΠΛΟΥΤΙΣΜΕΝΗ ΕΞΑΓΩΓΗ:

1. ΠΡΑΓΜΑΤΙΚΑ ΣΤΟΙΧΕΙΑ ΑΝΑΛΥΤΙΚΟΥ ΠΡΟΓΡΑΜΜΑΤΟΣ:
   - Βρείτε τους πραγματικούς μαθησιακούς στόχους που αναφέρονται
   - Εξάγετε συγκεκριμένες δεξιότητες που περιγράφονται
   - Προσδιορίστε πραγματικά θέματα που αναφέρονται

2. ΔΙΕΘΝΗΣ ΕΥΘΥΓΡΑΜΜΙΣΗ:
   - curriculum:συσχετίζεταιΜεCEFR (A1, A2, B1, B2, C1, C2)
   - curriculum:παιδαγωγικήΠροσέγγιση (project_based, collaborative κλπ)
   - curriculum:διεθνήςΣυμβατότητα

3. ΠΡΑΓΜΑΤΙΚΕΣ ΣΧΕΣΕΙΣ από το κείμενο:
   - curriculum:απαιτείΠροηγούμενη
   - curriculum:οδηγείΣε  
   - curriculum:εμπλουτίζετται

Παράδειγμα με ΠΡΑΓΜΑΤΙΚΟ περιεχόμενο:
```turtle
curriculum:Στόχος_Δημιουργία_Κειμένων a curriculum:LearningObjective ;
    curriculum:hasDescription "Δημιουργία απλών κειμένων με χρήση ΤΠΕ" ;
    curriculum:συσχετίζεταιΜεCEFR curriculum:A2 ;
    curriculum:παιδαγωγικήΠροσέγγιση "collaborative_learning" ;
    curriculum:απαιτείΠροηγούμενη curriculum:Δεξιότητα_Βασικός_Χειρισμός .
```

Εξάγετε ΜΟΝΟ ό,τι αναφέρεται ρητά στο ελληνικό κείμενο."""

            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                turtle_chunk = self._clean_turtle_output(response)
                if turtle_chunk.strip():
                    all_turtle_outputs.append(turtle_chunk)
            except Exception as e:
                logger.error(f"Failed to process enhanced chunk {i+1}: {e}")
                continue

        combined_turtle = self._combine_turtle_outputs(all_turtle_outputs)

        return {
            'mode': 'llm_enhanced',
            'provider': provider.value,
            'filename': filename,
            'turtle_output': combined_turtle,
            'extracted_elements': self._count_elements(combined_turtle),
            'enhancement': 'external_knowledge',
            'enhanced_concepts': list(enhanced_knowledge.keys()),
            'chunks_processed': len(chunks),
            'chunks_successful': len(all_turtle_outputs)
        }

    def _extract_rag_only(self, chunks: List[str], provider: LLMProvider, filename: str) -> Dict[str, Any]:
        """Extract using RAG with cross-curriculum relationships"""
        
        # Extract objectives from first chunk for RAG context
        first_chunk = chunks[0] if chunks else ""
        current_objectives = self._extract_basic_objectives(first_chunk)
        rag_contexts = []
        
        for objective in current_objectives[:3]:
            try:
                relevant_chunks = rag_service.retrieve_relevant_context(objective, top_k=3)
                if relevant_chunks:
                    rag_contexts.extend(relevant_chunks)
            except Exception as e:
                logger.warning(f"RAG retrieval failed for {objective}: {e}")

        rag_context_text = self._format_rag_context(rag_contexts, [])
        all_turtle_outputs = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing RAG chunk {i+1}/{len(chunks)} for {filename}")
            
            prompt = f"""Εξάγετε οντολογία με διαπρογραμματικές σχέσεις χρησιμοποιώντας RAG context:

ΤΡΕΧΟΝ ΑΝΑΛΥΤΙΚΟ ΠΡΟΓΡΑΜΜΑ:
{chunk}

ΣΧΕΤΙΚΑ ΑΝΑΛΥΤΙΚΑ ΠΡΟΓΡΑΜΜΑΤΑ:
{rag_context_text}

ΕΡΓΑΣΙΑ: Δημιουργήστε οντολογία με εξέλιξη αναλυτικού προγράμματος και διασυνδέσεις:

1. ΣΧΕΣΕΙΣ ΕΞΕΛΙΞΗΣ:
   - curriculum:prerequisiteIn (εμφανίζεται ως προαπαιτούμενο σε άλλα προγράμματα)
   - curriculum:continuesIn (συνεχίζει στην επόμενη τάξη/πρόγραμμα)
   - curriculum:alignsWithCurriculum (συσχετίζεται με άλλο πρόγραμμα)

2. ΕΞΕΛΙΞΕΙΣ ΒΑΘΜΙΔΩΝ:
   - curriculum:gradeProgression (Α' Τάξη -> Β' Τάξη -> Ε' Τάξη)
   - curriculum:skillBuildsOn (δεξιότητα βασίζεται σε προηγούμενη)

3. ΔΙΑΠΡΟΓΡΑΜΜΑΤΙΚΕΣ ΑΝΤΙΣΤΟΙΧΙΣΕΙΣ:
   - curriculum:appearsIn (σε ποια προγράμματα εμφανίζεται)
   - curriculum:sharedWith (κοινά με άλλα μαθήματα)

Παράδειγμα εμπλουτισμένης μορφής RAG:
```turtle
curriculum:Δεξιότητα_Βασικός_Υπολογιστής a curriculum:Skill ;
    curriculum:hasDescription "Βασικός χειρισμός υπολογιστή" ;
    curriculum:prerequisiteIn curriculum:Προχωρημένο_Πρόγραμμα ;
    curriculum:continuesIn curriculum:Δεξιότητα_Προγραμματισμός ;
    curriculum:appearsIn "Μαθηματικά_Πρόγραμμα, Φυσικές_Επιστήμες" .
```

Δημιουργήστε Turtle με πλούσιες διαπρογραμματικές σχέσεις."""

            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                turtle_chunk = self._clean_turtle_output(response)
                if turtle_chunk.strip():
                    all_turtle_outputs.append(turtle_chunk)
            except Exception as e:
                logger.error(f"Failed to process RAG chunk {i+1}: {e}")
                continue

        combined_turtle = self._combine_turtle_outputs(all_turtle_outputs)

        return {
            'mode': 'rag_only',
            'provider': provider.value,
            'filename': filename,
            'turtle_output': combined_turtle,
            'extracted_elements': self._count_elements(combined_turtle),
            'rag_chunks_used': rag_contexts,  # ADD THIS LINE
            'rag_context': f"Retrieved {len(rag_contexts)} relevant chunks",
            'chunks_processed': len(chunks),
            'chunks_successful': len(all_turtle_outputs)
        }

    def _extract_rag_enhanced(self, chunks: List[str], provider: LLMProvider, filename: str) -> Dict[str, Any]:
        """Extract using RAG + external knowledge (full enhancement)"""
        
        # Combine both RAG and knowledge enhancement
        first_chunk = chunks[0] if chunks else ""
        current_objectives = self._extract_basic_objectives(first_chunk)
        print(f"DEBUG: Found {len(current_objectives)} objectives: {current_objectives}")
        rag_contexts = []
        
        for objective in current_objectives[:2]:
            try:
                relevant_chunks = rag_service.retrieve_relevant_context(objective, top_k=2)
                rag_contexts.extend(relevant_chunks)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        concepts = self._extract_key_concepts(first_chunk)
        enhanced_knowledge = {}
        
        for concept in concepts[:2]:
            try:
                enhancement = knowledge_enhancer.enhance_learning_objective(concept)
                enhanced_knowledge[concept] = enhancement
            except Exception as e:
                logger.warning(f"Knowledge enhancement failed: {e}")

        rag_context_text = self._format_rag_context(rag_contexts, [])
        knowledge_context = self._format_enhanced_knowledge(enhanced_knowledge)
        all_turtle_outputs = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing full enhanced chunk {i+1}/{len(chunks)} for {filename}")
            
            prompt = f"""Δημιουργήστε την πιο ολοκληρωμένη εκπαιδευτική οντολογία συνδυάζοντας τρέχον πρόγραμμα, διαπρογραμματικά δεδομένα και διεθνή πρότυπα:

ΤΡΕΧΟΝ ΑΝΑΛΥΤΙΚΟ ΠΡΟΓΡΑΜΜΑ:
{chunk}

RAG CONTEXT:
{rag_context_text}

ΔΙΕΘΝΕΙΣ ΓΝΩΣΕΙΣ:
{knowledge_context}

ΕΡΓΑΣΙΑ: Δημιουργήστε την πλουσιότερη δυνατή οντολογία με όλους τους τύπους σχέσεων:

1. ΠΛΗΡΕΙΣ ΙΔΙΟΤΗΤΕΣ ΟΝΤΟΤΗΤΩΝ:
   - Βασικές: hasDescription, hasGradeLevel, hasCognitiveLevel
   - Διεθνείς: alignsWithCEFR, pedagogicalApproach, alignsWithStandards
   - Διαπρογραμματικές: appearsIn, prerequisiteIn, continuesIn

2. ΟΛΟΚΛΗΡΩΜΕΝΕΣ ΣΧΕΣΕΙΣ:
   - Ιεραρχικές: hasPrerequisite, progressesTo, belongsToGrade
   - Διαπρογραμματικές: alignsWithCurriculum, sharedWith, mapsToCurriculum
   - Αξιολόγησης: assessedBy, evaluatedThrough, measuredBy
   - Παιδαγωγικές: taughtUsing, learnedThrough, supportedBy

3. ΠΛΟΥΣΙΑ ΜΕΤΑΔΕΔΟΜΕΝΑ:
   - curriculum:enhancedByRAG "yes"
   - curriculum:enhancedByKnowledge "yes" 
   - curriculum:comprehensiveness "full"
   - curriculum:qualityScore (1-10 βάσει πλούτου)

Δημιουργήστε την πιο λεπτομερή, διασυνδεδεμένη οντολογία αναλυτικού προγράμματος που είναι δυνατή."""

            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                turtle_chunk = self._clean_turtle_output(response)
                if turtle_chunk.strip():
                    all_turtle_outputs.append(turtle_chunk)
            except Exception as e:
                logger.error(f"Failed to process full enhanced chunk {i+1}: {e}")
                continue

        combined_turtle = self._combine_turtle_outputs(all_turtle_outputs)

        return {
            'mode': 'rag_enhanced',
            'provider': provider.value,
            'filename': filename,
            'turtle_output': combined_turtle,
            'extracted_elements': self._count_elements(combined_turtle),
            'enhancement': 'full_rag_knowledge',
            'rag_chunks_used': rag_contexts,  # ADD THIS LINE
            'rag_chunks': len(rag_contexts),
            'enhanced_concepts': list(enhanced_knowledge.keys()),
            'chunks_processed': len(chunks),
            'chunks_successful': len(all_turtle_outputs)
        }

    def _combine_turtle_outputs(self, turtle_outputs: List[str]) -> str:
        """Combine multiple turtle outputs into one coherent output"""
        if not turtle_outputs:
            return ""
        
        # Remove duplicate prefixes and combine
        combined_lines = []
        prefixes_added = False
        
        for turtle in turtle_outputs:
            lines = turtle.split('\n')
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('@prefix'):
                    if not prefixes_added:
                        combined_lines.append(line)
                elif stripped and not stripped.startswith('#'):
                    combined_lines.append(line)
            
            if not prefixes_added:
                prefixes_added = True
                combined_lines.append('')  # Empty line after prefixes
        
        return '\n'.join(combined_lines)

    def _convert_turtle_to_comprehensive_graph(self, turtle_string: str) -> Dict[str, Any]:
        """Convert Turtle RDF to comprehensive JSON graph for visualization with proper entity/concept classification"""
        
        if not turtle_string.strip():
            return {
                "nodes": [],
                "links": [],
                "overview": {"entities": 0, "concepts": 0, "relations": 0}
            }

        # Ensure prefixes
        turtle_string = self._ensure_turtle_prefixes(turtle_string)
        
        g = Graph()
        try:
            g.parse(data=turtle_string, format="turtle")
        except Exception as e:
            logger.error(f"Failed to parse RDF Turtle: {e}")
            return {"nodes": [], "links": [], "overview": {"entities": 0, "concepts": 0, "relations": 0}}

        nodes = {}
        links = []
        
        CURRICULUM = Namespace("http://curriculum.edu.gr/2022/")
        RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        
        # Extract descriptions for entities
        descriptions = {}
        for s, p, o in g.triples((None, CURRICULUM.hasDescription, None)):
            descriptions[str(s)] = str(o)

        # Extract grade levels
        grade_levels = {}
        for s, p, o in g.triples((None, CURRICULUM.hasGradeLevel, None)):
            grade_levels[str(s)] = str(o).split(":")[-1] if ":" in str(o) else str(o)

        # Extract cognitive levels
        cognitive_levels = {}
        for s, p, o in g.triples((None, CURRICULUM.hasCognitiveLevel, None)):
            cognitive_levels[str(s)] = str(o).split(":")[-1] if ":" in str(o) else str(o)

        # Extract skill types
        skill_types = {}
        for s, p, o in g.triples((None, CURRICULUM.hasSkillType, None)):
            skill_types[str(s)] = str(o).split(":")[-1] if ":" in str(o) else str(o)

        # Create nodes from all subjects and objects
        for s, p, o in g:
            for uri in [s, o]:
                if isinstance(uri, URIRef):
                    uri_str = str(uri)
                    
                    if uri_str not in nodes:
                        # Get Greek description if available
                        greek_description = descriptions.get(uri_str, "")
                        
                        # Get clean URI fragment for fallback
                        uri_fragment = uri.n3(g.namespace_manager)
                        if uri_fragment.startswith("curriculum:"):
                            uri_fragment = uri_fragment.split(":")[-1]
                        
                        # More lenient filtering - only skip truly generic placeholders
                        generic_patterns = ["Topic_1", "Skill_1", "Objective_1", "Assessment_1", "Entity_1"]
                        if uri_fragment in generic_patterns and not greek_description:
                            continue
                        
                        # Determine category and type based on RDF type
                        rdf_type = g.value(uri, RDF.type)
                        
                        if rdf_type:
                            type_str = str(rdf_type)
                            
                            # Entities: instances of curriculum classes
                            if ("LearningObjective" in type_str or "Topic" in type_str or "Skill" in type_str or "AssessmentMethod" in type_str):
                                category = "entity"
                                if "LearningObjective" in type_str:
                                    entity_type = "LearningObjective"
                                elif "Topic" in type_str:
                                    entity_type = "Topic"
                                elif "Skill" in type_str:
                                    entity_type = "Skill"
                                elif "AssessmentMethod" in type_str:
                                    entity_type = "AssessmentMethod"
                                
                                # Use Greek description as label, fallback to URI fragment
                                label = greek_description if greek_description else uri_fragment
                                
                            # Concepts: classification types
                            elif ("GradeLevel" in type_str or "CognitiveLevel" in type_str or "SkillType" in type_str):
                                category = "concept"
                                if "GradeLevel" in type_str:
                                    entity_type = "GradeLevel"
                                elif "CognitiveLevel" in type_str:
                                    entity_type = "CognitiveLevel"
                                elif "SkillType" in type_str:
                                    entity_type = "SkillType"
                                
                                # Use Greek description as label, fallback to URI fragment
                                label = greek_description if greek_description else uri_fragment
                            else:
                                # Unknown RDF type
                                if greek_description:
                                    category = "entity"
                                    entity_type = "Entity"
                                    label = greek_description
                                else:
                                    category = "concept"
                                    entity_type = "Concept"
                                    label = uri_fragment
                        else:
                            # No RDF type - determine by content
                            if greek_description:
                                category = "entity"
                                entity_type = "Entity"
                                label = greek_description
                            else:
                                category = "concept"
                                entity_type = "Concept"
                                label = uri_fragment
                        
                        # Truncate long labels for display
                        if len(label) > 50:
                            label = label[:50] + "..."
                        
                        nodes[uri_str] = {
                            "id": uri_str,
                            "label": label,
                            "type": entity_type,
                            "category": category,
                            "description": greek_description or f"Concept: {uri_fragment}",
                            "gradeLevel": grade_levels.get(uri_str, ""),
                            "cognitiveLevel": cognitive_levels.get(uri_str, ""),
                            "skillType": skill_types.get(uri_str, ""),
                            "color": self._get_node_color(entity_type),
                            "uri_fragment": uri_fragment
                        }

            # Create links - filter out rdf:type relationships
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                predicate_str = str(p)
                predicate_label = p.n3(g.namespace_manager).split(":")[-1]
                
                # Skip rdf:type relationships as they're not meaningful curriculum relationships
                if predicate_str.endswith("type") or "rdf-syntax-ns#type" in predicate_str:
                    continue
                
                links.append({
                    "source": str(s),
                    "target": str(o),
                    "label": predicate_label,
                    "type": self._get_link_type(predicate_label)
                })

        # Calculate overview statistics and create enriched links with entity context
        node_list = list(nodes.values())
        
        # Enrich links with source and target entity information for better display
        enriched_links = []
        for link in links:
            source_node = nodes.get(link["source"])
            target_node = nodes.get(link["target"])
            
            # Only add link if both source and target nodes exist (weren't filtered out)
            if source_node and target_node:
                # Create enriched link with entity labels
                source_label = source_node["label"][:25] + "..." if len(source_node["label"]) > 25 else source_node["label"]
                target_label = target_node["label"][:25] + "..." if len(target_node["label"]) > 25 else target_node["label"]
                
                enriched_link = {
                    "source": link["source"],
                    "target": link["target"],
                    "label": link["label"],
                    "type": link["type"],
                    "source_label": source_label,
                    "target_label": target_label,
                    "full_relation": f"{source_label} —[{link['label']}]→ {target_label}"
                }
                enriched_links.append(enriched_link)
        
        entities = len([n for n in node_list if n["category"] == "entity"])
        concepts = len([n for n in node_list if n["category"] == "concept"])
        relations = len(enriched_links)

        return {
            "nodes": node_list,
            "links": enriched_links,
            "overview": {
                "entities": entities,
                "concepts": concepts,
                "relations": relations,
                "total_nodes": len(node_list)
            }
        }

    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            "LearningObjective": "#3b82f6",  # Blue
            "Topic": "#10b981",              # Green
            "Skill": "#f59e0b",              # Orange
            "AssessmentMethod": "#ef4444",   # Red
            "GradeLevel": "#8b5cf6",         # Purple
            "CognitiveLevel": "#06b6d4",     # Cyan
            "SkillType": "#f97316",          # Orange-red
            "Entity": "#3b82f6",             # Blue for generic entities
            "Concept": "#6b7280",            # Gray for generic concepts
            "unknown": "#6b7280"             # Gray
        }
        return color_map.get(entity_type, "#6b7280")

    def _get_link_type(self, predicate_label: str) -> str:
        """Categorize link types for styling"""
        if predicate_label in ["hasPrerequisite", "progressesTo", "prerequisiteFor"]:
            return "progression"
        elif predicate_label in ["assessedBy", "evaluatedThrough", "measuredBy"]:
            return "assessment"
        elif predicate_label in ["belongsToGrade", "hasGradeLevel", "hasCognitiveLevel"]:
            return "classification"
        elif predicate_label in ["supportsTopic", "enablesSkill", "relatesTo"]:
            return "support"
        else:
            return "general"

    def _ensure_turtle_prefixes(self, ttl: str) -> str:
        """Ensure required @prefix declarations exist in a Turtle string."""
        needed = {
            "curriculum": "http://curriculum.edu.gr/2022/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
        existing = set()
        for line in ttl.splitlines():
            m = re.match(r'\s*@prefix\s+([A-Za-z][\w\-]*)\s*:\s*<[^>]+>\s*\.\s*', line)
            if m:
                existing.add(m.group(1))

        header_lines = []
        for pfx, iri in needed.items():
            if pfx not in existing:
                header_lines.append(f"@prefix {pfx}: <{iri}> .")

        if header_lines:
            return "\n".join(header_lines) + "\n\n" + ttl
        return ttl

    def _extract_key_concepts(self, text: str) -> List[str]:
        concept_patterns = [
            r'μαθησιακός στόχος', r'μαθησιακοί στόχοι',
            r'δεξιότητα', r'δεξιότητες',
            r'ικανότητα', r'ικανότητες',
            r'αξιολόγηση', r'διαμορφωτική αξιολόγηση',
            r'γραμματική', r'λεξιλόγιο', r'ορθογραφία',
            r'ανάγνωση', r'γραφή', r'προφορικός λόγος',
            r'λογοτεχνία', r'αφήγηση', r'ποίηση',
            r'ΤΠΕ', r'τεχνολογία', r'υπολογιστής',
            r'επίλυση προβλημάτων', r'κριτική σκέψη'
        ]
        
        concepts = []
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))[:10]

    def _extract_basic_objectives(self, text: str) -> List[str]:
        objective_patterns = [
            r'[Οο] μαθητής.*?να ([^.]+)',
            r'[Οο]ι μαθητές.*?να ([^.]+)',
            r'Στόχος.*?([^.]+)',
            r'Επιδιώκεται.*?([^.]+)',
            r'Αναπτύσσει.*?([^.]+)',
            r'Κατανοεί.*?([^.]+)',
            r'Χρησιμοποιεί.*?([^.]+)'
        ]
        
        objectives = []
        for pattern in objective_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            objectives.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return objectives[:5]

    def _format_rag_context(self, rag_contexts: List[Dict], progression_contexts: List[Dict]) -> str:
        if not rag_contexts and not progression_contexts:
            return "No relevant context retrieved."
        
        formatted = []
        
        if rag_contexts:
            formatted.append("RELATED CURRICULUM CONTENT:")
            for i, context in enumerate(rag_contexts[:5], 1):
                formatted.append(f"{i}. {context['text'][:200]}... (from {context['source_file']})")
        
        return '\n'.join(formatted)

    def _format_enhanced_knowledge(self, enhanced_knowledge: Dict) -> str:
        if not enhanced_knowledge:
            return "No external knowledge available."
        
        formatted = []
        
        for concept, knowledge in enhanced_knowledge.items():
            formatted.append(f"\nCONCEPT: {concept}")
            
            if 'pedagogical_enhancement' in knowledge:
                enhancement = knowledge['pedagogical_enhancement']
                if 'bloom_taxonomy' in enhancement:
                    bloom = enhancement['bloom_taxonomy']
                    formatted.append(f"- Bloom's Level: {bloom.get('primary_level', 'unknown')}")
                
                if 'cefr_alignment' in enhancement:
                    cefr = enhancement['cefr_alignment']
                    formatted.append(f"- CEFR Level: {cefr.get('level', 'N/A')}")
        
        return '\n'.join(formatted)

    def _clean_turtle_output(self, raw_output: str) -> str:
        """Clean and format turtle output"""
        # Extract turtle code block if present
        match = re.search(r"```(?:turtle)?\s*(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If no code block, try to extract turtle-like content
        lines = raw_output.strip().split('\n')
        turtle_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('curriculum:') or stripped.startswith('@prefix') or 
                ':' in stripped or stripped.endswith(' ;') or stripped.endswith(' .')):
                turtle_lines.append(line)
        
        return '\n'.join(turtle_lines) if turtle_lines else raw_output

    def _count_elements(self, turtle_output: str) -> Dict[str, int]:
        """Count extracted ontology elements"""
        counts = {
            'LearningObjective': len(re.findall(r'curriculum:LearningObjective', turtle_output)),
            'Topic': len(re.findall(r'curriculum:Topic', turtle_output)),
            'Skill': len(re.findall(r'curriculum:Skill', turtle_output)),
            'GradeLevel': len(re.findall(r'curriculum:GradeLevel', turtle_output)),
            'AssessmentMethod': len(re.findall(r'curriculum:AssessmentMethod', turtle_output)),
            'Entities': len(re.findall(r'curriculum:\w+\s+a\s+curriculum:', turtle_output)),
            'Relations': len(re.findall(r'curriculum:\w+\s+curriculum:\w+', turtle_output))
        }
        return counts

    def save_ontology(self, result: Dict[str, Any], output_path: Path):
        """Save extracted ontology to file"""
        
        # Add prefixes
        prefixes = """@prefix curriculum: <http://curriculum.edu.gr/2022/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""
        
        # Add metadata
        metadata = f"""# Greek Curriculum Ontology Extraction
# Mode: {result['mode']}
# Provider: {result['provider']}
# Source: {result['filename']}
# Extracted: {sum(result['extracted_elements'].values())} elements
# Chunks processed: {result.get('chunks_processed', 'N/A')}
# Successful chunks: {result.get('chunks_successful', 'N/A')}

"""
        
        full_content = prefixes + metadata + result['turtle_output']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        logger.info(f"Saved ontology to {output_path}")
        return output_path