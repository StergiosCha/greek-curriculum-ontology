import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re
import json
import time # Added for retry delay
from rdflib import Graph, RDF, RDFS, URIRef, Namespace, Literal


from app.services.llm_service import MultiLLMService, LLMProvider
from app.services.rag_service import rag_service
from app.services.contradiction_detector import ContradictionDetector


logger = logging.getLogger(__name__)


CURRICULUM = Namespace("http://curriculum.edu.gr/2022/")
CURRKG = Namespace("http://curriculum-kg.org/ontology/")


class RAGContradictionDetector(ContradictionDetector):
    """RAG-enhanced contradiction detector - inherits basic functionality and adds RAG enhancement"""
    
    def __init__(self):
        super().__init__()
        self.rag_service = rag_service
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Improved LLM response parser with multiple fallback strategies.
        Handles conversational text, nested JSON, and malformed responses.
        """
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")
        
        # Strategy 1: Try multiple regex patterns for JSON extraction
        json_patterns = [
            # Pattern 1: Balanced braces (handles nested JSON properly)
            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',
            # Pattern 2: Simple greedy match as fallback
            r'\{.*\}',
            # Pattern 3: JSON array if needed
            r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]'
        ]
        
        for i, pattern in enumerate(json_patterns, 1):
            matches = re.findall(pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    cleaned_match = match.strip()
                    parsed = json.loads(cleaned_match)
                    logger.info(f"Successfully parsed JSON using pattern {i}")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"Pattern {i} match failed JSON parsing: {e}")
                    continue
        
        # Strategy 2: Extract content between first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = response[first_brace:last_brace + 1]
            try:
                parsed = json.loads(potential_json)
                logger.info("Successfully parsed JSON using brace extraction")
                return parsed
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try to clean common JSON issues
        cleaned_response = self._clean_malformed_json(response)
        if cleaned_response:
            try:
                parsed = json.loads(cleaned_response)
                logger.info("Successfully parsed JSON after cleaning")
                return parsed
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Create fallback response based on content
        logger.error(f"All JSON parsing strategies failed. Response preview: {response[:200]}...")
        
        if "contradictions" in response.lower():
            return {
                "contradictions": [],
                "analysis": "JSON parsing failed - response contained contradictions analysis but was malformed",
                "raw_response": response[:500],
                "parsing_error": True
            }
        else:
            return {
                "error": "Failed to parse LLM response as JSON", 
                "raw_response": response[:500],
                "parsing_error": True
            }

    def _clean_malformed_json(self, response: str) -> str:
        """
        Attempt to clean common JSON formatting issues in LLM responses
        """
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return None
        
        json_content = json_match.group(0)
        cleaned = json_content
        
        # Fix trailing commas before } or ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        
        # Fix missing quotes around keys (basic cases)
        cleaned = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', cleaned)
        
        # Fix double-escaped quotes
        cleaned = re.sub(r'\\"', '"', cleaned)
        
        # Fix unicode issues (if any)
        try:
            cleaned = cleaned.encode('utf-8').decode('utf-8')
        except:
            pass
        
        return cleaned

    def _generate_with_retry(self, provider: LLMProvider, prompt: str, max_retries: int = 3) -> str:
        """
        Enhanced retry mechanism with better error handling and adaptive prompts
        """
        last_response = None
        last_error = None
        
        for retry_count in range(max_retries + 1):
            try:
                if retry_count == 0:
                    # First attempt - use original prompt with JSON enforcement
                    enhanced_prompt = prompt + self._get_json_enforcement_suffix()
                    response = self.llm_service.generate_with_provider(provider, enhanced_prompt)
                else:
                    # Retry attempts - use adaptive prompts
                    if retry_count == 1:
                        adaptive_prompt = f"""The previous response was not valid JSON. Please provide ONLY a valid JSON object with no additional text.

{prompt}

CRITICAL: Response must be ONLY valid JSON - no explanations, no Greek text outside the JSON, no markdown formatting."""
                    else:
                        adaptive_prompt = f"""The previous response failed JSON parsing with error: {last_error}

Please provide ONLY a valid JSON object that follows this exact format:
{{
  "contradictions": [...],
  "analysis": "..."
}}

CRITICAL: Return ONLY the JSON object, nothing else."""
                    
                    response = self.llm_service.generate_with_provider(provider, adaptive_prompt)
                
                # Test parsing
                self._parse_llm_response(response)
                logger.info(f"Successfully generated parseable response on attempt {retry_count + 1}")
                return response
                
            except (ValueError, json.JSONDecodeError) as e:
                last_error = str(e)
                last_response = response if 'response' in locals() else None
                
                logger.warning(f"Attempt {retry_count + 1} failed: {e}")
                
                if retry_count < max_retries:
                    time.sleep(1 + retry_count)  # Increasing delay
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
                    return last_response if last_response else ""
            
            except Exception as e:
                logger.error(f"Unexpected error on attempt {retry_count + 1}: {e}")
                if retry_count == max_retries:
                    raise Exception(f"Failed after {max_retries + 1} attempts: {e}") from e
        
        return last_response if last_response else ""

    def _get_json_enforcement_suffix(self) -> str:
        """Get JSON enforcement suffix for prompts"""
        return """

ΚΡΙΣΙΜΟ: Η απάντηση πρέπει να είναι ΜΟΝΟ ένα έγκυρο JSON αντικείμενο. 
ΜΗΝ περιλάβετε ΚΑΜΙΑ άλλη εξήγηση, κείμενο ή μορφοποίηση. 
ΜΟΝΟ το JSON αντικείμενο."""
    
    def detect_internal_contradictions(self, ontology_path: Path, provider: LLMProvider) -> Dict[str, Any]:
        """RAG-enhanced internal contradiction detection"""
        g = self.load_ontology(ontology_path)
        curriculum_content = self._extract_curriculum_content(g)
        
        if not curriculum_content:
            return {'contradictions': [], 'analysis': 'No curriculum content found'}
        
        rag_context = self._get_rag_contradiction_context(curriculum_content)
        enhanced_prompt = self._build_rag_enhanced_internal_prompt(curriculum_content, rag_context)
        
        try:
            response = self._generate_with_retry(provider, enhanced_prompt)
            result = self._parse_llm_response(response)
            
            result['rag_enhancement'] = {
                'contexts_used': len(rag_context),
                'method': 'rag_enhanced_internal',
                'rag_sources': [ctx['source_file'] for ctx in rag_context]
            }
            return result
        except Exception as e:
            logger.error(f"RAG-enhanced internal analysis failed: {e}")
            return {'error': str(e), 'contradictions': []}
    
    def detect_cross_curriculum_contradictions(self, ontology_paths: List[Path], provider: LLMProvider) -> Dict[str, Any]:
        """RAG-enhanced cross-curriculum contradiction detection"""
        
        curricula_content = {}
        subject_relationships = {}
        
        for path in ontology_paths:
            g = self.load_ontology(path)
            curriculum_data = self._extract_curriculum_content(g)
            if curriculum_data:
                curricula_content[path.stem] = curriculum_data
                subject_area = self._infer_subject_area(curriculum_data['curriculum_title'])
                grade_levels = set()
                for module in curriculum_data.get('modules', []):
                    if module.get('grade_level'):
                        grade_levels.add(module['grade_level'])
                
                subject_relationships[path.stem] = {
                    'subject_area': subject_area,
                    'grade_levels': list(grade_levels),
                    'is_language_related': self._is_language_related_subject(subject_area, curriculum_data['curriculum_title'])
                }
        
        if len(curricula_content) < 2:
            return {'contradictions': [], 'analysis': 'Need at least 2 curricula for comparison'}
        
        cross_patterns = self._get_cross_curricular_rag_patterns(curricula_content)
        prerequisite_context = self._get_prerequisite_rag_context(curricula_content)
        enhanced_prompt = self._build_rag_enhanced_cross_prompt(
            curricula_content, subject_relationships, cross_patterns, prerequisite_context
        )
        
        try:
            response = self._generate_with_retry(provider, enhanced_prompt)
            result = self._parse_llm_response(response)
            
            result['rag_enhancement'] = {
                'cross_patterns_found': len(cross_patterns),
                'prerequisite_contexts': len(prerequisite_context),
                'method': 'rag_enhanced_cross_curriculum',
                'rag_sources': list(set([ctx['source_file'] for ctx in cross_patterns + prerequisite_context]))
            }
            return result
        except Exception as e:
            logger.error(f"RAG-enhanced cross-curriculum analysis failed: {e}")
            return {'error': str(e), 'contradictions': []}
    
    def analyze_progression_coherence(self, ontology_paths: List[Path], provider: LLMProvider) -> Dict[str, Any]:
        """RAG-enhanced learning progression coherence analysis"""
        
        grade_curricula = {}
        for path in ontology_paths:
            g = self.load_ontology(path)
            grade_info = self._extract_grade_progression(g)
            if grade_info:
                grade_curricula[path.stem] = grade_info
        
        if not grade_curricula:
            return {'analysis': 'No grade progression data found'}
        
        progression_patterns = self._get_progression_rag_patterns(grade_curricula)
        skill_development_context = self._get_skill_development_rag_context(grade_curricula)
        enhanced_prompt = self._build_rag_enhanced_progression_prompt(
            grade_curricula, progression_patterns, skill_development_context
        )
        
        try:
            response = self._generate_with_retry(provider, enhanced_prompt)
            result = self._parse_llm_response(response)
            
            result['rag_enhancement'] = {
                'progression_patterns_analyzed': len(progression_patterns),
                'skill_contexts_used': len(skill_development_context),
                'method': 'rag_enhanced_progression',
                'rag_sources': list(set([ctx['source_file'] for ctx in progression_patterns + skill_development_context]))
            }
            return result
        except Exception as e:
            logger.error(f"RAG-enhanced progression analysis failed: {e}")
            return {'error': str(e)}

    def _get_rag_contradiction_context(self, curriculum_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant RAG context for internal contradiction detection with debugging"""
        context = []
        
        logger.info(f"RAG database has {len(self.rag_service.curriculum_db)} curricula")
        for key in self.rag_service.curriculum_db.keys():
            logger.info(f"Available curriculum: {key}")
        
        key_concepts = self._extract_key_concepts(curriculum_content)
        logger.info(f"Extracted key concepts: {key_concepts[:3]}...")
        
        contradiction_queries = [
            "αντιφάσεις προβλήματα αναλυτικό πρόγραμμα",
            "προαπαιτούμενες γνώσεις κυκλικές εξαρτήσεις", 
            "γνωστικό επίπεδο ασυνέπειες",
            "βαθμίδα ακατάλληλη δυσκολία",
            "οικιακή οικονομία γυμνάσιο",
            "διατροφή υγιεινή δεξιότητες",
            "μαθησιακοί στόχοι ενότητες"
        ]
        
        for concept in key_concepts[:3]:
            if len(concept.strip()) > 3:
                contradiction_queries.append(f"προβλήματα αντιφάσεις {concept}")
                contradiction_queries.append(concept)
        
        logger.info(f"Using {len(contradiction_queries)} RAG queries")
        
        for i, query in enumerate(contradiction_queries):
            try:
                logger.info(f"RAG Query {i+1}: '{query}'")
                relevant_chunks = self.rag_service.retrieve_relevant_context(query, top_k=2)
                logger.info(f"Found {len(relevant_chunks)} chunks for query: {query}")
                
                for chunk in relevant_chunks:
                    logger.info(f"Chunk similarity: {chunk['similarity']:.3f}, Source: {chunk['source_file']}")
                    logger.info(f"Chunk preview: {chunk['text'][:100]}...")
                    
                context.extend(relevant_chunks)
            except Exception as e:
                logger.error(f"RAG query failed for '{query}': {e}")
        
        logger.info(f"Total RAG context chunks before deduplication: {len(context)}")
        
        seen_sources = {}
        unique_context = []
        
        for ctx in sorted(context, key=lambda x: x['similarity'], reverse=True):
            source = ctx['source_file']
            if source not in seen_sources or ctx['similarity'] > seen_sources[source]['similarity']:
                seen_sources[source] = ctx
        
        unique_context = list(seen_sources.values())
        
        filtered_context = [ctx for ctx in unique_context if ctx['similarity'] > 0.2]
        final_context = filtered_context[:8]
        
        logger.info(f"Final RAG context: {len(final_context)} chunks")
        for ctx in final_context:
            logger.info(f"Final chunk: {ctx['source_file']}, similarity: {ctx['similarity']:.3f}")
        
        return final_context
    
    def _get_cross_curricular_rag_patterns(self, curricula_content: Dict) -> List[Dict[str, Any]]:
        """Get cross-curricular patterns from RAG database"""
        patterns = []
        
        subjects = []
        for name, data in curricula_content.items():
            title_lower = data.get('curriculum_title', '').lower()
            if 'γλώσσα' in title_lower:
                subjects.append('γλώσσα')
            elif 'λογοτεχνία' in title_lower:
                subjects.append('λογοτεχνία')
            elif 'μαθηματικ' in title_lower:
                subjects.append('μαθηματικά')
        
        cross_queries = [
            "διαθεματικές συνδέσεις συνεργασία",
            "μεταφορά γνώσης μεταξύ μαθημάτων",
            "κοινές δεξιότητες διαφορετικά μαθήματα"
        ]
        
        for i, subject1 in enumerate(subjects):
            for subject2 in subjects[i+1:]:
                cross_queries.append(f"{subject1} {subject2} συνδέσεις")
        
        for query in cross_queries:
            context = self.rag_service.retrieve_relevant_context(query, top_k=1)
            patterns.extend(context)
        
        return patterns[:8]
    
    def _get_prerequisite_rag_context(self, curricula_content: Dict) -> List[Dict[str, Any]]:
        """Get prerequisite relationship context from RAG"""
        prereq_context = []
        
        prereq_queries = [
            "προαπαιτούμενες γνώσεις δεξιότητες",
            "σειρά διδασκαλίας μαθησιακή πορεία",
            "βάση προϋποθέσεις μάθηση"
        ]
        
        for name, data in curricula_content.items():
            for module in data.get('modules', []):
                if module.get('prerequisites'):
                    prereq_queries.append(f"προαπαιτούμενα {module.get('title', '')}")
        
        for query in prereq_queries:
            context = self.rag_service.retrieve_relevant_context(query, top_k=1)
            prereq_context.extend(context)
        
        return prereq_context[:5]
    
    def _get_progression_rag_patterns(self, grade_curricula: Dict) -> List[Dict[str, Any]]:
        """Get learning progression patterns from RAG"""
        patterns = []
        
        progression_queries = [
            "εξελικτική πορεία μάθηση βαθμίδες",
            "γνωστική ανάπτυξη πρόοδος",
            "μαθησιακή εξέλιξη στάδια",
            "δεξιότητες βαθμίδα εξέλιξη"
        ]
        
        for curriculum in grade_curricula.keys():
            progression_queries.append(f"εξέλιξη πρόοδος {curriculum}")
        
        for query in progression_queries:
            context = self.rag_service.retrieve_relevant_context(query, top_k=1)
            patterns.extend(context)
        
        return patterns[:6]
    
    def _get_skill_development_rag_context(self, grade_curricula: Dict) -> List[Dict[str, Any]]:
        """Get skill development context from RAG"""
        skill_context = []
        
        skill_queries = [
            "ανάπτυξη δεξιοτήτων μαθητές",
            "γνωστικές ικανότητες εξέλιξη",
            "μαθησιακές προϋποθέσεις ηλικία",
            "ψυχοπαιδαγωγική ανάπτυξη"
        ]
        
        for query in skill_queries:
            context = self.rag_service.retrieve_relevant_context(query, top_k=1)
            skill_context.extend(context)
        
        return skill_context[:4]
    
    def _extract_key_concepts(self, curriculum_content: Dict[str, Any]) -> List[str]:
        """Extract key concepts from curriculum content for RAG queries"""
        concepts = []
        
        if curriculum_content.get('curriculum_title'):
            concepts.append(curriculum_content['curriculum_title'])
        
        for module in curriculum_content.get('modules', []):
            if module.get('title'):
                concepts.append(module['title'])
            if module.get('topics'):
                concepts.extend(module['topics'][:2])
        
        for obj in curriculum_content.get('objectives', []):
            if obj.get('description'):
                desc_words = obj['description'].split()[:8]
                concepts.append(' '.join(desc_words))
        
        return concepts[:8]
    
    def _build_rag_enhanced_internal_prompt(self, curriculum_content: Dict[str, Any], 
                                          rag_context: List[Dict[str, Any]]) -> str:
        """Build RAG-enhanced prompt for internal contradiction analysis"""
        
        formatted_content = self._format_curriculum_for_analysis(curriculum_content)
        
        rag_section = ""
        if rag_context:
            rag_section = "\nΣΥΜΦΡΑΖΟΜΕΝΑ ΑΠΟ RAG DATABASE:\n"
            for i, ctx in enumerate(rag_context, 1):
                rag_section += f"{i}. Πηγή: {ctx['source_file']} (Ομοιότητα: {ctx['similarity']:.2f})\n"
                rag_section += f"   Περιεχόμενο: {ctx['text'][:350]}...\n"
                rag_section += f"   Θεματική: {ctx.get('subject', 'N/A')} | Βαθμίδα: {ctx.get('education_level', 'N/A')}\n\n"
        
        prompt = f"""Αναλύστε το ελληνικό αναλυτικό πρόγραμμα για εσωτερικές αντιφάσεις χρησιμοποιώντας RAG συμφραζόμενα:

ΠΡΟΓΡΑΜΜΑ ΠΡΟΣ ΑΝΑΛΥΣΗ:
{formatted_content}

{rag_section}

ΟΔΗΓΙΕΣ RAG-ENHANCED ΑΝΑΛΥΣΗΣ:

1. ΧΡΗΣΙΜΟΠΟΙΗΣΤΕ τα RAG συμφραζόμενα για:
   - Εντοπισμό παρόμοιων προβλημάτων σε άλλα προγράμματα
   - Σύγκριση με επιτυχημένες πρακτικές
   - Εμπλουτισμό των προτάσεων λύσης

2. ΕΝΤΟΠΙΣΤΕ αντιφάσεις με RAG στήριξη:
   - ΚΥΚΛΙΚΕΣ ΠΡΟΑΠΑΙΤΟΥΜΕΝΕΣ: Κυκλικές εξαρτήσεις δεξιοτήτων
   - ΓΝΩΣΤΙΚΕΣ ΑΝΤΙΦΑΣΕΙΣ: Ασυνεπείς γνωστικές απαιτήσεις
   - ΒΑΘΜΙΔΙΚΕΣ ΑΣΥΝΕΠΕΙΕΣ: Ακατάλληλο επίπεδο για βαθμίδα
   - ΑΝΤΙΦΑΤΙΚΕΣ ΔΕΞΙΟΤΗΤΕΣ: Συγκρουόμενες απαιτήσεις

3. ΠΑΡΕΧΕΤΕ RAG-τεκμηριωμένες προτάσεις:
   - Αναφέρετε παραδείγματα από τα συμφραζόμενα
   - Προτείνετε λύσεις βασισμένες σε επιτυχημένες πρακτικές

JSON ΑΠΑΝΤΗΣΗ:
{{
  "contradictions": [
    {{
      "type": "prerequisite_loop|cognitive_conflict|grade_misalignment|skill_contradiction",
      "severity": "critical|high|medium|low",
      "description": "Περιγραφή αντίφασης",
      "elements": ["στοιχεία"],
      "rag_support": "Πώς τα RAG συμφραζόμενα υποστηρίζουν αυτή την ανάλυση",
      "recommendation": "Πρόταση λύσης βασισμένη στα RAG συμφραζόμενα",
      "similar_cases": "Παρόμοια περιστατικά από RAG"
    }}
  ]],
  "rag_insights": [
    {{
      "insight": "Σημαντική γνώση από RAG ανάλυση",
      "source": "Πηγή από RAG database",
      "relevance": "Σχετικότητα με το πρόγραμμα"
    }}
  ]],
  "overall_assessment": "Γενική αξιολόγηση με RAG στοιχεία",
  "best_practices_from_rag": ["Βέλτιστες πρακτικές από RAG"]
}}"""

        return prompt
    
    def _build_rag_enhanced_cross_prompt(self, curricula_data: Dict, relationships: Dict,
                                       cross_patterns: List[Dict], prereq_context: List[Dict]) -> str:
        """Build RAG-enhanced prompt for cross-curriculum analysis"""
        
        # Format curricula data
        formatted_analysis = "RAG-ENHANCED ΔΙΑΠΡΟΓΡΑΜΜΑΤΙΚΗ ΑΝΑΛΥΣΗ:\n\n"
        
        for name, data in curricula_data.items():
            rel = relationships[name]
            formatted_analysis += f"=== {name} ===\n"
            formatted_analysis += f"Θεματική: {rel['subject_area']} | Βαθμίδες: {', '.join(rel['grade_levels'])}\n"
            formatted_analysis += f"Τίτλος: {data['curriculum_title']}\n\n"
        
        # Add RAG context sections
        rag_section = ""
        if cross_patterns:
            rag_section += "\nΔΙΑΘΕΜΑΤΙΚΑ ΠΡΟΤΥΠΑ (RAG):\n"
            for i, pattern in enumerate(cross_patterns, 1):
                rag_section += f"{i}. {pattern['text'][:200]}... (Πηγή: {pattern['source_file']})\n"
        
        if prereq_context:
            rag_section += "\nΠΡΟΑΠΑΙΤΟΥΜΕΝΑ ΠΡΟΤΥΠΑ (RAG):\n"
            for i, ctx in enumerate(prereq_context, 1):
                rag_section += f"{i}. {ctx['text'][:200]}... (Πηγή: {ctx['source_file']})\n"
        
        prompt = f"""Αναλύστε διαπρογραμματικές αντιφάσεις με RAG-enhanced μεθοδολογία:

{formatted_analysis}

{rag_section}

RAG-ENHANCED ΟΔΗΓΙΕΣ:

1. ΧΡΗΣΙΜΟΠΟΙΗΣΤΕ RAG συμφραζόμενα για:
   - Εντοπισμό κοινών προτύπων αντιφάσεων
   - Σύγκριση με επιτυχημένα διαθεματικά μοντέλα
   - Τεκμηριωμένες προτάσεις λύσεων

2. ΑΝΑΛΥΣΤΕ με RAG στήριξη:
   - PREREQUISITE_GAPS: Χάσματα προαπαιτούμενων με RAG στοιχεία
   - COGNITIVE_REVERSALS: Αντίστροφη γνωστική πορεία
   - GRADE_ANACHRONISMS: Ακατάλληλες βαθμίδες
   - IMPOSSIBLE_PROGRESSIONS: Αδύνατες εξελίξεις

JSON με RAG ENHANCEMENT:
{{
  "contradictions": [
    {{
      "type": "prerequisite_gap|cognitive_reversal|grade_anachronism|impossible_progression",
      "severity": "critical|high|medium",
      "description": "Περιγραφή με RAG στοιχεία",
      "elements": ["στοιχεία"],
      "rag_evidence": "Στοιχεία από RAG που υποστηρίζουν",
      "cross_curricular_impact": "Διαθεματικές επιπτώσεις",
      "recommendation": "Λύση βασισμένη στα RAG προτύπα"
    }}
  ]],
  "rag_supported_synergies": [
    {{
      "description": "Θετικές συνέργιες από RAG ανάλυση",
      "rag_source": "Πηγή RAG στοιχείων"
    }}
  ]],
  "cross_patterns_analysis": "Ανάλυση διαθεματικών προτύπων από RAG"
}}"""

        return prompt
    
    def _build_rag_enhanced_progression_prompt(self, grade_curricula: Dict,
                                             progression_patterns: List[Dict],
                                             skill_context: List[Dict]) -> str:
        """Build RAG-enhanced prompt for progression analysis"""
        
        formatted_progression = ""
        for curriculum, grades in grade_curricula.items():
            formatted_progression += f"\n=== {curriculum} ===\n"
            for grade, content in grades.items():
                formatted_progression += f"ΒΑΘΜΙΔΑ {grade}:\n{content}\n"
        
        # Add RAG context
        rag_section = ""
        if progression_patterns:
            rag_section += "\nΠΡΟΤΥΠΑ ΕΞΕΛΙΞΗΣ (RAG):\n"
            for i, pattern in enumerate(progression_patterns, 1):
                rag_section += f"{i}. {pattern['text'][:250]}... (Πηγή: {pattern['source_file']})\n"
        
        if skill_context:
            rag_section += "\nΑΝΑΠΤΥΞΗ ΔΕΞΙΟΤΗΤΩΝ (RAG):\n"
            for i, ctx in enumerate(skill_context, 1):
                rag_section += f"{i}. {ctx['text'][:250]}... (Πηγή: {ctx['source_file']})\n"
        
        prompt = f"""RAG-ENHANCED ανάλυση εξελικτικής συνοχής μάθησης:

{formatted_progression}

{rag_section}

RAG-ENHANCED ΑΞΙΟΛΟΓΗΣΗ:

1. ΣΥΓΚΡΙΣΗ με RAG πρότυπα:
   - Επιτυχημένα μοντέλα εξέλιξης από database
   - Βέλτιστες πρακτικές προόδου

2. ΕΝΤΟΠΙΣΜΟΣ με RAG στήριξη:
   - Κενά στη λογική προοδo
   - Γνωστικές ασυνέπειες εξέλιξης
   - Προβληματικές προαπαιτούμενες γνώσεις
   - Διακοπές στη συνέχεια μάθησης

JSON ΜΕ RAG INSIGHTS:
{{
  "progression_analysis": {{
    "coherence_score": "0-10",
    "rag_comparison_score": "Σύγκριση με RAG πρότυπα",
    "gaps": ["Κενά εντοπισμένα με RAG"],
    "overlaps": ["Επαναλήψεις βάσει RAG"],
    "rag_benchmarking": "Σύγκριση με καλύτερα προγράμματα"
  }},
  "rag_supported_recommendations": [
    {{
      "recommendation": "Πρόταση βασισμένη στα RAG",
      "rag_justification": "Αιτιολόγηση από RAG στοιχεία",
      "success_examples": "Παραδείγματα επιτυχίας από RAG"
    }}
  ]],
  "best_practices_identified": ["Βέλτιστες πρακτικές από RAG database"]
}}"""

        return prompt


# Factory function to create RAG-enhanced detector
def create_rag_contradiction_detector() -> RAGContradictionDetector:
    """
    Factory function to create a RAG-enhanced contradiction detector
    
    Returns:
        RAGContradictionDetector instance ready for use
    """
    detector = RAGContradictionDetector()
    
    # Verify RAG service is available
    if not detector.rag_service.curriculum_db:
        logger.warning("RAG database is empty. Building curriculum database...")
        try:
            detector.rag_service.build_curriculum_database(force_rebuild=False)
        except Exception as e:
            logger.error(f"Failed to build RAG database: {e}")
            logger.warning("RAG-enhanced detector will have limited functionality")
    
    return detector
