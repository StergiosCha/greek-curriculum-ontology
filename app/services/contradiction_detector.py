"""
COMPLETE FIXED Contradiction Detector with Full Progression Support and Name Validation
Handles all new ontology properties from enhanced extraction
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re
import json
import time
from rdflib import Graph, RDF, RDFS, URIRef, Namespace, Literal

from app.services.llm_service import MultiLLMService, LLMProvider

logger = logging.getLogger(__name__)

CURRICULUM = Namespace("http://curriculum.edu.gr/2022/")
CURRKG = Namespace("http://curriculum-kg.org/ontology/")

class ContradictionDetector:
    """Enhanced LLM-powered contradiction detector with FULL PROGRESSION SUPPORT and NAME VALIDATION"""
    
    def __init__(self):
        self.llm_service = MultiLLMService()
        self.grade_level_mappings = {
            'A_Dimotikou': 1, 'B_Dimotikou': 2, 'C_Dimotikou': 3,
            'D_Dimotikou': 4, 'E_Dimotikou': 5, 'ST_Dimotikou': 6,
            'A_Gymnasiou': 7, 'B_Gymnasiou': 8, 'C_Gymnasiou': 9,
            'A_Lykeiou': 10, 'B_Lykeiou': 11, 'C_Lykeiou': 12,
            'A Gymnasio': 7, 'B Gymnasio': 8, 'C Gymnasio': 9,
            'Α΄ Γυμνασίου': 7, 'Β΄ Γυμνασίου': 8, 'Γ΄ Γυμνασίου': 9
        }
        
    def setup_llm(self, provider: LLMProvider, api_key: str):
        """Setup LLM for semantic analysis"""
        self.llm_service.add_service(provider, api_key)
    
    def load_ontology(self, file_path: Path) -> Graph:
        """Load RDF ontology from file"""
        g = Graph()
        try:
            g.parse(file_path, format="turtle")
            logger.info(f"Loaded ontology with {len(g)} triples from {file_path}")
            return g
        except Exception as e:
            logger.error(f"Error loading ontology {file_path}: {e}")
            return Graph()
    
    def _extract_actual_names(self, curriculum_content: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract actual module and outcome names for validation"""
        names = {
            'modules': [],
            'outcomes': [],
            'strategies': []
        }
        
        for module in curriculum_content.get('modules', []):
            if module.get('title'):
                full_name = module['title']
                if module.get('grade_level'):
                    full_name += f" ({module['grade_level']})"
                names['modules'].append(full_name)
        
        for outcome in curriculum_content.get('learning_outcomes', []):
            if outcome.get('text'):
                text = outcome['text'][:80]
                if outcome.get('grade_levels'):
                    text += f" ({', '.join(outcome['grade_levels'])})"
                names['outcomes'].append(text)
        
        for strategy in curriculum_content.get('assessment_strategies', []):
            if strategy.get('greek_term'):
                names['strategies'].append(strategy['greek_term'])
        
        return names
    
    def _validate_response_names(self, response: Dict[str, Any], actual_names: Dict[str, List[str]]) -> bool:
        """Validate that response uses actual names, not placeholders"""
        
        # Generic placeholders to detect
        generic_patterns = [
            r'Ενότητα [ΑΒΓΔΕΖΗΘIabcdefghxyz]',
            r'Module [ΑΒΓΔΕΖΗΘIabcdefghxyz]',
            r'Στόχος \d+',
            r'Μάθημα [ΑΒΓΔΕΖΗΘIabcdefghxyz]',
            r'Πρόγραμμα [ΑΒΓΔΕΖΗΘIabcdefghxyz]',
            r'element\d+',
            r'Lesson [ΑΒΓΔΕΖΗΘIabcdefghxyz]'
        ]
        
        # Check contradictions for generic names
        for contradiction in response.get('contradictions', []):
            description = contradiction.get('description', '')
            elements = contradiction.get('elements', [])
            
            # Check description and elements
            text_to_check = description + ' ' + ' '.join(elements)
            
            for pattern in generic_patterns:
                if re.search(pattern, text_to_check, re.IGNORECASE):
                    logger.warning(f"Found generic placeholder matching pattern: {pattern}")
                    return False
        
        return True

    def detect_internal_contradictions(self, ontology_path: Path, provider: LLMProvider) -> Dict[str, Any]:
        """Find contradictions within a single curriculum with PROGRESSION ANALYSIS"""
        g = self.load_ontology(ontology_path)
        
        # Extract curriculum content WITH PROGRESSION
        curriculum_content = self._extract_curriculum_content_with_progression(g)
        
        if not curriculum_content:
            return {'contradictions': [], 'analysis': 'No curriculum content found'}
        
        # Extract actual names for validation
        actual_names = self._extract_actual_names(curriculum_content)
        
        # Format for analysis
        formatted_content = self._format_curriculum_with_progression(curriculum_content)
        
        prompt = f"""⚠️⚠️⚠️ ΚΡΙΣΙΜΗ ΑΠΑΙΤΗΣΗ ⚠️⚠️⚠️

ΘΑ ΠΡΕΠΕΙ ΝΑ ΧΡΗΣΙΜΟΠΟΙΗΣΕΤΕ ΜΟΝΟ ΤΑ ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ΑΠΟ ΤΗ ΛΙΣΤΑ ΠΑΡΑΚΑΤΩ!

ΔΙΑΘΕΣΙΜΑ ΠΡΑΓΜΑΤΙΚΑ ΟΝΟΜΑΤΑ ΕΝΟΤΗΤΩΝ:
{chr(10).join(f"- {name}" for name in actual_names['modules'][:15])}

ΔΙΑΘΕΣΙΜΑ ΠΡΑΓΜΑΤΙΚΑ ΜΑΘΗΣΙΑΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ:
{chr(10).join(f"- {name[:100]}" for name in actual_names['outcomes'][:10])}

⚠️ ΑΠΟΛΥΤΑ ΑΠΑΓΟΡΕΥΜΕΝΑ PLACEHOLDERS:
- "Ενότητα Α", "Ενότητα Β", "Ενότητα Γ"
- "Module A", "Module B"
- "Στόχος 1", "Στόχος 2"
- "Μάθημα Χ", "Πρόγραμμα Ψ"
- Οποιοδήποτε γενικό όνομα

Αναλύστε το ακόλουθο περιεχόμενο ελληνικού αναλυτικού προγράμματος για εσωτερικές αντιφάσεις ΚΑΙ προβλήματα προόδου:

ΠΕΡΙΕΧΟΜΕΝΟ ΑΝΑΛΥΤΙΚΟΥ ΠΡΟΓΡΑΜΜΑΤΟΣ ΜΕ ΠΡΟΟΔΟ:
{formatted_content}

ΕΝΤΟΠΙΣΤΕ ΤΙΣ ΑΚΟΛΟΥΘΕΣ ΑΝΤΙΦΑΣΕΙΣ:

1. ΚΥΚΛΙΚΕΣ ΠΡΟΑΠΑΙΤΟΥΜΕΝΕΣ ΓΝΩΣΕΙΣ:
   - Ενότητες που απαιτούν η μία την άλλη κυκλικά
   - Προαπαιτούμενα που δημιουργούν αδιέξοδα

2. ΠΡΟΒΛΗΜΑΤΑ ΠΡΟΟΔΟΥ:
   - Λάθος σειρά διδασκαλίας (προχωρημένο πριν από βασικό)
   - Ελλείπεις προαπαιτούμενες δεξιότητες
   - Πολύ μεγάλα άλματα δυσκολίας

3. ΑΣΥΝΕΠΕΙΕΣ ΥΠΟΣΤΗΡΙΞΗΣ:
   - Μαθητές χρειάζονται υποστήριξη αλλά δεν παρέχεται
   - Υποστήριξη δεν μειώνεται με την πρόοδο

4. ΓΝΩΣΤΙΚΕΣ ΑΝΤΙΦΑΣΕΙΣ:
   - Παρόμοιοι στόχοι με διαφορετικά γνωστικά επίπεδα
   - Ασυνέπειες στην πολυπλοκότητα

5. ΒΑΘΜΙΔΙΚΕΣ ΑΣΥΝΕΠΕΙΕΣ:
   - Υπερβολικά δύσκολες δεξιότητες για τη βαθμίδα
   - Υπερβολικά εύκολοι στόχοι για τη βαθμίδα

ΥΠΟΧΡΕΩΤΙΚΗ ΜΟΡΦΗ ΑΠΑΝΤΗΣΗΣ (ΧΡΗΣΙΜΟΠΟΙΩΝΤΑΣ ΠΡΑΓΜΑΤΙΚΑ ΟΝΟΜΑΤΑ):

❌ ΑΠΑΡΑΔΕΚΤΟ ΠΑΡΑΔΕΙΓΜΑ:
{{
  "contradictions": [
    {{
      "type": "prerequisite_loop",
      "description": "Κυκλική εξάρτηση μεταξύ ενοτήτων",
      "elements": ["Ενότητα Α", "Ενότητα Β"]
    }}
  ]
}}

✅ ΣΩΣΤΟ ΠΑΡΑΔΕΙΓΜΑ (ΜΕ ΠΡΑΓΜΑΤΙΚΑ ΟΝΟΜΑΤΑ):
{{
  "contradictions": [
    {{
      "type": "prerequisite_loop",
      "severity": "critical",
      "description": "Η ενότητα 'Φωνολογία και Φωνητική Α' Δημοτικού' απαιτεί ως προαπαιτούμενο τη 'Μορφολογία Λέξεων Β' Δημοτικού', η οποία με τη σειρά της απαιτεί την 'Φωνολογία' δημιουργώντας κυκλική εξάρτηση που εμποδίζει τη διδασκαλία",
      "elements": [
        "Φωνολογία και Φωνητική (Α' Δημοτικού)",
        "Μορφολογία Λέξεων (Β' Δημοτικού)"
      ],
      "impact": "Οι μαθητές δεν μπορούν να ξεκινήσουν καμία από τις δύο ενότητες χωρίς να έχουν ολοκληρώσει την άλλη",
      "recommendation": "Διαχωρισμός της 'Φωνολογία και Φωνητική' σε 'Βασική Φωνολογία' (χωρίς προαπαιτούμενα) και 'Προχωρημένη Φωνητική' (μετά τη Μορφολογία)"
    }}
  ],
  "progression_quality": {{
    "overall_score": 7,
    "strengths": ["Καλά δομημένες ενότητες στο Α' Δημοτικού"],
    "weaknesses": ["Κυκλικές εξαρτήσεις στο Β' Δημοτικού"]
  }},
  "overall_assessment": "Το πρόγραμμα έχει καλή δομή αλλά χρειάζεται επανασχεδιασμό των προαπαιτούμενων",
  "priority_fixes": ["Επίλυση κυκλικής εξάρτησης Φωνολογίας-Μορφολογίας"]
}}

Απαντήστε σε JSON μορφή:
{{
  "contradictions": [
    {{
      "type": "prerequisite_loop|progression_error|support_inconsistency|cognitive_conflict|grade_misalignment",
      "severity": "critical|high|medium|low",
      "description": "ΛΕΠΤΟΜΕΡΗΣ περιγραφή με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ενοτήτων από το κείμενο",
      "elements": ["ΑΚΡΙΒΟ_ΟΝΟΜΑ_ΕΝΟΤΗΤΑΣ_1 (ΒΑΘΜΙΔΑ)", "ΑΚΡΙΒΟ_ΟΝΟΜΑ_ΕΝΟΤΗΤΑΣ_2 (ΒΑΘΜΙΔΑ)"],
      "impact": "Συγκεκριμένη επίπτωση στη μάθηση",
      "recommendation": "Συγκεκριμένη λύση με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ"
    }}
  ],
  "progression_quality": {{
    "overall_score": "1-10",
    "strengths": ["Δυνατά σημεία με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ"],
    "weaknesses": ["Αδύναμα σημεία με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ"]
  }},
  "overall_assessment": "Γενική αξιολόγηση",
  "priority_fixes": ["Επείγουσες διορθώσεις με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ"]
}}

ΜΟΝΟ JSON - ΤΙΠΟΤΑ ΑΛΛΟ.
ΧΡΗΣΙΜΟΠΟΙΗΣΤΕ ΜΟΝΟ ΤΑ ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ΑΠΟ ΤΗ ΛΙΣΤΑ ΠΟΥ ΔΟΘΗΚΕ."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                parsed_response = self._parse_llm_response(response)
                
                # Validate names
                if self._validate_response_names(parsed_response, actual_names):
                    logger.info(f"Response validation successful on attempt {attempt + 1}")
                    return parsed_response
                else:
                    logger.warning(f"Response validation failed on attempt {attempt + 1} - contains generic placeholders")
                    if attempt < max_retries - 1:
                        # Add stronger warning for retry
                        prompt = f"""ΠΡΟΗΓΟΥΜΕΝΗ ΑΠΑΝΤΗΣΗ ΑΠΟΡΡΙΦΘΗΚΕ - ΧΡΗΣΙΜΟΠΟΙΗΣΑΤΕ ΓΕΝΙΚΑ PLACEHOLDERS!

ΘΑ ΠΡΕΠΕΙ ΝΑ ΧΡΗΣΙΜΟΠΟΙΗΣΕΤΕ ΜΟΝΟ ΑΥΤΑ ΤΑ ΟΝΟΜΑΤΑ:
{chr(10).join(f"✓ {name}" for name in actual_names['modules'][:20])}

""" + prompt
                        time.sleep(1)  # Brief pause before retry
                    
            except Exception as e:
                logger.error(f"LLM analysis failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {'error': str(e), 'contradictions': []}
        
        # If all retries failed validation
        logger.error("All retry attempts failed validation - returning with warning")
        parsed_response['validation_warning'] = "Response contains generic placeholders instead of actual names"
        return parsed_response
    
    def detect_cross_curriculum_contradictions(self, ontology_paths: List[Path], provider: LLMProvider) -> Dict[str, Any]:
        """Find contradictions between curricula WITH PROGRESSION ANALYSIS"""
        
        curricula_content = {}
        subject_relationships = {}
        all_actual_names = {}
        
        for path in ontology_paths:
            g = self.load_ontology(path)
            curriculum_data = self._extract_curriculum_content_with_progression(g)
            if curriculum_data:
                curricula_content[path.stem] = curriculum_data
                all_actual_names[path.stem] = self._extract_actual_names(curriculum_data)
                
                # Extract subject area and grade levels
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
        
        return self._analyze_contradictions_with_progression(curricula_content, subject_relationships, provider, all_actual_names)
    
    def analyze_progression_coherence(self, ontology_paths: List[Path], provider: LLMProvider) -> Dict[str, Any]:
        """Analyze learning progression coherence WITH OUTCOME PROGRESSIONS"""
        
        # Group curricula by grade level WITH OUTCOME LINKS
        grade_curricula = {}
        outcome_progressions = {}
        all_actual_names = {}
        
        for path in ontology_paths:
            g = self.load_ontology(path)
            
            # Extract grade progression
            grade_info = self._extract_grade_progression_enhanced(g)
            if grade_info:
                grade_curricula[path.stem] = grade_info
            
            # Extract outcome progressions
            outcome_prog = self._extract_outcome_progressions(g)
            if outcome_prog:
                outcome_progressions[path.stem] = outcome_prog
            
            # Extract actual names
            curriculum_data = self._extract_curriculum_content_with_progression(g)
            if curriculum_data:
                all_actual_names[path.stem] = self._extract_actual_names(curriculum_data)
        
        if not grade_curricula:
            return {'analysis': 'No grade progression data found'}
        
        # Build names list for validation
        names_summary = "\n\nΔΙΑΘΕΣΙMA ΠΡΑΓΜΑΤΙΚΑ ΟΝΟΜΑΤΑ:"
        for curriculum, names in all_actual_names.items():
            names_summary += f"\n\n{curriculum}:"
            names_summary += f"\n  Ενότητες: {', '.join(names['modules'][:10])}"
        
        formatted_progression = ""
        for curriculum, grades in grade_curricula.items():
            formatted_progression += f"\n=== {curriculum} ===\n"
            for grade, content in grades.items():
                formatted_progression += f"ΒΑΘΜΙΔΑ {grade}:\n{content}\n"
        
        # Add outcome progression data
        if outcome_progressions:
            formatted_progression += "\n\n=== ΠΡΟΟΔΟΣ ΜΑΘΗΣΙΑΚΩΝ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ===\n"
            for curriculum, progressions in outcome_progressions.items():
                formatted_progression += f"\n{curriculum}:\n"
                for prog in progressions:
                    formatted_progression += f"  {prog['from']} → {prog['to']} (Επίπεδο: {prog['level']})\n"
        
        prompt = f"""⚠️⚠️⚠️ ΥΠΟΧΡΕΩΤΙΚΗ ΧΡΗΣΗ ΠΡΑΓΜΑΤΙΚΩΝ ΟΝΟΜΑΤΩΝ ⚠️⚠️⚠️

{names_summary}

Αναλύστε την εξελικτική συνοχή της μάθησης με ΠΛΗΡΗ ΑΝΑΛΥΣΗ ΠΡΟΟΔΟΥ:

{formatted_progression}

ΑΞΙΟΛΟΓΗΣΤΕ:

1. ΛΟΓΙΚΗ ΠΡΟΟΔΟ:
   - Οι δεξιότητες χτίζονται μεταξύ των βαθμίδων;
   - Τα μαθησιακά αποτελέσματα προχωρούν λογικά;
   - Υπάρχουν κενά ή άλματα στη μάθηση;

2. ΓΝΩΣΤΙΚΗ ΕΞΕΛΙΞΗ:
   - Τα γνωστικά επίπεδα αυξάνονται κατάλληλα;
   - Η πολυπλοκότητα αυξάνεται σταδιακά;
   - Υπάρχει κατάλληλη βαθμίδωση δυσκολίας;

3. ΠΡΟΑΠΑΙΤΟΥΜΕΝΕΣ ΓΝΩΣΕΙΣ:
   - Οι προαπαιτούμενες γνώσεις διδάσκονται πρώτα;
   - Τα προαπαιτούμενα μεταξύ ενοτήτων έχουν νόημα;
   - Υπάρχουν ελλείπεις προαπαιτούμενες;

4. ΥΠΟΣΤΗΡΙΞΗ ΜΑΘΗΤΩΝ:
   - Η υποστήριξη μειώνεται κατάλληλα με την πρόοδο;
   - Οι μαθητές γίνονται σταδιακά ανεξάρτητοι;

5. ΣΥΝΕΧΕΙΑ ΜΑΘΗΣΗΣ:
   - Υπάρχει ομαλή μετάβαση μεταξύ βαθμίδων;
   - Αποφεύγονται επαναλήψεις χωρίς λόγο;

⚠️ ΑΠΑΓΟΡΕΥΜΕΝΑ GENERICS:
❌ "Ενότητα Α", "Κενό μεταξύ Ενότητας Α και Β"
❌ "Module X", "Στόχος 1"

✅ ΥΠΟΧΡΕΩΤΙΚΑ ΠΡΑΓΜΑΤΙΚΑ ΟΝΟΜΑΤΑ:
"Φωνολογία (Α' Δημοτικού)", "Σύνταξη (Γ' Δημοτικού)"
"Κενό μεταξύ 'Φωνολογία' (Α' Δημοτικού) και 'Σύνταξη' (Γ' Δημοτικού)"

ΠΑΡΑΔΕΙΓΜΑ ΣΩΣΤΗΣ ΑΠΑΝΤΗΣΗΣ:
{{
  "progression_analysis": {{
    "gaps": [
      "Κενό μεταξύ 'Φωνολογία και Φωνητική' (Α' Δημοτικού) και 'Σύνταξη Πρότασης' (Γ' Δημοτικού) - λείπει ενδιάμεση ενότητα Μορφολογίας"
    ],
    "overlaps": [
      "Η 'Βασική Γραμματική' (Β' Δημοτικού) και 'Εισαγωγή στη Σύνταξη' (Β' Δημοτικού) διδάσκουν το ίδιο περιεχόμενο"
    ]
  }},
  "grade_specific_issues": {{
    "Γ' Δημοτικού": [
      "Η 'Σύνταξη Πρότασης' είναι πολύ προχωρημένη χωρίς την ενδιάμεση ενότητα 'Μορφολογία Λέξεων' στο Β' Δημοτικού"
    ]
  }},
  "recommendations": [
    "Προσθήκη ενότητας 'Μορφολογία Λέξεων' στο Β' Δημοτικού μεταξύ 'Φωνολογία' και 'Σύνταξη'"
  ]
}}

Απαντήστε σε JSON μορφή:
{{
  "progression_analysis": {{
    "coherence_score": "βαθμός_συνοχής_0_10",
    "logical_flow_score": "βαθμός_λογικής_ροής_0_10",
    "gaps": ["Κενά με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ενοτήτων και βαθμίδων"],
    "overlaps": ["Επαναλήψεις με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ενοτήτων και βαθμίδων"],
    "logical_flow": "αξιολόγηση_λογικής_ροής",
    "cognitive_progression": "αξιολόγηση_γνωστικής_εξέλιξης",
    "prerequisite_alignment": "αξιολόγηση_προαπαιτούμενων",
    "support_scaffolding": "αξιολόγηση_υποστήριξης"
  }},
  "grade_specific_issues": {{
    "ΟΝΟΜΑ_ΒΑΘΜΙΔΑΣ": ["Θέματα με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ενοτήτων"]
  }},
  "recommendations": ["Προτάσεις με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ενοτήτων"],
  "restructuring_needed": ["Περιοχές με ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ"]
}}

ΧΡΗΣΙΜΟΠΟΙΗΣΤΕ ΜΟΝΟ ΤΑ ΑΚΡΙΒΗ ΟΝΟΜΑΤΑ ΑΠΟ ΤΗ ΛΙΣΤΑ ΠΑΡΑΠΑΝΩ."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            parsed = self._parse_llm_response(response)
            
            # Validate names in response
            has_generics = False
            for names in all_actual_names.values():
                if not self._validate_response_names(parsed, names):
                    has_generics = True
                    break
            
            if has_generics:
                parsed['validation_warning'] = "Response may contain generic placeholders"
            
            return parsed
        except Exception as e:
            logger.error(f"Progression analysis failed: {e}")
            return {'error': str(e)}

    def _is_hierarchical_parent_child(self, module1_title: str, module2_title: str) -> bool:
        """Check if two modules have parent-child relationship"""
        pattern = r'^(\d+(?:\.\d+)*)\.\s+'
        match1 = re.match(pattern, module1_title)
        match2 = re.match(pattern, module2_title)
        
        if not match1 or not match2:
            return False
        
        num1 = match1.group(1)
        num2 = match2.group(1)
        
        if num2.startswith(num1 + ".") or num1.startswith(num2 + "."):
            return True
        return False


    def _is_umbrella_module(self, module_title: str, module_topics: List[str]) -> bool:
        """Detect umbrella/overview modules"""
        umbrella_indicators = [
            'θεωρητικ', 'theoretical', 'εισαγωγ', 'introduction',
            'γενικ', 'general', 'πλαίσιο', 'framework',
            'προσέγγιση', 'approach', 'επισκόπηση', 'overview'
        ]
        
        title_lower = module_title.lower()
        if any(ind in title_lower for ind in umbrella_indicators):
            return True
        
        if re.match(r'^1\.\s+', module_title) and len(module_topics) > 5:
            return True
        
        return False


    def _extract_module_metadata_with_structure(self, modules: List[Dict]) -> List[Dict]:
        """Add hierarchical structure metadata"""
        enhanced = []
        
        for module in modules:
            enh = module.copy()
            title = module.get('title', '')
            topics = module.get('topics', [])
            
            enh['is_umbrella'] = self._is_umbrella_module(title, topics)
            
            match = re.match(r'^(\d+(?:\.\d+)*)\.\s+', title)
            if match:
                enh['numbering'] = match.group(1)
                enh['depth'] = len(match.group(1).split('.'))
            else:
                enh['numbering'] = None
                enh['depth'] = 0
            
            enh['children'] = []
            enh['parent'] = None
            
            for other in modules:
                other_title = other.get('title', '')
                if self._is_hierarchical_parent_child(title, other_title):
                    if enh.get('numbering') and other_title.startswith(enh['numbering'] + '.'):
                        enh['children'].append(other_title)
                    else:
                        enh['parent'] = other_title
            
            enhanced.append(enh)
        
        return enhanced
    def _extract_curriculum_content_with_progression(self, g: Graph) -> Dict[str, Any]:
        """Extract curriculum content WITH ALL PROGRESSION DATA"""
        
        # Get curriculum name/title
        curriculum_title = ""
        for subj, pred, obj in g.triples((None, CURRKG.hasTitle, None)):
            if "Curriculum" in str(subj):
                curriculum_title = str(obj)
                break
        
        # Extract modules with FULL PROGRESSION
        modules = []
        for subj, pred, obj in g.triples((None, RDF.type, CURRKG.Module)):
            
            # Get prerequisites
            prerequisites = []
            for _, _, prereq_uri in g.triples((subj, CURRKG.hasPrerequisite, None)):
                prereq_title = self._get_property_value(g, prereq_uri, "hasTitle")
                if prereq_title:
                    prerequisites.append(prereq_title)
            
            # Get progression level
            progression_level = ""
            for _, _, prog_uri in g.triples((subj, CURRKG.hasProgressionLevel, None)):
                progression_level = str(prog_uri).split('/')[-1].replace('currkg:', '')
            
            # Get complexity indicators
            cognitive_level = ""
            independence_level = ""
            for _, _, cog_uri in g.triples((subj, CURRKG.cognitiveLevel, None)):
                cognitive_level = str(cog_uri).split('/')[-1].replace('currkg:', '')
            for _, _, ind_uri in g.triples((subj, CURRKG.independenceLevel, None)):
                independence_level = str(ind_uri).split('/')[-1].replace('currkg:', '')
            
            module_data = {
                'uri': str(subj),
                'title': self._get_property_value(g, subj, "hasTitle"),
                'description': self._get_property_value(g, subj, "hasDescription"),
                'grade_level': self._get_property_value(g, subj, "hasGradeLevel"),
                'progression_level': progression_level,
                'cognitive_level': cognitive_level,
                'independence_level': independence_level,
                'topics': self._get_all_topic_descriptions(g, subj),
                'prerequisites': prerequisites,
                'level': self._get_property_value(g, subj, "hasLevel"),
                'category': self._get_property_value(g, subj, "belongsTo"),
                'curriculum_title': curriculum_title  # Add curriculum title to each module
            }
            modules.append(module_data)
        
        # CRITICAL FIX: Sort modules by URI number to preserve document order
        # RDF graphs don't guarantee order, so we must sort explicitly
        def get_module_number(module):
            """Extract module number from URI like 'currkg:Module_1' """
            uri = module.get('uri', '')
            match = re.search(r'Module_(\d+)', uri)
            return int(match.group(1)) if match else 999
        
        modules.sort(key=get_module_number)
        logger.info(f"Extracted and sorted {len(modules)} modules by URI number")
        if modules:
            logger.info(f"Module order: {[m.get('title', 'Unknown')[:40] for m in modules[:5]]}")
        
        # Extract learning outcomes WITH PROGRESSION
        learning_outcomes = []
        for subj, pred, obj in g.triples((None, RDF.type, CURRKG.LearningOutcome)):
            
            # Get progression level
            progression_level = ""
            for _, _, prog_uri in g.triples((subj, CURRKG.progressionLevel, None)):
                progression_level = str(prog_uri).split('/')[-1].replace('currkg:', '')
            
            # Get support level
            support_level = ""
            for _, _, supp_uri in g.triples((subj, CURRKG.supportLevel, None)):
                support_level = str(supp_uri).split('/')[-1].replace('currkg:', '')
            
            # Get bloom level
            bloom_level = ""
            for _, _, bloom_uri in g.triples((subj, CURRKG.bloomLevel, None)):
                bloom_level = str(bloom_uri).split('/')[-1].replace('currkg:', '')
            
            # Get skill category
            skill_category = ""
            for _, _, skill_uri in g.triples((subj, CURRKG.skillCategory, None)):
                skill_category = str(skill_uri).split('/')[-1].replace('currkg:', '')
            
            # Get grades
            grades = []
            for _, _, grade_uri in g.triples((subj, CURRKG.applicableToGrade, None)):
                grade = str(grade_uri).split('/')[-1].replace('currkg:', '')
                grades.append(grade)
            
            # Get progressions to other outcomes
            progresses_to = []
            for _, _, target_uri in g.triples((subj, CURRKG.progressesTo, None)):
                progresses_to.append(str(target_uri))
            
            outcome_data = {
                'uri': str(subj),
                'text': self._get_property_value(g, subj, "hasText"),
                'grade_levels': grades,
                'progression_level': progression_level,
                'support_level': support_level,
                'bloom_level': bloom_level,
                'skill_category': skill_category,
                'progresses_to': progresses_to
            }
            learning_outcomes.append(outcome_data)
        
        # Extract assessment strategies WITH PROGRESSION
        assessment_strategies = []
        for subj, pred, obj in g.triples((None, RDF.type, CURRKG.AssessmentStrategy)):
            
            # Get assessment progression
            assessment_prog = ""
            for _, _, prog_uri in g.triples((subj, CURRKG.assessmentProgression, None)):
                assessment_prog = str(prog_uri).split('/')[-1].replace('currkg:', '')
            
            # Get complexity level
            complexity = ""
            for _, _, comp_uri in g.triples((subj, CURRKG.complexityLevel, None)):
                complexity = str(comp_uri).split('/')[-1].replace('currkg:', '')
            
            strategy_data = {
                'uri': str(subj),
                'type': self._get_property_value(g, subj, "strategyType"),
                'greek_term': self._get_property_value(g, subj, "greekTerm"),
                'complexity_level': complexity,
                'assessment_progression': assessment_prog,
                'progression_notes': self._get_property_value(g, subj, "progressionNotes")
            }
            assessment_strategies.append(strategy_data)
        
        # Extract teaching strategies WITH PROGRESSION
        teaching_strategies = []
        for subj, pred, obj in g.triples((None, RDF.type, CURRKG.TeachingStrategy)):
            
            # Get scaffolding type
            scaffolding = ""
            for _, _, scaff_uri in g.triples((subj, CURRKG.scaffoldingType, None)):
                scaffolding = str(scaff_uri).split('/')[-1].replace('currkg:', '')
            
            # Get teaching stage
            teaching_stage = ""
            for _, _, stage_uri in g.triples((subj, CURRKG.teachingStage, None)):
                teaching_stage = str(stage_uri).split('/')[-1].replace('currkg:', '')
            
            strategy_data = {
                'uri': str(subj),
                'name': self._get_property_value(g, subj, "strategyName"),
                'scaffolding_type': scaffolding,
                'teaching_stage': teaching_stage,
                'progression_notes': self._get_property_value(g, subj, "progressionNotes")
            }
            teaching_strategies.append(strategy_data)
        
        return {
            'curriculum_title': curriculum_title,
            'modules': modules,
            'learning_outcomes': learning_outcomes,
            'assessment_strategies': assessment_strategies,
            'teaching_strategies': teaching_strategies
        }
    def _extract_module_order_from_ontology(self, g: Graph) -> Dict[str, int]:
        """Extract module order from URI numbers (Module_1, Module_2, etc.)"""
        module_order = {}
        
        for subj, pred, obj in g.triples((None, RDF.type, CURRKG.Module)):
            uri_str = str(subj)
            # Extract number from URI like "currkg:Module_1"
            match = re.search(r'Module_(\d+)', uri_str)
            if match:
                module_number = int(match.group(1))
                module_title = self._get_property_value(g, subj, "hasTitle")
                if module_title:
                    module_order[module_title] = module_number
        
        return module_order
    def _extract_outcome_progressions(self, g: Graph) -> List[Dict[str, str]]:
        """Extract outcome progression relationships"""
        progressions = []
        
        for source, pred, target in g.triples((None, CURRKG.progressesTo, None)):
            source_text = self._get_property_value(g, source, "hasText")
            target_text = self._get_property_value(g, target, "hasText")
            
            source_level = ""
            for _, _, level_uri in g.triples((source, CURRKG.progressionLevel, None)):
                source_level = str(level_uri).split('/')[-1]
            
            if source_text and target_text:
                progressions.append({
                    'from': source_text[:100] + "..." if len(source_text) > 100 else source_text,
                    'to': target_text[:100] + "..." if len(target_text) > 100 else target_text,
                    'level': source_level
                })
        
        return progressions
    
    def _extract_grade_progression_enhanced(self, g: Graph) -> Dict[str, str]:
        """Extract grade-level progression WITH COMPLEXITY INFO"""
        grades = {}
        
        for subj, pred, obj in g.triples((None, RDF.type, CURRKG.Module)):
            grade_level = self._get_property_value(g, subj, "hasGradeLevel")
            if grade_level:
                module_title = self._get_property_value(g, subj, "hasTitle")
                module_desc = self._get_property_value(g, subj, "hasDescription")
                
                # Get progression info
                progression_level = ""
                for _, _, prog_uri in g.triples((subj, CURRKG.hasProgressionLevel, None)):
                    progression_level = str(prog_uri).split('/')[-1]
                
                cognitive_level = ""
                for _, _, cog_uri in g.triples((subj, CURRKG.cognitiveLevel, None)):
                    cognitive_level = str(cog_uri).split('/')[-1]
                
                if grade_level not in grades:
                    grades[grade_level] = []
                
                content = f"Ενότητα: {module_title}"
                if module_desc:
                    content += f" - {module_desc}"
                if progression_level:
                    content += f" [Πρόοδος: {progression_level}]"
                if cognitive_level:
                    content += f" [Γνωστικό: {cognitive_level}]"
                
                grades[grade_level].append(content)
        
        # Format for LLM
        formatted_grades = {}
        for grade, content_list in grades.items():
            formatted_grades[grade] = '\n'.join(content_list)
        
        return formatted_grades
    
    def _format_curriculum_with_progression(self, curriculum_data: Dict[str, Any]) -> str:
        """Format curriculum data WITH ALL PROGRESSION INFO"""
        
        formatted = f"ΤΙΤΛΟΣ: {curriculum_data['curriculum_title']}\n\n"
        
        formatted += "=" * 80 + "\n"
        formatted += "ΕΝΟΤΗΤΕΣ ΜΕ ΠΡΟΟΔΟ:\n"
        formatted += "=" * 80 + "\n"
        for i, module in enumerate(curriculum_data['modules'], 1):
            formatted += f"\n{i}. {module['title']}\n"
            if module['description']:
                formatted += f"   Περιγραφή: {module['description']}\n"
            if module['grade_level']:
                formatted += f"   Βαθμίδα: {module['grade_level']}\n"
            if module['progression_level']:
                formatted += f"   Επίπεδο Προόδου: {module['progression_level']}\n"
            if module['cognitive_level']:
                formatted += f"   Γνωστικό Επίπεδο: {module['cognitive_level']}\n"
            if module['independence_level']:
                formatted += f"   Επίπεδο Ανεξαρτησίας: {module['independence_level']}\n"
            if module['level']:
                formatted += f"   Επίπεδο: {module['level']}\n"
            if module['topics']:
                formatted += f"   Θέματα: {', '.join(module['topics'][:5])}\n"
            if module['prerequisites']:
                formatted += f"   Προαπαιτούμενα: {', '.join(module['prerequisites'])}\n"
            formatted += "\n"
        
        if curriculum_data.get('learning_outcomes'):
            formatted += "\n" + "=" * 80 + "\n"
            formatted += "ΜΑΘΗΣΙΑΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΕ ΠΡΟΟΔΟ:\n"
            formatted += "=" * 80 + "\n"
            for i, outcome in enumerate(curriculum_data['learning_outcomes'], 1):
                formatted += f"\n{i}. {outcome['text'][:150]}...\n" if len(outcome['text']) > 150 else f"\n{i}. {outcome['text']}\n"
                if outcome['grade_levels']:
                    formatted += f"   Βαθμίδες: {', '.join(outcome['grade_levels'])}\n"
                if outcome['progression_level']:
                    formatted += f"   Επίπεδο Προόδου: {outcome['progression_level']}\n"
                if outcome['support_level']:
                    formatted += f"   Επίπεδο Υποστήριξης: {outcome['support_level']}\n"
                if outcome['bloom_level']:
                    formatted += f"   Bloom Επίπεδο: {outcome['bloom_level']}\n"
                if outcome['skill_category']:
                    formatted += f"   Κατηγορία Δεξιότητας: {outcome['skill_category']}\n"
                if outcome['progresses_to']:
                    formatted += f"   Προχωρά σε: {len(outcome['progresses_to'])} επόμενα αποτελέσματα\n"
        
        if curriculum_data.get('assessment_strategies'):
            formatted += "\n" + "=" * 80 + "\n"
            formatted += "ΣΤΡΑΤΗΓΙΚΕΣ ΑΞΙΟΛΟΓΗΣΗΣ ΜΕ ΠΡΟΟΔΟ:\n"
            formatted += "=" * 80 + "\n"
            for i, strategy in enumerate(curriculum_data['assessment_strategies'], 1):
                formatted += f"\n{i}. {strategy['greek_term']}\n"
                if strategy['type']:
                    formatted += f"   Τύπος: {strategy['type']}\n"
                if strategy['complexity_level']:
                    formatted += f"   Επίπεδο Πολυπλοκότητας: {strategy['complexity_level']}\n"
                if strategy['assessment_progression']:
                    formatted += f"   Πρόοδος Αξιολόγησης: {strategy['assessment_progression']}\n"
                if strategy['progression_notes']:
                    formatted += f"   Σημειώσεις: {strategy['progression_notes']}\n"
        
        if curriculum_data.get('teaching_strategies'):
            formatted += "\n" + "=" * 80 + "\n"
            formatted += "ΣΤΡΑΤΗΓΙΚΕΣ ΔΙΔΑΣΚΑΛΙΑΣ ΜΕ ΠΡΟΟΔΟ:\n"
            formatted += "=" * 80 + "\n"
            for i, strategy in enumerate(curriculum_data['teaching_strategies'], 1):
                formatted += f"\n{i}. {strategy['name']}\n"
                if strategy['scaffolding_type']:
                    formatted += f"   Τύπος Υποστήριξης: {strategy['scaffolding_type']}\n"
                if strategy['teaching_stage']:
                    formatted += f"   Στάδιο Διδασκαλίας: {strategy['teaching_stage']}\n"
                if strategy['progression_notes']:
                    formatted += f"   Σημειώσεις: {strategy['progression_notes']}\n"
        
        return formatted
    
    def _analyze_contradictions_with_progression(self, curricula_data: Dict, relationships: Dict, provider: LLMProvider, all_actual_names: Dict) -> Dict[str, Any]:
        """Analyze contradictions WITH PROGRESSION CONTEXT and HIERARCHICAL STRUCTURE"""
        
        names_list = "\n\n⚠️⚠️⚠️ ΔΙΑΘΕΣΙΜΑ ΠΡΑΓΜΑΤΙΚΑ ΟΝΟΜΑΤΑ ⚠️⚠️⚠️\n"
        for curriculum, names in all_actual_names.items():
            names_list += f"\n{curriculum}:\n"
            names_list += f"  Ενότητες: {', '.join(names['modules'][:15])}\n"
            if names['outcomes']:
                names_list += f"  Αποτελέσματα: {', '.join([o[:50] for o in names['outcomes'][:5]])}\n"
        
        formatted_analysis = "ΑΝΑΛΥΣΗ ΑΝΤΙΦΑΣΕΩΝ ΜΕ ΠΡΟΟΔΟ ΚΑΙ ΙΕΡΑΡΧΙΚΗ ΔΟΜΗ:\n\n"
        
        for name, data in curricula_data.items():
            rel = relationships[name]
            formatted_analysis += f"{'=' * 80}\n"
            formatted_analysis += f"=== {name} ===\n"
            formatted_analysis += f"{'=' * 80}\n"
            formatted_analysis += f"Θεματική Περιοχή: {rel['subject_area']}\n"
            formatted_analysis += f"Βαθμίδες: {', '.join(rel['grade_levels'])}\n"
            formatted_analysis += f"Τίτλος: {data['curriculum_title']}\n\n"
            
            modules_structured = self._extract_module_metadata_with_structure(data['modules'])
            
            # BUILD COMPARISON TABLE
            formatted_analysis += "┌" + "─" * 78 + "┐\n"
            formatted_analysis += "│ ΣΕΙΡΑ ΔΙΔΑΣΚΑΛΙΑΣ - ΔΙΑΒΑΣΤΕ ΑΠΟ ΠΑΝΩ ΠΡΟΣ ΤΑ ΚΑΤΩ" + " " * 24 + "│\n"
            formatted_analysis += "├" + "─" * 78 + "┤\n"
            formatted_analysis += "│ ΠΡΩΤΗ → ΔΕΥΤΕΡΗ → ΤΡΙΤΗ (χρονολογική σειρά)" + " " * 31 + "│\n"
            formatted_analysis += "└" + "─" * 78 + "┘\n\n"
            
            for i, module in enumerate(modules_structured, 1):
                indent = "  " * (module.get('depth', 1) - 1)
                cog_level = module.get('cognitive_level', 'N/A')
                
                formatted_analysis += f"{indent}POSITION {i} (taught #{i}):\n"
                formatted_analysis += f"{indent}  Title: {module['title']}\n"
                formatted_analysis += f"{indent}  Cognitive: {cog_level}\n"
                
                if module.get('is_umbrella'):
                    formatted_analysis += f"{indent}  Type: ⚠️ UMBRELLA\n"
                if module.get('children'):
                    formatted_analysis += f"{indent}  Type: 👥 PARENT\n"
                if module.get('parent'):
                    formatted_analysis += f"{indent}  Type: 📁 CHILD of {module['parent']}\n"
                
                formatted_analysis += "\n"
            
            # COMPARISON TABLE
            formatted_analysis += "PROGRESSION CHECK:\n"
            formatted_analysis += "┌─────────┬──────────────────────────────┬──────────────┐\n"
            formatted_analysis += "│ ORDER   │ MODULE TITLE                 │ COGNITIVE    │\n"
            formatted_analysis += "├─────────┼──────────────────────────────┼──────────────┤\n"
            
            for i, module in enumerate(modules_structured, 1):
                title_short = module['title'][:28].ljust(28)
                cog = module.get('cognitive_level', 'Unknown')[:12].ljust(12)
                formatted_analysis += f"│ #{i}      │ {title_short} │ {cog} │\n"
                
                if i < len(modules_structured):
                    next_module = modules_structured[i]
                    curr_level = {'Foundational': 1, 'Medium': 2, 'High': 3}.get(module.get('cognitive_level'), 0)
                    next_level = {'Foundational': 1, 'Medium': 2, 'High': 3}.get(next_module.get('cognitive_level'), 0)
                    
                    if curr_level < next_level:
                        arrow = "↗ INCREASE"
                    elif curr_level == next_level:
                        arrow = "→ SAME"
                    else:
                        arrow = "↘ DECREASE"
                    
                    formatted_analysis += f"│         │ {arrow.ljust(28)} │              │\n"
            
            formatted_analysis += "└─────────┴──────────────────────────────┴──────────────┘\n\n"
            
            # EXPLICIT FLOW
            cognitive_levels = []
            for m in modules_structured:
                cog = m.get('cognitive_level', 'Unknown')
                level_num = {'Foundational': '1', 'Medium': '2', 'High': '3'}.get(cog, '?')
                cognitive_levels.append(f"{cog}({level_num})")
            
            formatted_analysis += f"COGNITIVE FLOW: {' → '.join(cognitive_levels)}\n"
            formatted_analysis += "=" * 80 + "\n\n"
        
        prompt = f"""{names_list}

    {formatted_analysis}

    ⚠️⚠️⚠️ ΚΡΙΣΙΜΟ: ΠΩΣ ΝΑ ΔΙΑΒΑΣΕΤΕ ΤΗ ΣΕΙΡΑ ⚠️⚠️⚠️

    Η σειρά διδασκαλίας είναι ΑΠΟ ΠΑΝΩ ΠΡΟΣ ΤΑ ΚΑΤΩ στον πίνακα:

    POSITION 1 → διδάσκεται ΠΡΩΤΗ (χρονικά πρώτη)
    POSITION 2 → διδάσκεται ΔΕΥΤΕΡΗ (μετά την 1)
    POSITION 3 → διδάσκεται ΤΡΙΤΗ (μετά την 2)

    ΠΑΡΑΔΕΙΓΜΑ ΑΝΑΓΝΩΣΗΣ:

    ┌─────────┬──────────────────────────────┬──────────────┐
    │ ORDER   │ MODULE TITLE                 │ COGNITIVE    │
    ├─────────┼──────────────────────────────┼──────────────┤
    │ #1      │ Θεωρητική προσέγγιση         │ Foundational │ ← TAUGHT FIRST
    │         │ ↗ INCREASE                   │              │
    │ #2      │ Θεματικές ενότητες           │ Foundational │ ← TAUGHT SECOND
    │         │ ↗ INCREASE                   │              │
    │ #3      │ Διδακτική μεθοδολογία        │ Medium       │ ← TAUGHT THIRD
    └─────────┴──────────────────────────────┴──────────────┘

    Αυτό σημαίνει:
    - #1 (Foundational) comes BEFORE #3 (Medium) chronologically
    - #2 (Foundational) comes BEFORE #3 (Medium) chronologically  
    - #3 (Medium) comes AFTER #1 and #2 chronologically

    Flow: Foundational(1) → Foundational(1) → Medium(2)
    Math: 1 → 1 → 2 = NO DECREASE = NO CONTRADICTION ✅

    ❌ WRONG INTERPRETATION:
    "#3 (Medium) προηγείται #2 (Foundational)"
    NO! #3 has higher order number, so it comes LATER!

    ✅ CORRECT INTERPRETATION:
    "#2 (Foundational) προηγείται #3 (Medium)"
    YES! Lower order number comes first!

    RULE: Lower ORDER number = Taught EARLIER

    ΚΑΝΟΝΑΣ ΓΙΑ PROGRESSION_REVERSAL:

    ONLY flag if ORDER #N has HIGHER cognitive level than ORDER #(N+1):

    Example 1:
    ORDER #1: Medium (2)
    ORDER #2: Foundational (1)
    2 > 1 → REVERSAL ❌

    Example 2:
    ORDER #1: Foundational (1)
    ORDER #2: Medium (2)
    1 < 2 → NO REVERSAL ✅

    Example 3:
    ORDER #1: Foundational (1)
    ORDER #2: Foundational (1)
    ORDER #3: Medium (2)
    1 → 1 → 2 → NO REVERSAL ✅

    ΜΑΘΗΜΑΤΙΚΟΣ ΕΛΕΓΧΟΣ:

    For each consecutive pair (ORDER #i, ORDER #(i+1)):
    - Get cognitive_level_i as number (Foundational=1, Medium=2, High=3)
    - Get cognitive_level_(i+1) as number
    - IF cognitive_level_i > cognitive_level_(i+1):
        → FLAG as PROGRESSION_REVERSAL
    - ELSE:
        → NO PROBLEM

    ❌ ΜΗΝ ΣΥΓΚΡΙΝΕΤΕ:
    - Parent με Child (διαφορετικό depth)
    - Umbrella modules (⚠️)
    - Non-consecutive modules

    ✅ ΜΟΝΟ ΣΥΓΚΡΙΝΕΤΕ:
    - ORDER #1 με ORDER #2
    - ORDER #2 με ORDER #3
    - Consecutive orders only

    JSON ΜΟΡΦΗ:
    {{
    "contradictions": [
        {{
        "type": "progression_reversal",
        "severity": "high",
        "description": "ORDER #X (CognitiveLevel=Y) followed by ORDER #(X+1) (CognitiveLevel=Z) where Y > Z",
        "elements": ["ORDER #X: title", "ORDER #(X+1): title"],
        "impact": "Students learn advanced before basic",
        "recommendation": "Swap order"
        }}
    ],
    "normal_progressions": [
        "List progressions that are CORRECT (increasing or same level)"
    ]
    }}

    CRITICAL: Use ORDER numbers from the table. Lower ORDER = Earlier teaching."""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.llm_service.generate_with_provider(provider, prompt)
                parsed = self._parse_llm_response(response)
                
                has_valid_names = True
                for names in all_actual_names.values():
                    if not self._validate_response_names(parsed, names):
                        has_valid_names = False
                        break
                
                if has_valid_names or attempt == max_retries - 1:
                    if not has_valid_names:
                        parsed['validation_warning'] = "Response contains generic placeholders"
                    return parsed
                
                logger.warning(f"Validation failed on attempt {attempt + 1}")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Analysis failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {'error': str(e), 'contradictions': []}
        
        return {'error': 'All validation attempts failed', 'contradictions': []}
    def _is_language_related_subject(self, subject_area: str, title: str) -> bool:
        """Determine if subjects are language-related"""
        language_indicators = [
            'γλώσσα', 'λογοτεχνία', 'language', 'literature', 
            'ελληνικ', 'greek', 'φιλολογ', 'κείμεν'
        ]
        
        text_to_check = f"{subject_area} {title}".lower()
        return any(indicator in text_to_check for indicator in language_indicators)
    
    def _infer_subject_area(self, title: str) -> str:
        """Infer subject area from curriculum title"""
        if not title:
            return 'general'
            
        title_lower = title.lower()
        
        if any(term in title_lower for term in ['γλώσσα', 'language', 'γλωσσ']):
            return 'greek_language'
        elif any(term in title_lower for term in ['λογοτεχνία', 'literature', 'λογοτεχν']):
            return 'literature'
        elif any(term in title_lower for term in ['μαθηματικ', 'mathematics', 'math']):
            return 'mathematics'
        elif any(term in title_lower for term in ['ιστορί', 'history', 'hist']):
            return 'history'
        elif any(term in title_lower for term in ['φυσικ', 'physics', 'science']):
            return 'science'
        else:
            return 'general'
    
    def _get_all_topic_descriptions(self, g: Graph, module_uri: URIRef) -> List[str]:
        """Get all topic descriptions for a module"""
        topics = []
        
        for _, _, topic_uri in g.triples((module_uri, CURRKG.coversTopic, None)):
            topic_desc = self._get_property_value(g, topic_uri, "hasDescription")
            if not topic_desc:
                topic_desc = self._get_property_value(g, topic_uri, "asString")
            if topic_desc:
                topics.append(topic_desc)
        
        return topics
    
    def _get_property_value(self, g: Graph, subject: URIRef, property_name: str) -> str:
        """Get single property value from multiple possible namespaces"""
        # Try CURRKG namespace first
        prop_uri = URIRef(f"http://curriculum-kg.org/ontology/{property_name}")
        for _, _, obj in g.triples((subject, prop_uri, None)):
            return str(obj)
        
        # Try CURRICULUM namespace as fallback
        prop_uri = URIRef(f"http://curriculum.edu.gr/2022/{property_name}")
        for _, _, obj in g.triples((subject, prop_uri, None)):
            return str(obj)
        
        return ""

    def _get_all_property_values(self, g: Graph, subject: URIRef, property_name: str) -> List[str]:
        """Get all values for a property"""
        values = []
        
        prop_uri = URIRef(f"http://curriculum-kg.org/ontology/{property_name}")
        for _, _, obj in g.triples((subject, prop_uri, None)):
            values.append(str(obj))
        
        prop_uri = URIRef(f"http://curriculum.edu.gr/2022/{property_name}")
        for _, _, obj in g.triples((subject, prop_uri, None)):
            values.append(str(obj))
        
        return values
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Improved LLM response parser with multiple fallback strategies"""
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")
        
        # Try multiple regex patterns
        json_patterns = [
            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # Balanced braces
            r'\{.*\}',  # Simple greedy
        ]
        
        for i, pattern in enumerate(json_patterns, 1):
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    cleaned_match = match.strip()
                    parsed = json.loads(cleaned_match)
                    logger.info(f"Successfully parsed JSON using pattern {i}")
                    return parsed
                except json.JSONDecodeError:
                    continue
        
        # Brace extraction
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = response[first_brace:last_brace + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                pass
        
        # Fallback
        logger.error(f"JSON parsing failed. Response: {response[:200]}...")
        return {
            "contradictions": [],
            "analysis": "JSON parsing failed - malformed response", 
            "raw_response": response[:500],
            "parsing_error": True
        }
    
    def generate_contradiction_report(self, 
                                    internal_results: Dict[str, Any],
                                    cross_results: Dict[str, Any],
                                    progression_results: Dict[str, Any],
                                    provider: LLMProvider) -> str:
        """Generate comprehensive report WITH PROGRESSION ANALYSIS"""
        
        prompt = f"""Δημιουργήστε ολοκληρωμένη αναφορά αντιφάσεων ΜΕ ΑΝΑΛΥΣΗ ΠΡΟΟΔΟΥ:

ΕΣΩΤΕΡΙΚΕΣ ΑΝΤΙΦΑΣΕΙΣ:
{json.dumps(internal_results, ensure_ascii=False, indent=2)}

ΔΙΑΠΡΟΓΡΑΜΜΑΤΙΚΕΣ ΑΝΤΙΦΑΣΕΙΣ:
{json.dumps(cross_results, ensure_ascii=False, indent=2)}

ΑΝΑΛΥΣΗ ΠΡΟΟΔΟΥ:
{json.dumps(progression_results, ensure_ascii=False, indent=2)}

Δημιουργήστε δομημένη αναφορά:

1. ΕΚΤΕΛΕΣΤΙΚΗ ΠΕΡΙΛΗΨΗ
   - Συνολική εκτίμηση ποιότητας
   - Ποιότητα προόδου μάθησης
   - Κρίσιμα προβλήματα

2. ΑΝΑΛΥΣΗ ΠΡΟΟΔΟΥ
   - Πόσο καλά προοδεύουν οι μαθητές
   - Λογική ροή μάθησης
   - Προβλήματα στην εξέλιξη δυσκολίας

3. ΠΡΑΓΜΑΤΙΚΕΣ ΑΝΤΙΦΑΣΕΙΣ
   - Μόνο σοβαρά προβλήματα
   - Συγκεκριμένα παραδείγματα
   - Επίπτωση στη μάθηση

4. ΘΕΤΙΚΑ ΣΤΟΙΧΕΙΑ
   - Καλά σχεδιασμένες προόδους
   - Συνέργιες μεταξύ προγραμμάτων

5. ΣΥΣΤΑΣΕΙΣ
   - Προτεραιότητες δράσης
   - Συγκεκριμένες βελτιώσεις

Τονίστε τα θετικά πριν από τα προβλήματα."""

        try:
            return self.llm_service.generate_with_provider(provider, prompt)
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Αποτυχία δημιουργίας αναφοράς: {str(e)}"