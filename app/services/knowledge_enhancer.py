import requests
import time
from typing import List, Dict, Any, Optional
import logging
from urllib.parse import quote
import json
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

class KnowledgeEnhancer:
    """Real implementation for external knowledge enhancement"""
    
    def __init__(self):
        self.cache_file = Path(settings.cache_dir) / "knowledge_cache.json"
        self.cache = self._load_cache()
        
    def enhance_learning_objective(self, objective_text: str, subject: str = "language") -> Dict[str, Any]:
        """Enhance learning objective with external knowledge"""
        enhancement = {
            'bloom_taxonomy': self._get_bloom_mapping(objective_text),
            'cefr_alignment': self._get_cefr_alignment(objective_text, subject),
            'international_standards': self._get_international_standards(objective_text, subject),
            'pedagogical_approaches': self._get_pedagogical_approaches(objective_text),
            'assessment_methods': self._get_assessment_methods(objective_text)
        }
        
        return enhancement
    
    def get_dbpedia_facts(self, concept: str, language: str = "el") -> List[Dict[str, str]]:
        """Get facts from DBpedia"""
        cache_key = f"dbpedia_{concept}_{language}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try Greek first, then English
            for lang in [language, "en"]:
                entity_uri = f"http://dbpedia.org/resource/{quote(concept.replace(' ', '_'))}"
                
                sparql_query = f"""
                SELECT DISTINCT ?property ?value WHERE {{
                    <{entity_uri}> ?property ?value .
                    FILTER(?property IN (
                        dbo:abstract, dbo:type, dbo:subject, 
                        dbo:field, dbo:academicDiscipline, dbo:education
                    ))
                    FILTER(LANG(?value) = "{lang}" || !isLiteral(?value))
                }}
                LIMIT 10
                """
                
                response = requests.get(
                    "https://dbpedia.org/sparql",
                    params={'query': sparql_query, 'format': 'json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    facts = []
                    
                    for binding in data.get('results', {}).get('bindings', []):
                        prop = binding.get('property', {}).get('value', '')
                        value = binding.get('value', {}).get('value', '')
                        
                        if prop and value:
                            facts.append({
                                'property': prop.split('/')[-1],
                                'value': value[:200],  # Limit length
                                'source': 'dbpedia'
                            })
                    
                    if facts:
                        self.cache[cache_key] = facts
                        self._save_cache()
                        return facts
                
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.warning(f"DBpedia query failed for {concept}: {e}")
        
        return []
    
    def get_wikidata_facts(self, concept: str) -> List[Dict[str, str]]:
        """Get facts from Wikidata"""
        cache_key = f"wikidata_{concept}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Search for entity
            search_url = f"https://www.wikidata.org/w/api.php"
            search_params = {
                'action': 'wbsearchentities',
                'search': concept,
                'language': 'el',
                'format': 'json',
                'limit': 1
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            
            if response.status_code == 200:
                search_data = response.json()
                entities = search_data.get('search', [])
                
                if entities:
                    entity_id = entities[0]['id']
                    
                    # Get entity data
                    entity_url = f"https://www.wikidata.org/w/api.php"
                    entity_params = {
                        'action': 'wbgetentities',
                        'ids': entity_id,
                        'format': 'json',
                        'languages': 'el|en'
                    }
                    
                    entity_response = requests.get(entity_url, params=entity_params, timeout=10)
                    
                    if entity_response.status_code == 200:
                        entity_data = entity_response.json()
                        facts = self._parse_wikidata_entity(entity_data, entity_id)
                        
                        self.cache[cache_key] = facts
                        self._save_cache()
                        return facts
            
            time.sleep(0.5)  # Rate limiting
                        
        except Exception as e:
            logger.warning(f"Wikidata query failed for {concept}: {e}")
        
        return []
    
    def _get_bloom_mapping(self, objective_text: str) -> Dict[str, Any]:
        """Map objective to Bloom's taxonomy"""
        text_lower = objective_text.lower()
        
        # Greek verb patterns for Bloom's levels
        bloom_patterns = {
            'remember': ['θυμάται', 'αναγνωρίζει', 'αναφέρει', 'ονομάζει', 'απομνημονεύει'],
            'understand': ['κατανοεί', 'εξηγεί', 'ερμηνεύει', 'συμπεραίνει', 'παραφράζει'],
            'apply': ['εφαρμόζει', 'χρησιμοποιεί', 'λύνει', 'υπολογίζει', 'επιδεικνύει'],
            'analyze': ['αναλύει', 'συγκρίνει', 'διαχωρίζει', 'εξετάζει', 'ταξινομεί'],
            'evaluate': ['αξιολογεί', 'κρίνει', 'επιλέγει', 'τεκμηριώνει', 'υποστηρίζει'],
            'create': ['δημιουργεί', 'σχεδιάζει', 'συνθέτει', 'παράγει', 'επινοεί']
        }
        
        detected_levels = []
        confidence_scores = {}
        
        for level, patterns in bloom_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text_lower)
            if matches > 0:
                detected_levels.append(level)
                confidence_scores[level] = matches / len(patterns)
        
        primary_level = max(confidence_scores, key=confidence_scores.get) if confidence_scores else 'understand'
        
        return {
            'primary_level': primary_level,
            'detected_levels': detected_levels,
            'confidence_scores': confidence_scores,
            'bloom_hierarchy_position': self._get_bloom_position(primary_level)
        }
    
    def _get_cefr_alignment(self, objective_text: str, subject: str) -> Dict[str, Any]:
        """Align objective with CEFR levels"""
        if subject != "language":
            return {'level': 'N/A', 'reason': 'Not applicable to non-language subjects'}
        
        text_lower = objective_text.lower()
        
        # CEFR indicators for Greek language learning
        cefr_indicators = {
            'A1': ['αναγνωρίζει', 'γράμματα', 'απλές λέξεις', 'βασικές φράσεις'],
            'A2': ['κατανοεί', 'απλά κείμενα', 'καθημερινά θέματα', 'βασική επικοινωνία'],
            'B1': ['εκφράζεται', 'γνώριμα θέματα', 'απλά κείμενα', 'προσωπικές εμπειρίες'],
            'B2': ['αναλύει', 'πολύπλοκα κείμενα', 'αφηρημένα θέματα', 'επιχειρήματα'],
            'C1': ['ερμηνεύει', 'λογοτεχνικά κείμενα', 'κριτική σκέψη', 'σύνθετες ιδέες'],
            'C2': ['δημιουργεί', 'πρωτότυπα κείμενα', 'λογοτεχνική ανάλυση', 'κριτική αξιολόγηση']
        }
        
        level_scores = {}
        for level, indicators in cefr_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                level_scores[level] = score / len(indicators)
        
        if level_scores:
            primary_level = max(level_scores, key=level_scores.get)
            return {
                'level': primary_level,
                'confidence': level_scores[primary_level],
                'all_scores': level_scores,
                'description': self._get_cefr_description(primary_level)
            }
        
        return {'level': 'B1', 'confidence': 0.3, 'reason': 'Default middle level'}
    
    def _get_international_standards(self, objective_text: str, subject: str) -> List[Dict[str, str]]:
        """Map to international education standards"""
        standards = []
        
        # Common Core State Standards alignment (adapted for Greek context)
        if subject == "language":
            standards.extend([
                {
                    'framework': 'European Language Portfolio',
                    'standard': 'Self-assessment and reflection on language learning',
                    'relevance': 'high' if 'αυτοαξιολόγηση' in objective_text.lower() else 'medium'
                },
                {
                    'framework': 'CEFR Companion Volume',
                    'standard': 'Mediation and plurilingual competence',
                    'relevance': 'high' if any(term in objective_text.lower() for term in ['επικοινωνία', 'διάλογος']) else 'medium'
                }
            ])
        
        # UNESCO Education 2030 Framework
        standards.append({
            'framework': 'UNESCO Education 2030',
            'standard': 'Quality education and lifelong learning',
            'relevance': 'high'
        })
        
        return standards
    
    def _get_pedagogical_approaches(self, objective_text: str) -> List[Dict[str, str]]:
        """Suggest pedagogical approaches"""
        text_lower = objective_text.lower()
        approaches = []
        
        approach_mapping = {
            'Communicative Language Teaching': ['επικοινωνία', 'διάλογος', 'αλληλεπίδραση'],
            'Task-Based Learning': ['δραστηριότητες', 'εργασίες', 'πρακτική εφαρμογή'],
            'Content and Language Integrated Learning': ['περιεχόμενο', 'διαθεματική'],
            'Collaborative Learning': ['ομαδική', 'συνεργασία', 'συλλογική'],
            'Differentiated Instruction': ['διαφοροποίηση', 'εξατομίκευση'],
            'Assessment for Learning': ['αξιολόγηση', 'ανατροφοδότηση']
        }
        
        for approach, keywords in approach_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                approaches.append({
                    'approach': approach,
                    'description': f'Suitable for objectives involving {", ".join(keywords)}',
                    'evidence_strength': 'strong'
                })
        
        return approaches
    
    def _get_assessment_methods(self, objective_text: str) -> List[Dict[str, str]]:
        """Suggest appropriate assessment methods"""
        text_lower = objective_text.lower()
        methods = []
        
        assessment_mapping = {
            'Portfolio Assessment': ['συλλογή εργασιών', 'πρόοδος', 'portfolio'],
            'Performance Assessment': ['επίδοση', 'πρακτική', 'εφαρμογή'],
            'Peer Assessment': ['ετεροαξιολόγηση', 'συμμαθητές'],
            'Self Assessment': ['αυτοαξιολόγηση', 'στοχασμός'],
            'Formative Assessment': ['διαμορφωτική', 'συνεχής'],
            'Rubric-Based Assessment': ['κριτήρια', 'ποιότητα']
        }
        
        for method, indicators in assessment_mapping.items():
            if any(indicator in text_lower for indicator in indicators):
                methods.append({
                    'method': method,
                    'suitability': 'high',
                    'implementation': f'Use for objectives with {indicators[0]}'
                })
        
        # Default methods for common objective types
        if 'αναγνωρίζει' in text_lower:
            methods.append({
                'method': 'Recognition Test',
                'suitability': 'high',
                'implementation': 'Multiple choice or matching exercises'
            })
        
        if 'δημιουργεί' in text_lower:
            methods.append({
                'method': 'Creative Project',
                'suitability': 'high', 
                'implementation': 'Open-ended creative tasks'
            })
        
        return methods
    
    def _parse_wikidata_entity(self, entity_data: Dict, entity_id: str) -> List[Dict[str, str]]:
        """Parse Wikidata entity response"""
        facts = []
        
        try:
            entity = entity_data['entities'][entity_id]
            
            # Get description
            descriptions = entity.get('descriptions', {})
            if 'el' in descriptions:
                facts.append({
                    'property': 'description',
                    'value': descriptions['el']['value'],
                    'source': 'wikidata'
                })
            
            # Get some key claims
            claims = entity.get('claims', {})
            for prop_id, claim_list in claims.items():
                if len(facts) >= 5:  # Limit facts
                    break
                    
                for claim in claim_list[:1]:  # One claim per property
                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                        value = claim['mainsnak']['datavalue'].get('value')
                        if isinstance(value, str) and len(value) < 100:
                            facts.append({
                                'property': prop_id,
                                'value': value,
                                'source': 'wikidata'
                            })
        
        except Exception as e:
            logger.warning(f"Error parsing Wikidata entity: {e}")
        
        return facts
    
    def _get_bloom_position(self, level: str) -> int:
        """Get hierarchical position in Bloom's taxonomy"""
        positions = {
            'remember': 1, 'understand': 2, 'apply': 3,
            'analyze': 4, 'evaluate': 5, 'create': 6
        }
        return positions.get(level, 2)
    
    def _get_cefr_description(self, level: str) -> str:
        """Get CEFR level description"""
        descriptions = {
            'A1': 'Basic user - Breakthrough',
            'A2': 'Basic user - Waystage', 
            'B1': 'Independent user - Threshold',
            'B2': 'Independent user - Vantage',
            'C1': 'Proficient user - Effective Operational Proficiency',
            'C2': 'Proficient user - Mastery'
        }
        return descriptions.get(level, 'Unknown level')
    
    def _load_cache(self) -> Dict:
        """Load knowledge cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load knowledge cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save knowledge cache"""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save knowledge cache: {e}")

# Global knowledge enhancer instance
knowledge_enhancer = KnowledgeEnhancer()