"""
Focused Ontology RAG Service - Only TTL files and CEDS
Removes the curriculum PDF database and focuses on ontology-to-ontology queries
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import json
import time
from datetime import datetime
import requests

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from app.services.llm_service import MultiLLMService, LLMProvider

logger = logging.getLogger(__name__)

class FocusedOntologyRAGService:
    """RAG service for TTL ontologies and CEDS only"""
    
    def __init__(self):
        self.ontology_store = {}
        self.loaded_ontologies = []
        
    def load_existing_ontologies(self, ontology_dir: Path):
        """Load previously extracted TTL ontologies for RAG"""
        for ttl_file in ontology_dir.glob("*.ttl"):
            try:
                g = Graph()
                g.parse(ttl_file, format="turtle")
                
                ontology_key = ttl_file.stem
                self.ontology_store[ontology_key] = {
                    'graph': g,
                    'file_path': ttl_file,
                    'loaded_at': datetime.now(),
                    'triples_count': len(g),
                    'source': 'local_ttl'
                }
                
                self.loaded_ontologies.append(ontology_key)
                logger.info(f"Loaded TTL ontology: {ontology_key} ({len(g)} triples)")
                
            except Exception as e:
                logger.error(f"Failed to load TTL ontology {ttl_file}: {e}")

    def load_ceds_ontology(self):
        """Load CEDS ontology from the official RDF file"""
        
        ceds_url = "https://raw.githubusercontent.com/CEDStandards/CEDS-Ontology/main/src/CEDS-Ontology.rdf"
        cache_file = Path("data/ceds_cache/ceds.rdf")
        
        try:
            # Create cache directory
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download if not cached or cache is old (24 hours)
            if not cache_file.exists() or (time.time() - cache_file.stat().st_mtime) > 86400:
                logger.info("Downloading CEDS ontology...")
                
                response = requests.get(ceds_url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Educational Research Tool)'
                })
                response.raise_for_status()
                
                with open(cache_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"CEDS ontology downloaded ({len(response.content)} bytes)")
            else:
                logger.info("Using cached CEDS ontology")
            
            # Load into graph
            ceds_graph = Graph()
            ceds_graph.parse(cache_file, format="xml")
            
            # Add to ontology store
            self.ontology_store['ceds'] = {
                'graph': ceds_graph,
                'file_path': cache_file,
                'loaded_at': datetime.now(),
                'triples_count': len(ceds_graph),
                'source': 'CEDS Standards'
            }
            
            self.loaded_ontologies.append('ceds')
            logger.info(f"Loaded CEDS ontology: {len(ceds_graph)} triples")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download CEDS ontology: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load CEDS ontology: {e}")
            return False

    def query_ceds_for_curriculum_alignment(self, curriculum_subject: str, topics: List[str]) -> List[Dict]:
        """Query CEDS for curriculum alignment with specific topics"""
        
        if 'ceds' not in self.ontology_store:
            logger.warning("CEDS ontology not loaded")
            return []
        
        ceds_graph = self.ontology_store['ceds']['graph']
        all_alignments = []
        
        # Query for each topic
        for topic in topics[:3]:  # Limit to first 3 topics
            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?entity ?label ?description WHERE {{
                ?entity rdfs:label ?label .
                OPTIONAL {{ ?entity rdfs:comment ?description }}
                
                FILTER(
                    CONTAINS(LCASE(?label), "{topic.lower()}") ||
                    CONTAINS(LCASE(STR(?description)), "{topic.lower()}")
                )
            }}
            LIMIT 3
            """
            
            try:
                results = ceds_graph.query(query)
                for row in results:
                    all_alignments.append({
                        'source': 'CEDS',
                        'topic': topic,
                        'entity': str(row[0]) if row[0] else '',
                        'label': str(row[1]) if row[1] else '',
                        'description': str(row[2]) if row[2] else '',
                        'alignment_type': 'semantic_match'
                    })
            except Exception as e:
                logger.error(f"CEDS query failed for topic {topic}: {e}")
        
        return all_alignments

    def query_similar_learning_objectives(self, objective_text: str, top_k: int = 3) -> List[Dict]:
        """Find similar learning objectives across loaded TTL ontologies"""
        similar_objectives = []
        
        for ont_key, ont_data in self.ontology_store.items():
            if ont_data['source'] == 'CEDS Standards':
                continue  # Skip CEDS for learning objectives query
                
            g = ont_data['graph']
            
            # SPARQL query for learning objectives with descriptions
            query = """
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?objective ?description ?gradeLevel ?cognitiveLevel WHERE {
                ?objective a currkg:LearningObjective ;
                          currkg:hasDescription ?description .
                OPTIONAL { ?objective currkg:hasGradeLevel ?gradeLevel }
                OPTIONAL { ?objective currkg:hasCognitiveLevel ?cognitiveLevel }
            }
            """
            
            try:
                results = g.query(query)
                for row in results:
                    description = str(row[1])
                    
                    # Calculate text similarity
                    similarity = self._calculate_text_similarity(objective_text, description)
                    
                    if similarity > 0.3:  # Similarity threshold
                        similar_objectives.append({
                            'source_ontology': ont_key,
                            'objective_uri': str(row[0]),
                            'description': description,
                            'grade_level': str(row[2]) if row[2] else None,
                            'cognitive_level': str(row[3]) if row[3] else None,
                            'similarity': similarity
                        })
            except Exception as e:
                logger.error(f"Query failed for ontology {ont_key}: {e}")
        
        # Sort by similarity and return top results
        similar_objectives.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_objectives[:top_k]

    def find_cross_curricular_connections(self, module_topic: str) -> List[Dict]:
        """Find how a topic appears across different TTL curricula ontologies"""
        connections = []
        
        for ont_key, ont_data in self.ontology_store.items():
            if ont_data['source'] == 'CEDS Standards':
                continue  # Skip CEDS for cross-curricular
                
            g = ont_data['graph']
            
            # Query for modules covering similar topics
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?module ?topic ?description WHERE {{
                ?module a currkg:Module ;
                        currkg:coversTopic ?topic .
                ?topic currkg:hasDescription ?description .
                FILTER(CONTAINS(LCASE(?description), "{module_topic.lower()}"))
            }}
            LIMIT 5
            """
            
            try:
                results = g.query(query)
                for row in results:
                    connections.append({
                        'source_curriculum': ont_key,
                        'module_uri': str(row[0]),
                        'topic_uri': str(row[1]),
                        'topic_description': str(row[2]),
                        'connection_type': 'topic_overlap'
                    })
            except Exception as e:
                logger.error(f"Cross-curricular query failed for {ont_key}: {e}")
        
        return connections

    def get_learning_path_patterns(self, persona_type: str) -> List[Dict]:
        """Extract learning path patterns for specific personas from TTL ontologies"""
        patterns = []
        
        for ont_key, ont_data in self.ontology_store.items():
            if ont_data['source'] == 'CEDS Standards':
                continue  # Skip CEDS for learning paths
                
            g = ont_data['graph']
            
            # Query for learning paths associated with personas
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            SELECT ?path ?persona ?step ?stepDescription WHERE {{
                ?persona a currkg:Persona ;
                         currkg:hasType ?personaType ;
                         currkg:determines ?path .
                ?path currkg:hasLearningStep ?step .
                ?step currkg:hasDescription ?stepDescription .
                FILTER(CONTAINS(LCASE(STR(?personaType)), "{persona_type.lower()}"))
            }}
            ORDER BY ?path ?step
            """
            
            try:
                results = g.query(query)
                current_path = None
                path_steps = []
                
                for row in results:
                    path_uri = str(row[0])
                    if current_path != path_uri:
                        if current_path and path_steps:
                            patterns.append({
                                'source_curriculum': ont_key,
                                'path_uri': current_path,
                                'persona_type': persona_type,
                                'steps': path_steps.copy()
                            })
                        current_path = path_uri
                        path_steps = []
                    
                    path_steps.append({
                        'step_uri': str(row[2]),
                        'description': str(row[3])
                    })
                
                # Add final path
                if current_path and path_steps:
                    patterns.append({
                        'source_curriculum': ont_key,
                        'path_uri': current_path,
                        'persona_type': persona_type,
                        'steps': path_steps
                    })
                    
            except Exception as e:
                logger.error(f"Learning path query failed for {ont_key}: {e}")
        
        return patterns

    def get_ontology_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded ontologies"""
        stats = {
            'total_ontologies': len(self.ontology_store),
            'ontology_details': {},
            'total_triples': 0
        }
        
        for ont_key, ont_data in self.ontology_store.items():
            stats['ontology_details'][ont_key] = {
                'source': ont_data['source'],
                'triples_count': ont_data['triples_count'],
                'loaded_at': ont_data['loaded_at'].isoformat(),
                'file_path': str(ont_data['file_path'])
            }
            stats['total_triples'] += ont_data['triples_count']
        
        return stats

    def query_ontology_concepts(self, concept_type: str) -> List[Dict]:
        """Query all ontologies for specific concept types"""
        concepts = []
        
        for ont_key, ont_data in self.ontology_store.items():
            g = ont_data['graph']
            
            # Generic query for concept types
            query = f"""
            PREFIX currkg: <http://curriculum-kg.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?entity ?label ?description WHERE {{
                ?entity a currkg:{concept_type} ;
                        rdfs:label ?label .
                OPTIONAL {{ ?entity rdfs:comment ?description }}
            }}
            LIMIT 10
            """
            
            try:
                results = g.query(query)
                for row in results:
                    concepts.append({
                        'source_ontology': ont_key,
                        'entity_uri': str(row[0]),
                        'label': str(row[1]) if row[1] else '',
                        'description': str(row[2]) if row[2] else '',
                        'concept_type': concept_type
                    })
            except Exception as e:
                logger.debug(f"Concept query failed for {ont_key}: {e}")
        
        return concepts

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)

    def initialize_all_sources(self, ontology_dir: Path):
        """Initialize both TTL ontologies and CEDS"""
        logger.info("Initializing focused ontology RAG service...")
        
        # Load local TTL ontologies
        self.load_existing_ontologies(ontology_dir)
        
        # Load CEDS ontology
        ceds_loaded = self.load_ceds_ontology()
        
        total_ontologies = len(self.loaded_ontologies)
        logger.info(f"Focused RAG initialized with {total_ontologies} ontologies")
        
        if ceds_loaded:
            logger.info("✓ CEDS Standards ontology loaded")
        else:
            logger.warning("✗ CEDS Standards ontology failed to load")
        
        return total_ontologies > 0


