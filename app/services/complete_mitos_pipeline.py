# app/services/complete_mitos_pipeline.py
"""
Complete MITOS Pipeline Implementation - Stages 2-7
Automates the full methodology from the research paper
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re

from app.services.mitos_annotator import (
    MITOSTextAnnotator, 
    AnnotatedLegalText, 
    TOGAFAnnotation,
    PerspectiveAnnotation,
    AnnotationPerspective
)
from app.services.llm_service import MultiLLMService, LLMProvider

logger = logging.getLogger(__name__)

@dataclass
class ControlledVocabulary:
    """Stage 3: Controlled vocabulary with synonyms and hierarchies"""
    preferred_terms: Dict[str, str]  # variant -> preferred
    categories: Dict[str, List[str]]  # category -> terms
    definitions: Dict[str, str]       # term -> definition
    hierarchies: Dict[str, List[str]] # parent -> children
    synonym_rings: Dict[str, List[str]] # concept -> synonyms

@dataclass
class ProcessHierarchy:
    """Stage 4: APQC-style hierarchical process structure"""
    level_1_category: str
    level_2_process_group: str
    level_3_process: str
    level_4_activity: str
    level_5_tasks: List[str]
    standard_steps: List[str]  # 4 standard steps from paper

@dataclass
class ProcessModel:
    """Stage 5: Complete process model with BPMN-like structure"""
    process_id: str
    process_name: str
    actors: List[Dict[str, Any]]
    activities: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    gateways: List[Dict[str, Any]]
    data_objects: List[Dict[str, Any]]
    sequence_flows: List[Dict[str, Any]]

@dataclass
class OntologyModel:
    """Stage 6: RDF/OWL ontology representation"""
    namespaces: Dict[str, str]
    classes: List[Dict[str, Any]]
    properties: List[str]
    individuals: List[Dict[str, Any]]
    rdf_triples: List[str]
    json_graph: Dict[str, Any]
@dataclass
class CompleteMITOSResult:
    """Final result containing all pipeline stages"""
    original_annotation: AnnotatedLegalText
    controlled_vocabulary: ControlledVocabulary
    process_hierarchy: ProcessHierarchy
    process_model: ProcessModel
    ontology_model: OntologyModel
    human_readable_output: str
    machine_readable_output: str
    sparql_queries: List[str]
    confidence_scores: Dict[str, float]

class CompleteMITOSPipeline:
    """Complete MITOS pipeline implementing stages 2-7"""
    
    def __init__(self):
        self.annotator = MITOSTextAnnotator()
        self.llm_service = MultiLLMService()
        
    def setup_llm(self, provider: LLMProvider, api_key: str, model_name: Optional[str] = None):
        """Setup LLM provider"""
        self.annotator.setup_llm(provider, api_key, model_name)
        self.llm_service.add_service(provider, api_key, model_name)
    
    def run_complete_pipeline(self, 
                            text: str, 
                            text_title: str,
                            text_type: str,
                            provider: LLMProvider) -> CompleteMITOSResult:
        """Run the complete MITOS pipeline (stages 2-7)"""
        logger.info(f"Starting complete MITOS pipeline for: {text_title}")
        
        # Stage 2: Annotate text
        logger.info("Stage 2: Annotating text...")
        annotation = self.annotator.annotate_legal_text(text, text_title, text_type, provider)
        
        # Stage 3: Standardize metadata
        logger.info("Stage 3: Standardizing metadata...")
        vocabulary = self._standardize_metadata(annotation, provider)
        
        # Stage 4: Standardize metaprocesses  
        logger.info("Stage 4: Standardizing metaprocesses...")
        hierarchy = self._standardize_metaprocesses(annotation, vocabulary, provider)
        
        # Stage 5: Create process model
        logger.info("Stage 5: Creating process model...")
        model = self._create_process_model(annotation, vocabulary, hierarchy, provider)
        
        # Stage 6: Ontologize model
        logger.info("Stage 6: Ontologizing model...")
        ontology = self._ontologize_model(annotation, vocabulary, hierarchy, model, provider)
        
        # Stage 7: Publish information
        logger.info("Stage 7: Publishing information...")
        human_readable, machine_readable, queries = self._publish_information(
            annotation, vocabulary, hierarchy, model, ontology
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_pipeline_confidence(
            annotation, vocabulary, hierarchy, model, ontology
        )
        
        return CompleteMITOSResult(
            original_annotation=annotation,
            controlled_vocabulary=vocabulary,
            process_hierarchy=hierarchy,
            process_model=model,
            ontology_model=ontology,
            human_readable_output=human_readable,
            machine_readable_output=machine_readable,
            sparql_queries=queries,
            confidence_scores=confidence_scores
        )
    
    def _standardize_metadata(self, annotation: AnnotatedLegalText, provider: LLMProvider) -> ControlledVocabulary:
        """Stage 3: Create controlled vocabulary from annotation"""
        
        # Extract all terms from annotation
        all_terms = []
        togaf = annotation.togaf_annotation
        
        all_terms.extend(togaf.processes)
        all_terms.extend(togaf.functions)
        all_terms.extend(togaf.actors)
        all_terms.extend(togaf.roles)
        all_terms.extend(togaf.organization_units)
        all_terms.extend(togaf.events)
        all_terms.extend(togaf.products)
        all_terms.extend(togaf.controls)
        all_terms.extend(togaf.business_services)
        
        # Add perspective terms
        all_terms.extend(annotation.citizen_perspective.key_requirements)
        all_terms.extend(annotation.official_perspective.key_requirements)
        
        prompt = f"""Create a controlled vocabulary from these legal process terms:

TERMS: {', '.join(set(all_terms))}

Generate:
1. CATEGORIES: Organize terms into 6 categories (Person, Role, Organization, Geographic Locations, Physical Resources, Activities and Events)
2. PREFERRED_TERMS: For similar terms, choose one as preferred
3. DEFINITIONS: Define key terms clearly
4. HIERARCHIES: Parent-child relationships
5. SYNONYM_RINGS: Group related/similar terms

Return as JSON:
{{
    "categories": {{
        "Person": ["term1", "term2"],
        "Role": ["term3", "term4"],
        "Organization": ["term5"],
        "Geographic Locations": ["term6"],
        "Physical Resources": ["term7", "term8"],  
        "Activities and Events": ["term9", "term10"]
    }},
    "preferred_terms": {{
        "variant_term": "preferred_term"
    }},
    "definitions": {{
        "term": "clear definition"
    }},
    "hierarchies": {{
        "parent_term": ["child1", "child2"]
    }},
    "synonym_rings": {{
        "concept1": ["synonym1", "synonym2", "synonym3"]
    }}
}}

Focus on Greek public administration terminology."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            data = self._parse_json_response(response)
            
            return ControlledVocabulary(
                preferred_terms=data.get('preferred_terms', {}),
                categories=data.get('categories', {}),
                definitions=data.get('definitions', {}),
                hierarchies=data.get('hierarchies', {}),
                synonym_rings=data.get('synonym_rings', {})
            )
        except Exception as e:
            logger.error(f"Metadata standardization failed: {e}")
            return ControlledVocabulary({}, {}, {}, {}, {})
    
    def _standardize_metaprocesses(self, 
                                 annotation: AnnotatedLegalText,
                                 vocabulary: ControlledVocabulary, 
                                 provider: LLMProvider) -> ProcessHierarchy:
        """Stage 4: Create APQC-style process hierarchy"""
        
        prompt = f"""Create a hierarchical process structure using APQC methodology for this process:

PROCESS: {annotation.text_title}
PROCESS TYPE: {annotation.text_type}

TOGAF PROCESSES: {', '.join(annotation.togaf_annotation.processes)}

Create 6-level hierarchy following APQC Process Classification Framework:
- Level 1: Category (broad domain like "Citizen Services")
- Level 2: Process Group (like "Student Registration Services") 
- Level 3: Process (like "Student Enrollment")
- Level 4: Activity (specific version like "High School First Grade Enrollment")
- Level 5: Tasks (main activities from the process)

Also include the 4 standard steps found in most administrative processes:
1. "Application Submission"
2. "Application Check" 
3. "Decision Making"
4. "Decision Notification"

Return as JSON:
{{
    "level_1_category": "broad category name",
    "level_2_process_group": "process group name", 
    "level_3_process": "specific process name",
    "level_4_activity": "detailed activity name",
    "level_5_tasks": ["task1", "task2", "task3"],
    "standard_steps": ["Application Submission", "Application Check", "Decision Making", "Decision Notification"]
}}

Base this on Greek public administration structure."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            data = self._parse_json_response(response)
            
            return ProcessHierarchy(
                level_1_category=data.get('level_1_category', ''),
                level_2_process_group=data.get('level_2_process_group', ''),
                level_3_process=data.get('level_3_process', ''),
                level_4_activity=data.get('level_4_activity', ''),
                level_5_tasks=data.get('level_5_tasks', []),
                standard_steps=data.get('standard_steps', [
                    "Application Submission", "Application Check", 
                    "Decision Making", "Decision Notification"
                ])
            )
        except Exception as e:
            logger.error(f"Metaprocess standardization failed: {e}")
            return ProcessHierarchy('', '', '', '', [], [])
    
    def _create_process_model(self,
                            annotation: AnnotatedLegalText,
                            vocabulary: ControlledVocabulary,
                            hierarchy: ProcessHierarchy,
                            provider: LLMProvider) -> ProcessModel:
        """Stage 5: Create BPMN-like process model"""
        
        prompt = f"""Create a detailed process model for: {annotation.text_title}

ACTORS: {', '.join(annotation.togaf_annotation.actors)}
ROLES: {', '.join(annotation.togaf_annotation.roles)}
PROCESSES: {', '.join(annotation.togaf_annotation.processes)}
EVENTS: {', '.join(annotation.togaf_annotation.events)}

Create a BPMN-style process model with:
- ACTORS: People/systems involved with their roles
- ACTIVITIES: Tasks to be performed 
- EVENTS: Start/intermediate/end events
- GATEWAYS: Decision points or parallel flows
- DATA_OBJECTS: Documents/information produced or required
- SEQUENCE_FLOWS: Order of execution

Return as JSON:
{{
    "process_id": "unique_process_id",
    "process_name": "{annotation.text_title}",
    "actors": [
        {{"id": "actor1", "name": "Actor Name", "role": "Role", "organization": "Organization"}}
    ],
    "activities": [
        {{"id": "activity1", "name": "Activity Name", "actor": "actor1", "description": "What is done"}}
    ],
    "events": [
        {{"id": "event1", "type": "start|intermediate|end", "name": "Event Name", "trigger": "what triggers it"}}
    ],
    "gateways": [
        {{"id": "gateway1", "type": "exclusive|parallel|inclusive", "name": "Decision Point", "condition": "decision criteria"}}
    ],
    "data_objects": [
        {{"id": "data1", "name": "Document Name", "type": "input|output", "description": "Purpose"}}
    ],
    "sequence_flows": [
        {{"from": "element_id", "to": "next_element_id", "condition": "optional condition"}}
    ]
}}

Make it comprehensive and actionable for Greek public administration."""

        try:
            response = self.llm_service.generate_with_provider(provider, prompt)
            data = self._parse_json_response(response)
            
            return ProcessModel(
                process_id=data.get('process_id', f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                process_name=data.get('process_name', annotation.text_title),
                actors=data.get('actors', []),
                activities=data.get('activities', []),
                events=data.get('events', []),
                gateways=data.get('gateways', []),
                data_objects=data.get('data_objects', []),
                sequence_flows=data.get('sequence_flows', [])
            )
        except Exception as e:
            logger.error(f"Process model creation failed: {e}")
            return ProcessModel('', annotation.text_title, [], [], [], [], [], [])
    
    def _ontologize_model(self,
                        annotation: AnnotatedLegalText,
                        vocabulary: ControlledVocabulary,
                        hierarchy: ProcessHierarchy, 
                        model: ProcessModel,
                        provider: LLMProvider) -> OntologyModel:
        """Stage 6: Convert to RDF/OWL ontology and generate D3.js graph data."""
        
        # Generate RDF
        rdf_triples = []
        namespaces = {
            'ex': 'http://example.org/mitos#',
            'cpsv': 'http://purl.org/vocab/cpsv#',
            'cpov': 'http://purl.org/vocab/cpov#', 
            'person': 'http://www.w3.org/ns/person#',
            'org': 'http://www.w3.org/ns/org#',
            'dct': 'http://purl.org/dc/terms/',
            'psl': 'http://www.ontologyportal.org/translations/SUMO/ProcessSpecificationLanguage.owl#',
            'sem': 'http://semanticweb.cs.vu.nl/2009/11/sem/'
        }
        
        classes = [
            {'id': 'Process', 'label': 'Administrative Process', 'description': 'A public administration process'},
            {'id': 'Actor', 'label': 'Process Actor', 'description': 'Person or entity in process'},
            {'id': 'Role', 'label': 'Administrative Role', 'description': 'Role in administration'},
            {'id': 'OrganizationUnit', 'label': 'Organization Unit', 'description': 'Administrative unit'},
            {'id': 'Event', 'label': 'Process Event', 'description': 'Event in process'},
            {'id': 'Document', 'label': 'Administrative Document', 'description': 'Required document'},
            {'id': 'Activity', 'label': 'Process Activity', 'description': 'Activity in process'}
        ]
        
        properties = [
            'hasActor', 'hasRole', 'belongsTo', 'generates', 'requires', 'produces', 'hasSubProcess', 'hasStep'
        ]
        
        # Generate individuals from the process model with consistent ID normalization
        individuals = []
        uri_to_simplified = {}  # Keep track of URI mappings
        
        # Process individual
        process_id_normalized = re.sub(r'[^\w]', '_', annotation.text_title.lower())
        process_uri = f"ex:{process_id_normalized}"
        individuals.append({
            'id': process_uri, 
            'type': 'Process', 
            'label': annotation.text_title,
            'description': f"Administrative process: {annotation.text_title}"
        })
        uri_to_simplified[process_uri] = process_id_normalized
        
        # Actor individuals
        for actor_name in annotation.togaf_annotation.actors:
            actor_id_normalized = re.sub(r'[^\w]', '_', actor_name.lower())
            actor_uri = f"ex:{actor_id_normalized}"
            individuals.append({
                'id': actor_uri,
                'type': 'Actor',
                'label': actor_name,
                'description': f"An actor in the process: {actor_name}"
            })
            uri_to_simplified[actor_uri] = actor_id_normalized
        
        # Activity individuals
        for i, activity_name in enumerate(annotation.citizen_perspective.process_steps):
            activity_id_normalized = f"citizen_step_{i + 1}"
            activity_uri = f"ex:{activity_id_normalized}"
            individuals.append({
                'id': activity_uri, 
                'type': 'Activity', 
                'label': activity_name,
                'description': f"Step for citizens: {activity_name}"
            })
            uri_to_simplified[activity_uri] = activity_id_normalized

        # Data object individuals
        for doc_name in annotation.citizen_perspective.required_documents:
            doc_id_normalized = re.sub(r'[^\w]', '_', doc_name.lower())
            doc_uri = f"ex:{doc_id_normalized}"
            individuals.append({
                'id': doc_uri,
                'type': 'Document',
                'label': doc_name,
                'description': f"Required document: {doc_name}"
            })
            uri_to_simplified[doc_uri] = doc_id_normalized

        # Generate meaningful relationships and RDF triples
        relationship_triples = [
            (process_uri, 'a', 'ex:Process'),
            (process_uri, 'dct:title', f'"{annotation.text_title}"'),
            (process_uri, 'dct:description', f'"Administrative process for {annotation.text_title}"'),
            (process_uri, 'dct:created', f'"{datetime.now().isoformat()}"')
        ]

        # Create relationships between entities
        for actor in individuals:
            if actor['type'] == 'Actor':
                relationship_triples.append((actor['id'], 'a', 'ex:Actor'))
                relationship_triples.append((process_uri, 'ex:hasActor', actor['id']))

        for activity in individuals:
            if activity['type'] == 'Activity':
                relationship_triples.append((activity['id'], 'a', 'ex:Activity'))
                relationship_triples.append((process_uri, 'ex:hasSubProcess', activity['id']))

        for doc in individuals:
            if doc['type'] == 'Document':
                relationship_triples.append((doc['id'], 'a', 'ex:Document'))
                relationship_triples.append((process_uri, 'ex:requires', doc['id']))
                
        # Add sequential relationships for activities
        activities = [ind for ind in individuals if ind['type'] == 'Activity']
        for i in range(len(activities) - 1):
            current_activity = activities[i]['id']
            next_activity = activities[i + 1]['id']
            relationship_triples.append((current_activity, 'ex:precedes', next_activity))

        # Convert relationship tuples to RDF strings
        for subj, pred, obj in relationship_triples:
            if obj.startswith('"'):
                rdf_triples.append(f"{subj} {pred} {obj} .")
            else:
                rdf_triples.append(f"{subj} {pred} {obj} .")
                    
        # D3.js-compatible graph data generation with simplified IDs
        nodes = []
        links = []
        
        # Create nodes with simplified IDs
        for ind in individuals:
            simplified_id = uri_to_simplified.get(ind['id'], ind['id'].split(':')[-1])
            nodes.append({
                "id": simplified_id,
                "label": ind.get('label', simplified_id),
                "type": ind.get('type', ''),
                "description": ind.get('description', ''),
                "category": ind.get('type', '').lower(),
                "color": self._get_node_color(ind.get('type', ''))
            })
        
        # Create links from relationship triples (not all RDF triples)
        for subj_uri, pred, obj_uri in relationship_triples:
            # Skip literal values and type declarations for visualization
            if obj_uri.startswith('"') or pred == 'a':
                continue
                
            source_id = uri_to_simplified.get(subj_uri, subj_uri.split(':')[-1])
            target_id = uri_to_simplified.get(obj_uri, obj_uri.split(':')[-1])
            
            # Only add links between existing nodes
            if (source_id != target_id and 
                any(n['id'] == source_id for n in nodes) and 
                any(n['id'] == target_id for n in nodes)):
                
                # Clean up predicate label
                pred_label = pred.replace('ex:', '').replace('_', ' ').title()
                
                links.append({
                    "source": source_id,
                    "target": target_id,
                    "label": pred_label,
                    "full_relation": f"{source_id} {pred_label} {target_id}"
                })

        json_graph = {"nodes": nodes, "links": links}
        
        logger.info(f"Generated ontology with {len(nodes)} nodes and {len(links)} links")

        return OntologyModel(
            namespaces=namespaces,
            classes=classes,
            properties=properties,
            individuals=individuals,
            rdf_triples=rdf_triples,
            json_graph=json_graph
        )

    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type"""
        color_map = {
            'Process': '#10b981',      # Green
            'Actor': '#3b82f6',        # Blue
            'Activity': '#f59e0b',     # Orange
            'Document': '#ef4444',     # Red
            'Role': '#8b5cf6',         # Purple
            'Event': '#06b6d4',        # Cyan
            'OrganizationUnit': '#84cc16'  # Lime
        }
        return color_map.get(node_type, '#6b7280')  # Default gray
    def _publish_information(self,
                           annotation: AnnotatedLegalText,
                           vocabulary: ControlledVocabulary,
                           hierarchy: ProcessHierarchy,
                           model: ProcessModel,
                           ontology: OntologyModel) -> Tuple[str, str, List[str]]:
        """Stage 7: Generate human and machine readable outputs"""
        
        # Human readable output (for MITOS display)
        human_readable = f"""
# {annotation.text_title}

## Process Overview
**Type:** {annotation.text_type}
**Confidence Score:** {annotation.confidence_score:.2f}

## Process Hierarchy
- **Category:** {hierarchy.level_1_category}
- **Process Group:** {hierarchy.level_2_process_group}  
- **Process:** {hierarchy.level_3_process}
- **Activity:** {hierarchy.level_4_activity}

## Key Information

### Actors and Roles
{self._format_list(annotation.togaf_annotation.actors)}

### Required Steps
{self._format_list(hierarchy.standard_steps)}

### Required Documents  
{self._format_list(annotation.citizen_perspective.required_documents)}

### Process Steps for Citizens
{self._format_numbered_list(annotation.citizen_perspective.process_steps)}

### Process Steps for Officials
{self._format_numbered_list(annotation.official_perspective.process_steps)}

### Key Requirements
**For Citizens:**
{self._format_list(annotation.citizen_perspective.key_requirements)}

**For Officials:**
{self._format_list(annotation.official_perspective.key_requirements)}

## Controlled Vocabulary
**Categories:** {', '.join(vocabulary.categories.keys())}
**Total Terms:** {sum(len(terms) for terms in vocabulary.categories.values())}
"""

        # Machine readable output (RDF Turtle)
        machine_readable = self._generate_turtle_output(ontology)
        
        # SPARQL queries for common questions
        sparql_queries = [
            f"""
# Query 1: Get all actors and their roles in the process
SELECT ?actor ?role WHERE {{
    ex:process_{model.process_id} ex:hasActor ?actor .
    ?actor ex:hasRole ?role .
}}
""",
            f"""  
# Query 2: Get all required documents for citizens
SELECT ?document ?description WHERE {{
    ex:process_{model.process_id} ex:requires ?document .
    ?document dct:description ?description .
    ?document ex:requiredBy ex:Citizen .
}}
""",
            f"""
# Query 3: Get process steps in order
SELECT ?step ?order ?description WHERE {{
    ex:process_{model.process_id} psl:hasSubProcess ?step .
    ?step ex:order ?order .
    ?step dct:description ?description .
}} ORDER BY ?order
"""
        ]
        
        return human_readable, machine_readable, sparql_queries
    
    def _generate_turtle_output(self, ontology: OntologyModel) -> str:
        """Generate RDF Turtle format output"""
        turtle = []
        
        # Add namespace prefixes
        for prefix, uri in ontology.namespaces.items():
            turtle.append(f"@prefix {prefix}: <{uri}> .")
        turtle.append("")
        
        # Add RDF triples
        turtle.extend(ontology.rdf_triples)
        
        return "\n".join(turtle)
    
    def _calculate_pipeline_confidence(self,
                                     annotation: AnnotatedLegalText,
                                     vocabulary: ControlledVocabulary,
                                     hierarchy: ProcessHierarchy,
                                     model: ProcessModel,
                                     ontology: OntologyModel) -> Dict[str, float]:
        """Calculate confidence scores for each pipeline stage"""
        
        scores = {}
        
        # Stage 2: Annotation confidence (already calculated)
        scores['annotation'] = annotation.confidence_score
        
        # Stage 3: Vocabulary completeness
        vocab_score = 0.0
        if vocabulary.categories:
            filled_categories = sum(1 for terms in vocabulary.categories.values() if terms)
            vocab_score = filled_categories / len(vocabulary.categories)
        scores['vocabulary'] = vocab_score
        
        # Stage 4: Hierarchy completeness  
        hierarchy_fields = [
            hierarchy.level_1_category, hierarchy.level_2_process_group,
            hierarchy.level_3_process, hierarchy.level_4_activity
        ]
        filled_hierarchy = sum(1 for field in hierarchy_fields if field)
        scores['hierarchy'] = filled_hierarchy / len(hierarchy_fields) if hierarchy_fields else 0
        
        # Stage 5: Model completeness
        model_components = [model.actors, model.activities, model.events, model.data_objects]
        filled_model = sum(1 for component in model_components if component)
        scores['model'] = filled_model / len(model_components) if model_components else 0
        
        # Stage 6: Ontology completeness
        ontology_components = [ontology.classes, ontology.properties, ontology.individuals, ontology.rdf_triples]
        filled_ontology = sum(1 for component in ontology_components if component)
        scores['ontology'] = filled_ontology / len(ontology_components) if ontology_components else 0
        
        # Overall pipeline confidence
        scores['overall'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _format_list(self, items: List[str]) -> str:
        """Format list for human readable output"""
        if not items:
            return "- None specified"
        return '\n'.join(f"- {item}" for item in items)
    
    def _format_numbered_list(self, items: List[str]) -> str:
        """Format numbered list for human readable output"""
        if not items:
            return "1. None specified"
        return '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with error handling"""
        if not response or not response.strip():
            return {}
        
        try:
            # Clean the response
            cleaned = response.strip()
            
            # Remove markdown formatting
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            elif cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if len(lines) > 2:
                    cleaned = '\n'.join(lines[1:-1])
            
            # Try to find JSON within the response
            json_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return {}
    
    def save_complete_result(self, result: CompleteMITOSResult, output_path: Path):
        """Save complete pipeline result"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, AnnotationPerspective):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return convert_to_serializable(asdict(obj))
            else:
                return obj
        
        result_dict = {
            "pipeline_timestamp": str(datetime.now()),
            "original_annotation": convert_to_serializable(asdict(result.original_annotation)),
            "controlled_vocabulary": convert_to_serializable(asdict(result.controlled_vocabulary)),
            "process_hierarchy": convert_to_serializable(asdict(result.process_hierarchy)),
            "process_model": convert_to_serializable(asdict(result.process_model)),
            "ontology_model": convert_to_serializable(asdict(result.ontology_model)),
            "confidence_scores": result.confidence_scores,
            "human_readable_output": result.human_readable_output,
            "machine_readable_output": result.machine_readable_output,
            "sparql_queries": result.sparql_queries
        }
        
        # Save JSON result
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Save RDF output separately  
        rdf_path = output_path.with_suffix('.ttl')
        with open(rdf_path, 'w', encoding='utf-8') as f:
            f.write(result.machine_readable_output)
        
        # Save human readable output
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(result.human_readable_output)
        
        logger.info(f"Complete MITOS result saved to {output_path} (.json, .ttl, .md)")