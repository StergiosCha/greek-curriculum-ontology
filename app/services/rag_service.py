import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import hashlib

from app.utils.file_handler import extract_text_from_pdf
from app.core.config import settings

logger = logging.getLogger(__name__)

class CurriculumRAGService:
    """Real RAG implementation for curriculum context retrieval"""
    
    def __init__(self):
        self.embeddings_model = None
        self.curriculum_db = {}
        self.embeddings_cache = {}
        self.cache_file = Path(settings.cache_dir) / "rag_embeddings.pkl"
        self.db_file = Path(settings.cache_dir) / "curriculum_db.pkl"
        
    def initialize(self):
        """Initialize the embedding model and load cache"""
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')  # 90MB total
            logger.info("Loaded multilingual embedding model")
            
            # Load cached data
            self._load_cache()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    def build_curriculum_database(self, force_rebuild: bool = False):
        """Build searchable database from all curriculum PDFs"""
        if not force_rebuild and self.curriculum_db:
            logger.info("Using cached curriculum database")
            self.debug_database_contents()  # Debug existing cache
            return
        
        curricula_dir = Path(settings.curricula_dir)
        pdf_files = list(curricula_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No curriculum PDFs found")
            return
        
        logger.info(f"Building curriculum database from {len(pdf_files)} PDFs")
        
        for pdf_file in pdf_files:
            try:
                # Extract text
                text = extract_text_from_pdf(pdf_file)
                if not text:
                    continue
                
                # Chunk text into semantic units
                chunks = self._chunk_curriculum_text(text)
                
                # Create embeddings
                embeddings = self.embeddings_model.encode(chunks)
                
                # Store in database
                file_key = pdf_file.stem
                self.curriculum_db[file_key] = {
                    'filename': pdf_file.name,
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'metadata': {
                        'education_level': self._classify_education_level(pdf_file.name),
                        'subject': self._classify_subject(pdf_file.name),
                        'file_size': pdf_file.stat().st_size
                    }
                }
                
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        # Save to cache
        self._save_cache()
        logger.info(f"Built curriculum database with {len(self.curriculum_db)} curricula")
        
        # Debug the contents
        self.debug_database_contents()

    def debug_database_contents(self):
        """Debug method to see what's in the RAG database"""
        print("\n" + "="*60)
        print("RAG DATABASE CONTENTS")
        print("="*60)
        
        if not self.curriculum_db:
            print("Database is empty!")
            return
            
        for file_key, data in self.curriculum_db.items():
            print(f"\nFile: {data['filename']}")
            print(f"  Education level: {data['metadata']['education_level']}")
            print(f"  Subject: {data['metadata']['subject']}")
            print(f"  Total chunks: {len(data['chunks'])}")
            print("  Sample chunks:")
            for i, chunk in enumerate(data['chunks'][:3]):
                print(f"    Chunk {i}: {chunk[:200]}...")
            print("-" * 50)
    def retrieve_relevant_context(self, query: str, top_k: int = 3, 
                                 education_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve most relevant curriculum chunks for a query (limited for better prompt quality)"""
        if not self.curriculum_db:
            logger.warning("Curriculum database is empty")
            return []
        
        # Encode query
        query_embedding = self.embeddings_model.encode([query])
        
        # Search across all curricula
        results = []
        
        for file_key, curriculum_data in self.curriculum_db.items():
            # Filter by education level if specified
            if education_level and curriculum_data['metadata']['education_level'] != education_level:
                continue
            
            # Calculate similarities
            similarities = np.dot(query_embedding, curriculum_data['embeddings'].T).flatten()
            
            # Get top chunks from this curriculum
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Increased similarity threshold
                    # Truncate chunk text to avoid overwhelming prompts
                    chunk_text = curriculum_data['chunks'][idx]
                    if len(chunk_text) > 500:
                        chunk_text = chunk_text[:500] + "..."
                    
                    results.append({
                        'text': chunk_text,
                        'similarity': float(similarities[idx]),
                        'source_file': curriculum_data['filename'],
                        'education_level': curriculum_data['metadata']['education_level'],
                        'subject': curriculum_data['metadata']['subject']
                    })
        
        # Sort by similarity and return top results (limit to 3 max)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:3]
    
    def get_curriculum_progression(self, topic: str) -> List[Dict[str, Any]]:
        """Get how a topic progresses across grade levels"""
        # Find topic mentions across different education levels
        levels = ['primary', 'secondary_lower', 'secondary_upper']
        progression = []
        
        for level in levels:
            relevant_chunks = self.retrieve_relevant_context(topic, top_k=2, education_level=level)
            if relevant_chunks:
                progression.append({
                    'education_level': level,
                    'content': relevant_chunks[0]['text'],
                    'similarity': relevant_chunks[0]['similarity']
                })
        
        return progression
    
    def find_prerequisite_relationships(self, objective: str) -> List[Dict[str, Any]]:
        """Find potential prerequisite relationships using semantic similarity"""
        query = f"προαπαιτούμενα {objective}"  # Prerequisites + objective in Greek
        results = self.retrieve_relevant_context(query, top_k=5)
        
        # Filter for actual prerequisite content
        prerequisites = []
        for result in results:
            text = result['text'].lower()
            if any(keyword in text for keyword in ['προαπαιτούμενο', 'πριν', 'μετά', 'βάση']):
                prerequisites.append(result)
        
        return prerequisites
    
    def _chunk_curriculum_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Chunk curriculum text into smaller semantic units to avoid prompt overflow"""
        # Split by common curriculum section markers
        section_markers = [
            'ΣΤΟΧΟΙ', 'ΣΚΟΠΟΙ', 'ΜΑΘΗΣΙΑΚΟΙ ΣΤΟΧΟΙ',
            'ΘΕΜΑΤΙΚΕΣ ΕΝΟΤΗΤΕΣ', 'ΠΕΡΙΕΧΟΜΕΝΟ',
            'ΑΞΙΟΛΟΓΗΣΗ', 'ΜΕΘΟΔΟΛΟΓΙΑ', 'ΔΕΞΙΟΤΗΤΕΣ'
        ]
        
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            is_section_header = any(marker in line.upper() for marker in section_markers)
            
            if is_section_header and current_chunk:
                # Save current chunk and start new one
                if len(current_chunk) > 30:  # Minimum chunk size
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += " " + line
                
                # Split long chunks to keep them manageable
                if len(current_chunk) > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk and len(current_chunk) > 30:
            chunks.append(current_chunk.strip())
        
        # Filter out very short or generic chunks
        filtered_chunks = []
        for chunk in chunks:
            if (len(chunk) > 30 and 
                not chunk.lower().startswith('σελίδα') and 
                not chunk.lower().startswith('κεφάλαιο')):
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def _classify_education_level(self, filename: str) -> str:
        """Classify education level from filename"""
        filename_lower = filename.lower()
        if 'δημοτικ' in filename_lower or 'primary' in filename_lower:
            return 'primary'
        elif 'γυμνάσι' in filename_lower or 'secondary_lower' in filename_lower:
            return 'secondary_lower'
        elif 'λύκει' in filename_lower or 'secondary_upper' in filename_lower:
            return 'secondary_upper'
        else:
            return 'unknown'
    
    def _classify_subject(self, filename: str) -> str:
        """Classify subject from filename"""
        filename_lower = filename.lower()
        if any(term in filename_lower for term in ['ελληνικ', 'γλώσσα', 'λογοτεχνία']):
            return 'greek_language'
        elif 'μαθηματικ' in filename_lower:
            return 'mathematics'
        elif any(term in filename_lower for term in ['φυσικ', 'επιστήμ', 'βιολογ']):
            return 'science'
        elif 'τπε' in filename_lower:
            return 'ict'
        else:
            return 'other'
    
    def _load_cache(self):
        """Load cached embeddings and database"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                
            if self.db_file.exists():
                with open(self.db_file, 'rb') as f:
                    self.curriculum_db = pickle.load(f)
                    
                logger.info(f"Loaded cached curriculum database with {len(self.curriculum_db)} curricula")
                
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
    
    def _save_cache(self):
        """Save embeddings and database to cache"""
        try:
            os.makedirs(settings.cache_dir, exist_ok=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
                
            with open(self.db_file, 'wb') as f:
                pickle.dump(self.curriculum_db, f)
                
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

# Global RAG service instance
rag_service = CurriculumRAGService()