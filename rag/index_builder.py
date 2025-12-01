"""
FAISS Index Builder for RAG System

Creates FAISS vector index from knowledge base markdown files.
Designed for Docker build-time execution to reduce cold-start latency.

Reference:
    - leibniz_rag.py (lines 226-323, 626-655) - Original index building logic

Usage:
    # Command-line
    python -m leibniz_agent.services.rag.index_builder \\
        --knowledge-base leibniz_knowledge_base \\
        --output leibniz_agent/services/rag/index

    # Programmatic
    builder = IndexBuilder(config)
    success = builder.build_index()
"""

import os
import json
import logging
import argparse
import re
from typing import List, Dict, Any
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings

from leibniz_agent.services.rag.config import RAGConfig
from leibniz_agent.services.rag.chunking import intelligent_chunk_text

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builds FAISS index from knowledge base markdown files.
    
    Attributes:
        config: RAG configuration
        embeddings: HuggingFace embeddings model
        documents: Document chunks
        doc_metadata: Chunk metadata
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize builder with configuration.
        
        Args:
            config: RAG configuration instance
        """
        self.config = config
        
        # Initialize HuggingFace embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Storage
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        
        logger.info(f" IndexBuilder initialized: model={config.embedding_model_name}")
    
    def build_index(self) -> bool:
        """
        Build FAISS index from knowledge base.
        
        Process:
            1. Scan all markdown files in 12 category subdirectories
            2. Skip files starting with 00- (README, index files)
            3. Apply intelligent chunking via chunking.py
            4. Create embeddings using HuggingFace model
            5. Build FAISS IndexFlatL2
            6. Save index.faiss, metadata.json, texts.json
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate knowledge base path
            if not os.path.exists(self.config.knowledge_base_path):
                logger.error(f" Knowledge base not found: {self.config.knowledge_base_path}")
                return False
            
            logger.info(f" Scanning knowledge base: {self.config.knowledge_base_path}")
            
            all_documents = []
            all_metadata = []
            
            # Scan all markdown files
            for root, dirs, files in os.walk(self.config.knowledge_base_path):
                for file in files:
                    if not file.endswith('.md'):
                        continue
                    
                    # Skip README/index files
                    if file.startswith('00-'):
                        logger.debug(f"Skipping index file: {file}")
                        continue
                    
                    file_path = os.path.join(root, file)
                    
                    # Extract category from directory name
                    category = None
                    if root != self.config.knowledge_base_path:
                        # Get immediate parent directory name
                        parent_dir = os.path.basename(root)
                        
                        # Extract category from numeric prefix (e.g., "01_university_overview" -> "university_overview")
                        match = re.match(r'^\d+_(.+)$', parent_dir)
                        if match:
                            category = match.group(1)
                        else:
                            # Check if we're in root directory
                            if root == self.config.knowledge_base_path:
                                continue  # Skip root-level files
                            # Otherwise, it's a valid category without numeric prefix
                            category = parent_dir
                    else:
                        # Skip root-level files without category
                        continue
                    
                    # Read file content
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if not content.strip():
                            logger.warning(f"Empty file: {file}")
                            continue
                        
                        # Apply intelligent chunking
                        chunks = intelligent_chunk_text(content, file, self.config)
                        
                        # Add chunks with metadata
                        for idx, chunk in enumerate(chunks):
                            priority = self.get_content_priority(chunk, file, category)
                            
                            all_documents.append(chunk)
                            all_metadata.append({
                                'source': file,
                                'category': category,
                                'chunk_id': idx,
                                'priority': priority
                            })
                        
                        logger.debug(f"Processed {file}: {len(chunks)} chunks")
                    
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
                        continue
            
            # Validate documents
            if not all_documents:
                logger.error(" No documents found in knowledge base")
                return False
            
            logger.info(f" Total documents: {len(all_documents)} from {len(set(m['source'] for m in all_metadata))} files")
            
            # Create embeddings
            logger.info(" Creating embeddings...")
            embeddings_array = self.embeddings.embed_documents(all_documents)
            embeddings_array = np.array(embeddings_array, dtype=np.float32)
            
            # Build FAISS index
            dimension = embeddings_array.shape[1]
            logger.info(f" Building FAISS index: dimension={dimension}")
            
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Save to disk
            os.makedirs(self.config.vector_store_path, exist_ok=True)
            
            index_path = os.path.join(self.config.vector_store_path, "index.faiss")
            metadata_path = os.path.join(self.config.vector_store_path, "metadata.json")
            texts_path = os.path.join(self.config.vector_store_path, "texts.json")
            
            faiss.write_index(index, index_path)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)
            
            with open(texts_path, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            
            # Update instance variables
            self.documents = all_documents
            self.doc_metadata = all_metadata
            
            logger.info(f" FAISS index built successfully: {len(all_documents)} chunks")
            logger.info(f"   Index file: {index_path}")
            logger.info(f"   Metadata file: {metadata_path}")
            logger.info(f"   Texts file: {texts_path}")
            
            return True
        
        except Exception as e:
            logger.error(f" Error building index: {e}", exc_info=True)
            return False
    
    def get_content_priority(self, chunk: str, filename: str, category: str) -> int:
        """
        Assign priority scores to chunks for boosting during retrieval.
        
        Args:
            chunk: Document chunk text
            filename: Source filename
            category: Document category
            
        Returns:
            Priority score (higher = more important)
        """
        priority = 0
        
        # Boost for filename keywords
        filename_keywords = [
            'admission', 'enrollment', 'program', 'course',
            'tuition', 'fee', 'housing', 'library', 'contact', 'office'
        ]
        
        filename_lower = filename.lower()
        if any(kw in filename_lower for kw in filename_keywords):
            priority += 5
        
        # Boost for university-related terms in chunk
        chunk_lower = chunk.lower()
        university_terms = [
            'university', 'leibniz', 'hannover', 'campus',
            'student', 'faculty', 'department', 'program',
            'course', 'admission', 'enrollment'
        ]
        
        term_count = sum(1 for term in university_terms if term in chunk_lower)
        priority += min(term_count, 3)  # Max +3 for terms
        
        # Boost for longer chunks (more comprehensive)
        if len(chunk) > 600:
            priority += 2
        
        return priority
    
    def load_existing_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(self.config.vector_store_path, "index.faiss")
            metadata_path = os.path.join(self.config.vector_store_path, "metadata.json")
            texts_path = os.path.join(self.config.vector_store_path, "texts.json")
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [index_path, metadata_path, texts_path]):
                logger.warning("️ Index files not found")
                return False
            
            # Load FAISS index
            # Note: We don't store the index in this builder, just verify it loads
            index = faiss.read_index(index_path)
            
            # Load metadata and texts
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.doc_metadata = json.load(f)
            
            with open(texts_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            logger.info(f" Loaded existing index: {len(self.documents)} chunks")
            return True
        
        except Exception as e:
            logger.error(f" Error loading index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        # Determine embedding dimension dynamically
        embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Try to get from FAISS index if available
        index_path = os.path.join(self.config.vector_store_path, "index.faiss")
        if os.path.exists(index_path):
            try:
                temp_index = faiss.read_index(index_path)
                embedding_dimension = temp_index.d
            except Exception:
                # If documents exist, infer from first document embedding
                if self.documents:
                    try:
                        test_embedding = self.embeddings.embed_query(self.documents[0])
                        embedding_dimension = len(test_embedding)
                    except Exception:
                        pass  # Use default
        
        stats = {
            'total_documents': len(self.documents),
            'total_files': len(set(m['source'] for m in self.doc_metadata)) if self.doc_metadata else 0,
            'categories': len(set(m['category'] for m in self.doc_metadata)) if self.doc_metadata else 0,
            'embedding_dimension': embedding_dimension,
        }
        
        # Get index file size if exists
        if os.path.exists(index_path):
            stats['index_size_bytes'] = os.path.getsize(index_path)
        
        return stats


def main():
    """CLI interface for index building."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index from knowledge base"
    )
    parser.add_argument(
        '--knowledge-base',
        type=str,
        default=None,
        help='Knowledge base directory path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for index files'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config from environment
    config = RAGConfig.from_env()
    
    # Override with arguments if provided
    if args.knowledge_base:
        config.knowledge_base_path = args.knowledge_base
    if args.output:
        config.vector_store_path = args.output
    
    # Create builder
    builder = IndexBuilder(config)
    
    # Check if index exists
    if not args.rebuild and builder.load_existing_index():
        logger.info("Index already exists. Use --rebuild to force rebuild.")
        stats = builder.get_index_stats()
        logger.info(f"Index stats: {stats}")
        return 0
    
    # Build index
    success = builder.build_index()
    
    if success:
        stats = builder.get_index_stats()
        logger.info(f" Index statistics:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        return 0
    else:
        logger.error("Failed to build index")
        return 1


if __name__ == "__main__":
    exit(main())
