import os
import sys
import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import from leibniz_agent
# In container, /app is the workdir and leibniz_agent is inside /app
if "/app" not in sys.path:
    sys.path.append("/app")

try:
    from leibniz_agent.services.rag.config import RAGConfig
    from leibniz_agent.services.rag.rag_engine import RAGEngine
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

async def main():
    queries = [
        "I am looking for job oppertunituies at TASK",
        "can you tell me history about the task",
    ]

    print(f"\n🚀 Starting RAG Retrieval Debugger")
    print(f"=====================================================================")

    try:
        config = RAGConfig.from_env()
        print(f"✅ Config loaded. Knowledge base: {config.knowledge_base_path}")
        
        engine = RAGEngine(config)
        
        # Ensure index is loaded
        if not engine.load_index():
            print("❌ Failed to load FAISS index")
            return

        print(f"✅ Loaded index with {len(engine.documents)} documents")
        print(f"   Embedding model: {config.embedding_model_name}")

        import numpy as np

        for q in queries:
            print("\n" + "=" * 80)
            print(f"🔍 QUERY: {q}")
            print("=" * 80)

            # Embed using same path as RAG (sync cache helper)
            # Note: _embed_with_cache_sync might not be available if not initialized with Redis?
            # RAGEngine checks self.redis_client for cache.
            # But we can just use self.embeddings.embed_query(q) directly for debug.
            
            print("   Generating embedding...")
            embedding = engine.embeddings.embed_query(q)
            vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

            print(f"   Searching FAISS (top 5)...")
            distances, indices = engine.vector_store.search(vec, k=5)

            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
                if idx >= len(engine.documents):
                    print(f"  [Rank {rank}] invalid idx={idx}")
                    continue

                doc = engine.documents[idx]
                meta: Dict[str, Any] = engine.doc_metadata[idx] if idx < len(engine.doc_metadata) else {}
                source = meta.get("source", "Unknown")
                category = meta.get("category", "Unknown")
                
                # Approximate cosine similarity from L2 distance (assuming normalized)
                # L2 = 2 * (1 - cosine) -> cosine = 1 - L2/2
                similarity = 1.0 - (float(dist) * float(dist) / 2.0)

                print(f"  [Rank {rank}] Sim: {similarity:.4f} | Dist: {dist:.4f}")
                print(f"      Source: {source} | Category: {category}")
                preview = doc.replace("\n", " ")[:200]
                print(f"      Text: {preview}...")
                print("-" * 40)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
