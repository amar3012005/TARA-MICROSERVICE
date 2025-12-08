#!/usr/bin/env python3
"""
Experimental RAG Testing: Incremental vs Standard Processing

Tests the hypothesis that sending user input as 2-3 word chunks with realistic
speech delays can improve latency while maintaining accuracy through parallel
document retrieval.
"""

import asyncio
import time
import httpx
import json
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class TestResult:
    """Test result container"""
    method: str
    total_latency_ms: float
    response_accuracy: float
    chunks_sent: int
    chunk_latencies: List[float]
    final_response: str
    sources: List[str]
    confidence: float

class IncrementalRAGExperiment:
    """Experimental incremental RAG testing"""

    def __init__(self, rag_url: str = "http://localhost:8003"):
        self.rag_url = rag_url
        self.session_id = f"experiment_{int(time.time())}"

        # Test queries (long user inputs to chunk)
        self.test_queries = [
            {
                "full_query": "I want to know about admission requirements and eligibility criteria for engineering students at TASK",
                "expected_keywords": ["admission", "requirements", "eligibility", "engineering", "students", "TASK"],
                "description": "Admission requirements query"
            },
            {
                "full_query": "Can you tell me about the placement statistics and average salaries for graduates from TASK institute",
                "expected_keywords": ["placement", "statistics", "salaries", "graduates", "TASK"],
                "description": "Placement statistics query"
            },
            {
                "full_query": "What are the contact details and office hours for TASK customer service support team",
                "expected_keywords": ["contact", "office", "hours", "customer", "service", "support"],
                "description": "Contact information query"
            }
        ]

    def chunk_text(self, text: str, chunk_size_words: int = 3) -> List[str]:
        """Split text into chunks of approximately chunk_size_words"""
        words = re.findall(r'\b\w+\b', text)
        chunks = []

        for i in range(0, len(words), chunk_size_words):
            chunk_words = words[i:i + chunk_size_words]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

        return chunks

    async def test_standard_approach(self, query: str) -> TestResult:
        """Test standard single-query approach"""
        start_time = time.time()

        payload = {
            "query": query,
            "context": {
                "language": "te-mixed",
                "organization": "TASK"
            }
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.rag_url}/api/v1/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            result = response.json()
            total_latency = (time.time() - start_time) * 1000

            return TestResult(
                method="standard",
                total_latency_ms=total_latency,
                response_accuracy=self.evaluate_accuracy(result.get('answer', ''), query),
                chunks_sent=1,
                chunk_latencies=[total_latency],
                final_response=result.get('answer', ''),
                sources=result.get('sources', []),
                confidence=result.get('confidence', 0.0)
            )

    async def test_incremental_approach(self, full_query: str,
                                       chunk_delay_ms: int = 500,
                                       chunk_size_words: int = 3) -> TestResult:
        """Test incremental chunked approach"""
        session_id = f"incr_{int(time.time())}_{hash(full_query) % 1000}"
        chunks = self.chunk_text(full_query, chunk_size_words)

        print(f"📦 Chunking '{full_query[:50]}...' into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}: '{chunk}'")

        chunk_latencies = []
        start_time = time.time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Phase 1: Send chunks incrementally (is_final=false)
            for i, chunk in enumerate(chunks):
                chunk_start = time.time()

                payload = {
                    "text": chunk,
                    "session_id": session_id,
                    "is_final": False,
                    "sequence_number": i,
                    "context": {
                        "language": "te-mixed",
                        "organization": "TASK"
                    }
                }

                print(f"📤 Sending chunk {i+1}/{len(chunks)}: '{chunk}'")

                response = await client.post(
                    f"{self.rag_url}/api/v1/query/incremental",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                result = response.json()
                chunk_latency = (time.time() - chunk_start) * 1000
                chunk_latencies.append(chunk_latency)

                print(f"  ✅ Buffered {result.get('docs_retrieved', 0)} docs in {chunk_latency:.0f}ms (total: {result.get('buffer_size_chars', 0)} chars)")

                # Simulate human speech delay (except for last chunk)
                if i < len(chunks) - 1:
                    await asyncio.sleep(chunk_delay_ms / 1000.0)

            # Phase 2: Send final chunk (is_final=true) to trigger generation
            print(f"🎯 Sending final chunk to trigger generation...")
            final_start = time.time()

            payload = {
                "text": full_query,  # Send complete text for generation
                "session_id": session_id,
                "is_final": True,
                "context": {
                    "language": "te-mixed",
                    "organization": "TASK"
                }
            }

            # For streaming response, we need to handle Server-Sent Events
            async with client.stream(
                "POST",
                f"{self.rag_url}/api/v1/query/incremental",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                chunks_received = []
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            chunks_received.append(data)
                            print(f"  📄 Received: '{data.get('text', '')[:50]}...' (final: {data.get('is_final', False)})")
                            if data.get('is_final', False):
                                break
                        except json.JSONDecodeError:
                            continue

                # Combine streaming chunks
                final_response = ''.join([chunk.get('text', '') for chunk in chunks_received])
                generation_latency = (time.time() - final_start) * 1000
                chunk_latencies.append(generation_latency)

                total_latency = (time.time() - start_time) * 1000

                return TestResult(
                    method="incremental",
                    total_latency_ms=total_latency,
                    response_accuracy=self.evaluate_accuracy(final_response, full_query),
                    chunks_sent=len(chunks) + 1,  # +1 for final generation request
                    chunk_latencies=chunk_latencies,
                    final_response=final_response,
                    sources=[],  # Would need to extract from response
                    confidence=0.0  # Would need to extract from response
                )

    def evaluate_accuracy(self, response: str, original_query: str) -> float:
        """Simple accuracy evaluation based on keyword matching"""
        query_words = set(re.findall(r'\b\w+\b', original_query.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        # Calculate overlap
        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words)

    async def run_experiment(self, chunk_delay_ms: int = 500, chunk_size_words: int = 3):
        """Run the complete experiment"""
        print("🚀 Starting RAG Incremental Processing Experiment")
        print("=" * 60)
        print(f"Chunk delay: {chunk_delay_ms}ms | Chunk size: {chunk_size_words} words")
        print("=" * 60)

        results = []

        for i, test_case in enumerate(self.test_queries):
            print(f"\n📋 Test Case {i+1}: {test_case['description']}")
            print(f"Query: {test_case['full_query'][:80]}...")
            print("-" * 40)

            # Test standard approach
            print("🔍 Testing STANDARD approach...")
            standard_result = await self.test_standard_approach(test_case['full_query'])
            print(".0f"
            # Test incremental approach
            print("⚡ Testing INCREMENTAL approach...")
            incremental_result = await self.test_incremental_approach(
                test_case['full_query'],
                chunk_delay_ms,
                chunk_size_words
            )
            print(".0f"
            # Compare results
            latency_improvement = ((standard_result.total_latency_ms - incremental_result.total_latency_ms) /
                                 standard_result.total_latency_ms) * 100

            print("📊 COMPARISON:"            print(".0f"            print(".2f"            print(".2f"
            results.append({
                'test_case': test_case['description'],
                'standard': standard_result,
                'incremental': incremental_result,
                'latency_improvement_pct': latency_improvement
            })

        # Summary statistics
        self.print_summary(results)

    def print_summary(self, results: List[Dict]):
        """Print experiment summary"""
        print("\n" + "=" * 60)
        print("📈 EXPERIMENT SUMMARY")
        print("=" * 60)

        standard_latencies = [r['standard'].total_latency_ms for r in results]
        incremental_latencies = [r['incremental'].total_latency_ms for r in results]
        accuracy_improvements = [r['latency_improvement_pct'] for r in results]

        print("LATENCY COMPARISON:"        print(".0f"        print(".0f"        print(".0f"        print(".2f"
        print("
ACCURACY COMPARISON:"        print(".2f"        print(".2f"        print(".2f"
        print("
KEY FINDINGS:"        avg_improvement = statistics.mean(accuracy_improvements)
        if avg_improvement > 0:
            print(".1f"        else:
            print(".1f"
        # Detailed breakdown
        print("
DETAILED RESULTS:"        for i, result in enumerate(results):
            print(f"Test {i+1} ({result['test_case']}):")
            print(".0f"            print(".2f"            print(".2f"
async def main():
    """Main experiment runner"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Incremental Processing Experiment")
    parser.add_argument("--rag-url", default="http://localhost:8003", help="RAG service URL")
    parser.add_argument("--chunk-delay", type=int, default=500, help="Delay between chunks (ms)")
    parser.add_argument("--chunk-size", type=int, default=3, help="Words per chunk")

    args = parser.parse_args()

    experiment = IncrementalRAGExperiment(args.rag_url)
    await experiment.run_experiment(args.chunk_delay, args.chunk_size)

if __name__ == "__main__":
    asyncio.run(main())






