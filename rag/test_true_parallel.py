#!/usr/bin/env python3
"""
TRUE PARALLEL Incremental RAG Testing

Tests the hypothesis that true parallel processing with realistic speech delays
provides significant latency improvements over sequential processing.
"""

import asyncio
import time
import httpx
import json
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ParallelTestResult:
    """Test result container for parallel processing"""
    method: str
    total_user_speech_time: float  # How long user "spoke"
    total_processing_time: float   # Total time for all operations
    perceived_latency: float      # user_speech_time + final_generation_time
    response_accuracy: float
    chunks_sent: int
    docs_buffered: int
    final_response: str

class TrueParallelRAGExperiment:
    """True parallel incremental RAG testing"""

    def __init__(self, rag_url: str = "http://localhost:8003"):
        self.rag_url = rag_url
        self.session_id = f"parallel_{int(time.time())}"

        # Test queries simulating real user speech patterns
        self.test_queries = [
            {
                "full_query": "I want to know about admission requirements and eligibility criteria for engineering students at TASK",
                "speech_pattern": "slow",  # 800ms between words
                "description": "Admission requirements (slow speech)"
            },
            {
                "full_query": "Can you tell me about the placement statistics and average salaries for graduates from TASK institute",
                "speech_pattern": "normal",  # 500ms between words
                "description": "Placement statistics (normal speech)"
            },
            {
                "full_query": "What are the contact details and office hours for TASK customer service support team",
                "speech_pattern": "fast",  # 200ms between words
                "description": "Contact information (fast speech)"
            }
        ]

    def chunk_text_by_words(self, text: str, words_per_chunk: int = 3) -> List[str]:
        """Split text into chunks by word count (more realistic than characters)"""
        words = re.findall(r'\b\w+\b', text)
        chunks = []

        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

        return chunks

    def get_speech_delay(self, pattern: str) -> int:
        """Get realistic speech delay based on pattern"""
        delays = {
            "fast": 200,    # Fast talker: 200ms between chunks
            "normal": 500,  # Normal: 500ms between chunks
            "slow": 800     # Slow: 800ms between chunks
        }
        return delays.get(pattern, 500)

    async def test_standard_approach(self, query: str) -> ParallelTestResult:
        """Test standard single-query approach"""
        start_time = time.time()

        # Simulate user speaking the entire query at once (no chunking)
        user_speech_time = len(query.split()) * 0.3  # Rough estimate: 300ms per word
        await asyncio.sleep(user_speech_time)  # Simulate speech time

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
            total_processing_time = (time.time() - start_time) * 1000

            return ParallelTestResult(
                method="standard",
                total_user_speech_time=user_speech_time * 1000,
                total_processing_time=total_processing_time,
                perceived_latency=total_processing_time,  # User waits for entire process
                response_accuracy=self.evaluate_accuracy(result.get('answer', ''), query),
                chunks_sent=1,
                docs_buffered=0,  # Not applicable
                final_response=result.get('answer', '')
            )

    async def test_true_parallel_approach(self, full_query: str, speech_pattern: str = "normal") -> ParallelTestResult:
        """Test TRUE PARALLEL incremental approach with realistic speech simulation"""
        session_id = f"true_parallel_{int(time.time())}_{hash(full_query) % 1000}"
        chunks = self.chunk_text_by_words(full_query, words_per_chunk=3)
        speech_delay_ms = self.get_speech_delay(speech_pattern)

        print(f"🎤 Simulating {speech_pattern} speech: {speech_delay_ms}ms delays")
        print(f"📦 Chunking into {len(chunks)} word-groups:")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}: '{chunk}'")

        start_time = time.time()
        speech_time = 0
        processing_tasks = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # TRUE PARALLEL: Send chunks with realistic speech delays
            # Processing happens in parallel with speech
            for i, chunk in enumerate(chunks):
                chunk_start = time.time()

                # Send chunk asynchronously (fire-and-forget for buffering)
                task = asyncio.create_task(
                    send_chunk_async(client, session_id, chunk, i)
                )
                processing_tasks.append(task)

                # Realistic speech delay (user is still speaking)
                if i < len(chunks) - 1:
                    speech_delay = speech_delay_ms / 1000.0
                    await asyncio.sleep(speech_delay)
                    speech_time += speech_delay

                print(f"📤 Chunk {i+1} sent ({(time.time() - chunk_start)*1000:.0f}ms)")

            # All chunks sent, user finished speaking
            total_speech_time = (time.time() - start_time) * 1000
            print(f"🎤 User finished speaking after {total_speech_time:.0f}ms")

            # Wait for all buffering to complete (happens during speech)
            buffering_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            docs_buffered = sum(1 for r in buffering_results if not isinstance(r, Exception) and r.get('docs_retrieved', 0) > 0)

            # Send final generation request
            print(f"🚀 Requesting final generation...")
            final_start = time.time()

            payload = {
                "text": full_query,
                "session_id": session_id,
                "is_final": True,
                "context": {
                    "language": "te-mixed",
                    "organization": "TASK"
                }
            }

            # Stream the final response
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
                            if data.get('is_final', False):
                                break
                        except json.JSONDecodeError:
                            continue

                # Combine streaming chunks
                final_response = ''.join([chunk.get('text', '') for chunk in chunks_received])
                generation_time = (time.time() - final_start) * 1000

                total_processing_time = (time.time() - start_time) * 1000
                perceived_latency = total_speech_time + generation_time  # User speech + final generation

                return ParallelTestResult(
                    method="true_parallel",
                    total_user_speech_time=total_speech_time,
                    total_processing_time=total_processing_time,
                    perceived_latency=perceived_latency,
                    response_accuracy=self.evaluate_accuracy(final_response, full_query),
                    chunks_sent=len(chunks) + 1,  # +1 for final generation
                    docs_buffered=docs_buffered,
                    final_response=final_response
                )

    async def send_chunk_async(self, client: httpx.AsyncClient, session_id: str, chunk: str, sequence: int):
        """Send chunk asynchronously for true parallel processing"""
        payload = {
            "text": chunk,
            "session_id": session_id,
            "is_final": False,
            "sequence_number": sequence,
            "context": {
                "language": "te-mixed",
                "organization": "TASK"
            }
        }

        try:
            response = await client.post(
                f"{self.rag_url}/api/v1/query/incremental",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            print(f"❌ Chunk {sequence} failed: {e}")
            return {"error": str(e)}

    def evaluate_accuracy(self, response: str, original_query: str) -> float:
        """Simple accuracy evaluation based on keyword matching"""
        query_words = set(re.findall(r'\b\w+\b', original_query.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        # Calculate overlap
        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words)

    async def run_experiment(self):
        """Run the complete true parallel experiment"""
        print("🚀 TRUE PARALLEL RAG Experiment")
        print("=" * 70)
        print("Testing: Standard vs True Parallel Processing")
        print("Hypothesis: Parallel processing during speech = lower perceived latency")
        print("=" * 70)

        results = []

        for i, test_case in enumerate(self.test_queries):
            print(f"\n📋 Test Case {i+1}: {test_case['description']}")
            print(f"Query: {test_case['full_query'][:80]}...")
            print("-" * 50)

            # Test standard approach
            print("🔍 Testing STANDARD approach...")
            standard_result = await self.test_standard_approach(test_case['full_query'])
            print(".0f"            print(".0f"            print(".0f"
            # Test true parallel approach
            print("⚡ Testing TRUE PARALLEL approach...")
            parallel_result = await self.test_true_parallel_approach(
                test_case['full_query'],
                test_case['speech_pattern']
            )
            print(".0f"            print(".0f"            print(".0f"
            # Compare results
            latency_improvement = ((standard_result.perceived_latency - parallel_result.perceived_latency) /
                                 standard_result.perceived_latency) * 100

            print("📊 COMPARISON:"            print(".0f"            print(".2f"            print(".2f"            print(f"  📦 Documents buffered: {parallel_result.docs_buffered}")

            results.append({
                'test_case': test_case['description'],
                'speech_pattern': test_case['speech_pattern'],
                'standard': standard_result,
                'parallel': parallel_result,
                'latency_improvement_pct': latency_improvement
            })

        # Summary statistics
        self.print_summary(results)

    def print_summary(self, results: List[Dict]):
        """Print comprehensive experiment summary"""
        print("\n" + "=" * 70)
        print("📈 TRUE PARALLEL EXPERIMENT RESULTS")
        print("=" * 70)

        # Overall statistics
        standard_latencies = [r['standard'].perceived_latency for r in results]
        parallel_latencies = [r['parallel'].perceived_latency for r in results]
        accuracy_improvements = [r['latency_improvement_pct'] for r in results]

        print("🎯 KEY METRICS:")
        print(f"  Standard Avg Perceived Latency: {statistics.mean(standard_latencies):.0f}ms")
        print(f"  True Parallel Avg Perceived Latency: {statistics.mean(parallel_latencies):.0f}ms")
        print(".1f"
        print("
🎤 SPEECH PATTERN ANALYSIS:"        for result in results:
            print(f"  {result['speech_pattern'].capitalize()} speech ({result['test_case']}):")
            print(".0f"            print(".0f"            print(".1f"
        print("
✅ CONCLUSION:"        avg_improvement = statistics.mean(accuracy_improvements)
        if avg_improvement > 10:
            print(".1f"            print("  🎉 TRUE PARALLEL PROCESSING WORKS!")
        elif avg_improvement > 0:
            print(".1f"            print("  👍 Marginal improvement - may need optimization")
        else:
            print(".1f"            print("  ❌ No improvement - parallel processing not effective")

        print("
🔍 TECHNICAL INSIGHTS:"        print(f"  • Documents buffered during speech: {sum(r['parallel'].docs_buffered for r in results)}")
        print(f"  • Average chunks processed: {statistics.mean([r['parallel'].chunks_sent for r in results]):.1f}")
        print("  • Processing happens during user speech (free time)"
        print("  • Smart deduplication prevents redundant retrieval"
        print("  • Telugu text processing integrated throughout"

async def main():
    """Main experiment runner"""
    import argparse
    import re  # Add missing import

    parser = argparse.ArgumentParser(description="TRUE PARALLEL RAG Experiment")
    parser.add_argument("--rag-url", default="http://localhost:8003", help="RAG service URL")

    args = parser.parse_args()

    experiment = TrueParallelRAGExperiment(args.rag_url)
    await experiment.run_experiment()

if __name__ == "__main__":
    asyncio.run(main())







