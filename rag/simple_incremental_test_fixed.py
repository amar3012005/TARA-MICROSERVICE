#!/usr/bin/env python3
"""
Simple Incremental RAG Test - Mock Implementation

Tests the chunking logic and timing without requiring the full RAG service.
"""

import asyncio
import time
import re
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TestResult:
    """Test result container"""
    method: str
    total_latency_ms: float
    response_accuracy: float
    chunks_sent: int
    chunk_latencies: List[float]
    final_response: str

class MockRAGExperiment:
    """Mock experiment to test chunking logic and timing"""

    def __init__(self):
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

    async def mock_standard_query(self, query: str) -> TestResult:
        """Mock standard single-query approach"""
        start_time = time.time()

        # Simulate RAG processing latency (800-1400ms based on logs)
        processing_time = 1000 + (hash(query) % 600)  # 1000-1600ms
        await asyncio.sleep(processing_time / 1000.0)

        # Mock response generation
        response = self.generate_mock_response(query, "standard")

        total_latency = (time.time() - start_time) * 1000

        return TestResult(
            method="standard",
            total_latency_ms=total_latency,
            response_accuracy=self.evaluate_accuracy(response, query),
            chunks_sent=1,
            chunk_latencies=[total_latency],
            final_response=response
        )

    async def mock_incremental_query(self, full_query: str,
                                   chunk_delay_ms: int = 500,
                                   chunk_size_words: int = 3) -> TestResult:
        """Mock incremental chunked approach"""
        chunks = self.chunk_text(full_query, chunk_size_words)

        print(f"📦 Chunking '{full_query[:50]}...' into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}: '{chunk}'")

        chunk_latencies = []
        start_time = time.time()

        # Phase 1: Process chunks incrementally (fast document retrieval)
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()

            # Simulate fast document retrieval (50-200ms)
            retrieval_time = 50 + (hash(chunk) % 150)
            await asyncio.sleep(retrieval_time / 1000.0)

            chunk_latency = (time.time() - chunk_start) * 1000
            chunk_latencies.append(chunk_latency)

            print(f"  ✅ Retrieved docs for chunk {i+1} in {chunk_latency:.0f}ms")

            # Simulate human speech delay (except for last chunk)
            if i < len(chunks) - 1:
                await asyncio.sleep(chunk_delay_ms / 1000.0)

        # Phase 2: Generate final response (single LLM call)
        generation_start = time.time()

        # Simulate final LLM generation (300-800ms)
        generation_time = 300 + (hash(full_query) % 500)
        await asyncio.sleep(generation_time / 1000.0)

        # Mock response generation
        response = self.generate_mock_response(full_query, "incremental")

        generation_latency = (time.time() - generation_start) * 1000
        chunk_latencies.append(generation_latency)

        total_latency = (time.time() - start_time) * 1000

        return TestResult(
            method="incremental",
            total_latency_ms=total_latency,
            response_accuracy=self.evaluate_accuracy(response, full_query),
            chunks_sent=len(chunks) + 1,  # +1 for final generation
            chunk_latencies=chunk_latencies,
            final_response=response
        )

    def generate_mock_response(self, query: str, method: str) -> str:
        """Generate mock RAG response"""
        if "admission" in query.lower():
            return "TASK admission requirements include engineering diploma, minimum 60% marks, and valid entrance exam scores. Engineering students can apply through the online portal."
        elif "placement" in query.lower():
            return "TASK placement statistics show 85% placement rate with average salary of ₹4.5 LPA. Top companies include TCS, Infosys, and Wipro."
        elif "contact" in query.lower():
            return "TASK customer service: Phone 040-12345678, Email info@task.telangana.gov.in, Office hours 9AM-6PM Monday-Saturday."
        else:
            return f"This is a mock response for query: {query[:50]}... (method: {method})"

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
        print("🚀 Starting RAG Incremental Processing Experiment (Mock)")
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
            standard_result = await self.mock_standard_query(test_case['full_query'])
            print(f"  ⏱️  Total latency: {standard_result.total_latency_ms:.0f}ms")
            print(f"  🎯 Response: {standard_result.final_response[:100]}...")

            # Test incremental approach
            print("⚡ Testing INCREMENTAL approach...")
            incremental_result = await self.mock_incremental_query(
                test_case['full_query'],
                chunk_delay_ms,
                chunk_size_words
            )
            print(f"  ⏱️  Total latency: {incremental_result.total_latency_ms:.0f}ms")
            print(f"  📦 Chunks processed: {incremental_result.chunks_sent}")
            print(f"  🎯 Response: {incremental_result.final_response[:100]}...")

            # Compare results
            latency_improvement = ((standard_result.total_latency_ms - incremental_result.total_latency_ms) /
                                 standard_result.total_latency_ms) * 100

            print("📊 COMPARISON:")
            print(f"  ⚡ Latency improvement: {latency_improvement:.1f}%")
            print(f"  🎯 Accuracy - Standard: {standard_result.response_accuracy:.2f}, Incremental: {incremental_result.response_accuracy:.2f}")
            print(f"  📦 Chunks - Standard: {standard_result.chunks_sent}, Incremental: {incremental_result.chunks_sent}")

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

        print("LATENCY COMPARISON:")
        print(f"  Standard - Avg: {statistics.mean(standard_latencies):.0f}ms, Min: {min(standard_latencies):.0f}ms, Max: {max(standard_latencies):.0f}ms")
        print(f"  Incremental - Avg: {statistics.mean(incremental_latencies):.0f}ms, Min: {min(incremental_latencies):.0f}ms, Max: {max(incremental_latencies):.0f}ms")
        print(f"  Overall Improvement: {statistics.mean(accuracy_improvements):.2f}%")

        print("\nACCURACY COMPARISON:")
        standard_accuracies = [r['standard'].response_accuracy for r in results]
        incremental_accuracies = [r['incremental'].response_accuracy for r in results]
        print(f"  Standard - Avg: {statistics.mean(standard_accuracies):.2f}, Min: {min(standard_accuracies):.2f}, Max: {max(standard_accuracies):.2f}")
        print(f"  Incremental - Avg: {statistics.mean(incremental_accuracies):.2f}, Min: {min(incremental_accuracies):.2f}, Max: {max(incremental_accuracies):.2f}")

        print("\nKEY FINDINGS:")
        avg_improvement = statistics.mean(accuracy_improvements)
        if avg_improvement > 0:
            print(f"  ✅ Average latency improvement: {avg_improvement:.1f}%")
        else:
            print(f"  ❌ No latency improvement: {avg_improvement:.1f}%")

        # Detailed breakdown
        print("\nDETAILED RESULTS:")
        for i, result in enumerate(results):
            print(f"Test {i+1} ({result['test_case']}):")
            print(f"  Standard latency: {result['standard'].total_latency_ms:.0f}ms")
            print(f"  Incremental latency: {result['incremental'].total_latency_ms:.0f}ms")
            print(f"  Improvement: {result['latency_improvement_pct']:.2f}%")

async def main():
    """Main experiment runner"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Incremental Processing Experiment (Mock)")
    parser.add_argument("--chunk-delay", type=int, default=500, help="Delay between chunks (ms)")
    parser.add_argument("--chunk-size", type=int, default=3, help="Words per chunk")

    args = parser.parse_args()

    experiment = MockRAGExperiment()
    await experiment.run_experiment(args.chunk_delay, args.chunk_size)

if __name__ == "__main__":
    asyncio.run(main())
