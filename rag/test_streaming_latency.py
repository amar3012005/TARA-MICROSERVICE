#!/usr/bin/env python3
"""
Test Pre-LLM Accumulation Latency - Realistic Streaming Simulation

This test measures the TRUE latency improvement from Pre-LLM Accumulation by:
1. Simulating partial STT chunks (is_final=false) - like real speech streaming
2. Measuring time from FINAL chunk send to response received
3. Comparing with baseline (no pre-accumulation)

IMPORTANT: This test uses the /api/v1/query/incremental endpoint which implements
the Pre-LLM accumulation feature. The regular /api/v1/query endpoint does NOT
support this feature.
"""

import asyncio
import aiohttp
import time
import json
import uuid
from typing import List, Dict, Tuple

RAG_URL = "http://localhost:2003"
INCREMENTAL_ENDPOINT = f"{RAG_URL}/api/v1/query/incremental"
QUERY_ENDPOINT = f"{RAG_URL}/api/v1/query"

# Different test queries to avoid global cache hits
TEST_QUERIES = {
    "prellm": {
        "partials": ["టాస్క్", "టాస్క్ అంటే", "టాస్క్ అంటే ఏమిటి"],
        "final": "టాస్క్ అంటే ఏమిటి"
    },
    "baseline": {
        "final": "సబ్ టాస్క్ అంటే ఏమిటి"  # Different query to avoid cache
    },
    "iteration_prellm": [
        {"partials": ["రీమైండర్", "రీమైండర్ అంటే"], "final": "రీమైండర్ అంటే ఏమిటి"},
        {"partials": ["నాటిఫికేషన్", "నాటిఫికేషన్ అంటే"], "final": "నాటిఫికేషన్ అంటే ఏమిటి"},
        {"partials": ["అపాయింట్మెంట్", "అపాయింట్మెంట్ అంటే"], "final": "అపాయింట్మెంట్ అంటే ఏమిటి"},
    ],
    "iteration_baseline": [
        "టాస్క్ క్రియేట్ చేయడం ఎలా",
        "సబ్ టాస్క్ ఎలా క్రియేట్ చేయాలి",
        "టాస్క్ డిలీట్ చేయడం ఎలా",
    ]
}

async def send_incremental_chunk(session: aiohttp.ClientSession, text: str, session_id: str, is_final: bool) -> Tuple[float, Dict]:
    """
    Send a chunk to the INCREMENTAL endpoint which supports Pre-LLM accumulation.
    
    For is_final=False: Returns buffer status (quick, no LLM call)
    For is_final=True: Returns streaming response with LLM generation (using cached context)
    """
    start = time.perf_counter()
    
    # Payload format for IncrementalQueryRequest
    payload = {
        "session_id": session_id,
        "text": text,
        "is_final": is_final,
        "context": {
            "language": "te-mixed"
        }
    }
    
    result = {}
    
    async with session.post(INCREMENTAL_ENDPOINT, json=payload) as resp:
        if is_final:
            # Final response is STREAMING - read all chunks
            full_text = ""
            cached_detected = False
            async for line in resp.content:
                line_str = line.decode('utf-8').strip()
                if line_str:
                    try:
                        chunk_data = json.loads(line_str)
                        full_text += chunk_data.get("text", "")
                        # Check for cached flag in metadata
                        if chunk_data.get("cached"):
                            cached_detected = True
                    except json.JSONDecodeError:
                        pass
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            result = {
                "response": full_text,
                "cached": cached_detected,
                "streaming": True
            }
        else:
            # Partial response is JSON (IncrementalBufferResponse)
            result = await resp.json()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
    return elapsed, result


async def send_direct_query(session: aiohttp.ClientSession, query: str, session_id: str) -> Tuple[float, Dict]:
    """
    Send a direct query to the regular /api/v1/query endpoint (baseline - no pre-accumulation).
    """
    start = time.perf_counter()
    
    payload = {
        "query": query,
        "context": {
            "language": "te-mixed",
            "session_id": session_id
        }
    }
    
    async with session.post(QUERY_ENDPOINT, json=payload) as resp:
        result = await resp.json()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        return elapsed, result

async def test_with_pre_accumulation(session: aiohttp.ClientSession, query_set: Dict = None) -> Dict:
    """
    Test WITH Pre-LLM Accumulation using /api/v1/query/incremental endpoint.
    
    This endpoint:
    - is_final=False: Does Pre-LLM work (pattern detection, doc retrieval, prompt building)
    - is_final=True: Uses pre-built prompt (LLM-only, fast path)
    """
    session_id = f"prellm-test-{uuid.uuid4().hex[:8]}"
    
    # Use provided query set or default
    if query_set is None:
        query_set = TEST_QUERIES["prellm"]
    
    # Simulate realistic speech chunks (like STT would send)
    partial_chunks = query_set["partials"]
    final_query = query_set["final"]
    
    results = {
        "test_type": "WITH_PRE_ACCUMULATION (incremental endpoint)",
        "session_id": session_id,
        "partial_times_ms": [],
        "partial_statuses": [],
        "final_time_ms": 0,
        "final_response": None,
        "total_partial_time_ms": 0,
    }
    
    print(f"\n{'='*60}")
    print("TEST: WITH Pre-LLM Accumulation (/api/v1/query/incremental)")
    print(f"Session: {session_id}")
    print(f"{'='*60}")
    
    # Send partial chunks (is_final=false) - this is when pre-LLM work happens
    print("\n📋 Sending PARTIAL chunks (is_final=false) to /incremental...")
    for i, chunk in enumerate(partial_chunks):
        elapsed, resp = await send_incremental_chunk(session, chunk, session_id, is_final=False)
        results["partial_times_ms"].append(elapsed)
        status = resp.get("status", "unknown")
        docs = resp.get("docs_retrieved", 0)
        results["partial_statuses"].append(status)
        print(f"  Chunk {i+1}: '{chunk[:20]}...' → {elapsed:.1f}ms (status: {status}, docs: {docs})")
        
        # Small delay between chunks (simulating speech timing)
        await asyncio.sleep(0.3)  # 300ms between chunks - allows pre-LLM processing to complete
    
    results["total_partial_time_ms"] = sum(results["partial_times_ms"])
    
    # Wait a bit more to ensure pre-LLM processing completes
    print("\n⏳ Waiting 500ms for pre-LLM processing to complete...")
    await asyncio.sleep(0.5)
    
    # Send FINAL chunk (is_final=true) - THIS is what we measure for user latency
    print(f"\n🚀 Sending FINAL chunk (is_final=true)...")
    print(f"   Query: '{final_query}'")
    
    final_elapsed, final_resp = await send_incremental_chunk(session, final_query, session_id, is_final=True)
    results["final_time_ms"] = final_elapsed
    results["final_response"] = {
        "cached": final_resp.get("cached", False),
        "streaming": final_resp.get("streaming", False),
        "response_preview": final_resp.get("response", "")[:100] + "..." if final_resp.get("response") else "No response"
    }
    
    print(f"   → FINAL Response Time: {final_elapsed:.1f}ms")
    print(f"   → Used Pre-Built Prompt (Fast Path): {final_resp.get('cached', 'unknown')}")
    print(f"   → Streaming Response: {final_resp.get('streaming', False)}")
    
    return results

async def test_without_pre_accumulation(session: aiohttp.ClientSession, query: str = None) -> Dict:
    """
    Test WITHOUT Pre-LLM Accumulation (Baseline)
    Sends full query directly to /api/v1/query without any pre-processing
    """
    session_id = f"baseline-test-{uuid.uuid4().hex[:8]}"
    
    # Use provided query or default (DIFFERENT from pre-accumulation to avoid cache)
    if query is None:
        query = TEST_QUERIES["baseline"]["final"]
    final_query = query
    
    results = {
        "test_type": "WITHOUT_PRE_ACCUMULATION (BASELINE - /api/v1/query)",
        "session_id": session_id,
        "final_time_ms": 0,
        "final_response": None,
    }
    
    print(f"\n{'='*60}")
    print("TEST: WITHOUT Pre-LLM Accumulation (BASELINE - /api/v1/query)")
    print(f"Session: {session_id}")
    print(f"{'='*60}")
    
    # Send ONLY the final query - no pre-processing
    print(f"\n🚀 Sending DIRECT query to /api/v1/query (no pre-accumulation)...")
    print(f"   Query: '{final_query}'")
    
    final_elapsed, final_resp = await send_direct_query(session, final_query, session_id)
    results["final_time_ms"] = final_elapsed
    results["final_response"] = {
        "cached": final_resp.get("cached", False),
        "answer_preview": final_resp.get("answer", "")[:100] + "..." if final_resp.get("answer") else "No answer"
    }
    
    print(f"   → Response Time: {final_elapsed:.1f}ms")
    print(f"   → Query Cached: {final_resp.get('cached', False)}")
    
    return results

async def run_comparison_test():
    """Run both tests and compare latency."""
    
    print("\n" + "🔬 "*20)
    print("PRE-LLM ACCUMULATION LATENCY TEST")
    print("🔬 "*20)
    print("\nMeasuring TRUE latency improvement from Pre-LLM Accumulation")
    print("Key Metric: Time from FINAL chunk → LLM Response")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: WITH Pre-LLM Accumulation
        with_prellm = await test_with_pre_accumulation(session)
        
        # Wait between tests to avoid rate limiting
        print("\n⏳ Waiting 2 seconds before baseline test...")
        await asyncio.sleep(2)
        
        # Test 2: WITHOUT Pre-LLM Accumulation (Baseline)
        without_prellm = await test_without_pre_accumulation(session)
    
    # Calculate and display results
    print("\n" + "📊 "*20)
    print("RESULTS SUMMARY")
    print("📊 "*20)
    
    prellm_final = with_prellm["final_time_ms"]
    baseline_final = without_prellm["final_time_ms"]
    
    # Improvement calculation
    if baseline_final > 0:
        improvement_ms = baseline_final - prellm_final
        improvement_pct = (improvement_ms / baseline_final) * 100
    else:
        improvement_ms = 0
        improvement_pct = 0
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│                   LATENCY COMPARISON                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WITH Pre-LLM Accumulation:                                 │
│    - Partial chunks total: {with_prellm['total_partial_time_ms']:.1f}ms (done during speech)│
│    - FINAL chunk → Response: {prellm_final:.1f}ms ⭐             │
│    - Used cached context: {with_prellm['final_response']['cached']}                        │
│                                                             │
│  WITHOUT Pre-LLM (BASELINE):                                │
│    - FINAL chunk → Response: {baseline_final:.1f}ms                │
│    - (All processing done at once)                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎯 IMPROVEMENT:                                            │
│    - Time Saved: {improvement_ms:.1f}ms                               │
│    - Percentage: {improvement_pct:.1f}% faster                        │
│                                                             │
│  📝 NOTE: Partial chunk processing ({with_prellm['total_partial_time_ms']:.1f}ms)      │
│           happens DURING speech, not after.                 │
│           User only waits {prellm_final:.1f}ms after speech ends.        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")
    
    # Additional test: Multiple runs for consistency with DIFFERENT queries
    print("\n🔄 Running 3 more iterations with DIFFERENT queries (no cache hits)...")
    
    prellm_times = [prellm_final]
    baseline_times = [baseline_final]
    
    async with aiohttp.ClientSession() as session:
        for i in range(3):
            await asyncio.sleep(1)
            
            # With Pre-LLM - use different query each time (incremental endpoint)
            query_set = TEST_QUERIES["iteration_prellm"][i]
            sid = f"prellm-iter{i}-{uuid.uuid4().hex[:4]}"
            print(f"\n  Iteration {i+1} Pre-LLM (incremental): '{query_set['final'][:30]}...'")
            for chunk in query_set["partials"]:
                await send_incremental_chunk(session, chunk, sid, is_final=False)
                await asyncio.sleep(0.2)
            # Wait for pre-LLM processing
            await asyncio.sleep(0.5)
            elapsed, resp = await send_incremental_chunk(session, query_set["final"], sid, is_final=True)
            prellm_times.append(elapsed)
            print(f"    → Final: {elapsed:.1f}ms (pre-built: {resp.get('cached', 'unknown')})")
            
            await asyncio.sleep(1)
            
            # Baseline - use different query each time (regular query endpoint)
            baseline_query = TEST_QUERIES["iteration_baseline"][i]
            sid = f"base-iter{i}-{uuid.uuid4().hex[:4]}"
            print(f"  Iteration {i+1} Baseline (direct query): '{baseline_query[:30]}...'")
            elapsed, resp = await send_direct_query(session, baseline_query, sid)
            baseline_times.append(elapsed)
            print(f"    → Response: {elapsed:.1f}ms (query cached: {resp.get('cached', False)})")
    
    avg_prellm = sum(prellm_times) / len(prellm_times)
    avg_baseline = sum(baseline_times) / len(baseline_times)
    avg_improvement = avg_baseline - avg_prellm
    avg_improvement_pct = (avg_improvement / avg_baseline) * 100 if avg_baseline > 0 else 0
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│              AVERAGE OVER {len(prellm_times)} ITERATIONS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  With Pre-LLM (Final Time):    {avg_prellm:.1f}ms avg                │
│  Without Pre-LLM (Baseline):   {avg_baseline:.1f}ms avg               │
│                                                             │
│  🎯 AVERAGE IMPROVEMENT: {avg_improvement:.1f}ms ({avg_improvement_pct:.1f}%)           │
│                                                             │
│  Individual runs:                                           │
│    Pre-LLM: {', '.join([f'{t:.0f}' for t in prellm_times])}ms               │
│    Baseline: {', '.join([f'{t:.0f}' for t in baseline_times])}ms              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")
    
    return {
        "with_prellm": {
            "avg_final_ms": avg_prellm,
            "all_runs": prellm_times
        },
        "baseline": {
            "avg_final_ms": avg_baseline,
            "all_runs": baseline_times
        },
        "improvement": {
            "ms": avg_improvement,
            "percent": avg_improvement_pct
        }
    }

if __name__ == "__main__":
    asyncio.run(run_comparison_test())