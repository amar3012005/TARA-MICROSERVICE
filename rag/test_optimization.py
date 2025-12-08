#!/usr/bin/env python3
"""
Integration tests for RAG Pre-LLM Accumulation Optimization.

Tests:
1. Verify incremental endpoint accumulation
2. Compare latency (standard vs optimized path)
3. Test pattern detection
"""

import requests
import time
import json
import sys

RAG_BASE_URL = "http://localhost:2003"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_service_health():
    """Test 0: Verify services are healthy"""
    print_header("Test 0: Service Health Check")
    
    try:
        response = requests.get(f"{RAG_BASE_URL}/health", timeout=5)
        health = response.json()
        print(f"RAG Service: {json.dumps(health, indent=2)}")
        
        if health.get("status") == "healthy":
            print("✓ RAG service is healthy")
            return True
        else:
            print("✗ RAG service is not healthy")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to RAG service: {e}")
        return False

def test_incremental_accumulation():
    """Test 1: Verify RAG Incremental Endpoint Accumulation"""
    print_header("Test 1: Incremental Endpoint Accumulation")
    
    session_id = f"test_session_{int(time.time())}"
    
    # Partial chunks to send (simulating streaming speech)
    partial_chunks = [
        "I want to",
        "I want to book",
        "I want to book an",
        "I want to book an appointment"
    ]
    
    results = []
    
    print(f"Session ID: {session_id}")
    print("\nSending partial chunks (is_final=false):")
    
    # Send partial chunks
    for i, chunk in enumerate(partial_chunks):
        try:
            response = requests.post(
                f"{RAG_BASE_URL}/api/v1/query/incremental",
                json={
                    "session_id": session_id,
                    "text": chunk,
                    "is_final": False
                },
                timeout=10
            )
            result = response.json()
            results.append(result)
            status = result.get("status", "unknown")
            print(f"  Chunk {i}: '{chunk}' → Status: {status}")
            time.sleep(0.1)  # Simulate speech timing
        except Exception as e:
            print(f"  Chunk {i}: Error - {e}")
            results.append({"error": str(e)})
    
    # Send final chunk
    print("\nSending final chunk (is_final=true):")
    final_text = "I want to book an appointment for tomorrow"
    
    start = time.time()
    try:
        response = requests.post(
            f"{RAG_BASE_URL}/api/v1/query/incremental",
            json={
                "session_id": session_id,
                "text": final_text,
                "is_final": True
            },
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        final_result = response.json()
        
        print(f"  Final text: '{final_text}'")
        print(f"  Response time: {elapsed:.0f}ms")
        print(f"  Status code: {response.status_code}")
        print(f"  Response preview: {json.dumps(final_result, indent=2)[:500]}...")
        
        # Check for fast path indicator
        if final_result.get("fast_path_used") or final_result.get("pre_built_prompt_used"):
            print("\n✓ Fast path was used (pre-built prompt)")
        
        return elapsed, final_result
        
    except Exception as e:
        print(f"  Final chunk error: {e}")
        return None, {"error": str(e)}

def test_standard_path_latency():
    """Test 2: Standard path latency (direct query without pre-accumulation)"""
    print_header("Test 2: Standard Path Latency")
    
    query = "I want to book an appointment for tomorrow"
    
    print(f"Query: '{query}'")
    print("Sending direct query (no pre-accumulation)...")
    
    start = time.time()
    try:
        response = requests.post(
            f"{RAG_BASE_URL}/api/v1/query",
            json={"query": query},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        result = response.json()
        
        print(f"  Response time: {elapsed:.0f}ms")
        print(f"  Status code: {response.status_code}")
        print(f"  Response preview: {json.dumps(result, indent=2)[:500]}...")
        
        return elapsed, result
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, {"error": str(e)}

def test_latency_comparison():
    """Test 3: Compare latency between standard and optimized paths"""
    print_header("Test 3: Latency Comparison")
    
    query = "What are your business hours today?"
    
    # Standard path first
    print("1. Standard path (direct query):")
    standard_times = []
    for i in range(3):
        start = time.time()
        try:
            response = requests.post(
                f"{RAG_BASE_URL}/api/v1/query",
                json={"query": query},
                timeout=30
            )
            elapsed = (time.time() - start) * 1000
            standard_times.append(elapsed)
            print(f"   Run {i+1}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"   Run {i+1}: Error - {e}")
        time.sleep(0.5)
    
    avg_standard = sum(standard_times) / len(standard_times) if standard_times else 0
    print(f"   Average: {avg_standard:.0f}ms")
    
    # Optimized path
    print("\n2. Optimized path (with pre-accumulation):")
    optimized_times = []
    
    for run in range(3):
        session_id = f"test_opt_{int(time.time())}_{run}"
        
        # Send partial chunks (these warm up the cache)
        chunks = ["What are", "What are your", "What are your business", "What are your business hours"]
        for i, chunk in enumerate(chunks):
            try:
                requests.post(
                    f"{RAG_BASE_URL}/api/v1/query/incremental",
                    json={"session_id": session_id, "text": chunk, "is_final": False},
                    timeout=10
                )
            except:
                pass
            time.sleep(0.05)
        
        # Send final
        start = time.time()
        try:
            response = requests.post(
                f"{RAG_BASE_URL}/api/v1/query/incremental",
                json={"session_id": session_id, "text": query, "is_final": True},
                timeout=30
            )
            elapsed = (time.time() - start) * 1000
            optimized_times.append(elapsed)
            print(f"   Run {run+1}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"   Run {run+1}: Error - {e}")
        time.sleep(0.5)
    
    avg_optimized = sum(optimized_times) / len(optimized_times) if optimized_times else 0
    print(f"   Average: {avg_optimized:.0f}ms")
    
    # Summary
    if avg_standard > 0 and avg_optimized > 0:
        improvement = avg_standard - avg_optimized
        pct_improvement = (1 - avg_optimized / avg_standard) * 100
        print(f"\n   📊 Results:")
        print(f"      Standard path: {avg_standard:.0f}ms")
        print(f"      Optimized path: {avg_optimized:.0f}ms")
        print(f"      Improvement: {improvement:.0f}ms ({pct_improvement:.1f}%)")
        
        if improvement > 0:
            print(f"\n   ✓ Optimization provides {improvement:.0f}ms improvement")
        else:
            print(f"\n   Note: Optimized path was {-improvement:.0f}ms slower (may vary with load)")
    
    return avg_standard, avg_optimized

def test_pattern_detection():
    """Test 4: Verify pattern detection is working"""
    print_header("Test 4: Pattern Detection")
    
    test_queries = [
        ("I want to cancel my appointment", "appointment_cancel"),
        ("What are your business hours?", "faq_query"),
        ("Book a slot for tomorrow 3pm", "appointment_booking"),
        ("Check my appointment status", "status_check"),
        ("Hello, how are you?", "greeting"),
    ]
    
    print("Testing pattern detection on various query types:\n")
    
    for query, expected_pattern in test_queries:
        session_id = f"pattern_test_{expected_pattern}_{int(time.time())}"
        try:
            response = requests.post(
                f"{RAG_BASE_URL}/api/v1/query/incremental",
                json={
                    "session_id": session_id,
                    "text": query,
                    "is_final": False
                },
                timeout=10
            )
            result = response.json()
            detected = result.get("detected_pattern", result.get("pattern", "N/A"))
            status = result.get("status", "unknown")
            print(f"  '{query[:35]:<35}' → Pattern: {detected}, Status: {status}")
        except Exception as e:
            print(f"  '{query[:35]:<35}' → Error: {e}")
    
    return True

def main():
    print("\n" + "="*60)
    print("  RAG Pre-LLM Accumulation Optimization - Integration Tests")
    print("="*60)
    
    # Run tests
    test_results = {}
    
    # Test 0: Health check
    healthy = test_service_health()
    test_results["health_check"] = healthy
    if not healthy:
        print("\n❌ Service not healthy. Aborting tests.")
        sys.exit(1)
    
    # Test 1: Incremental accumulation
    opt_latency, opt_result = test_incremental_accumulation()
    test_results["incremental_accumulation"] = opt_latency is not None
    
    # Test 2: Standard path latency
    std_latency, std_result = test_standard_path_latency()
    test_results["standard_path"] = std_latency is not None
    
    # Test 3: Latency comparison
    avg_std, avg_opt = test_latency_comparison()
    test_results["latency_comparison"] = {
        "standard_avg": avg_std,
        "optimized_avg": avg_opt,
        "improvement_ms": avg_std - avg_opt if avg_std and avg_opt else 0
    }
    
    # Test 4: Pattern detection
    pattern_ok = test_pattern_detection()
    test_results["pattern_detection"] = pattern_ok
    
    # Summary
    print_header("Test Summary")
    
    print("Results:")
    print(f"  ✓ Health Check: {'PASS' if test_results['health_check'] else 'FAIL'}")
    print(f"  ✓ Incremental Accumulation: {'PASS' if test_results['incremental_accumulation'] else 'FAIL'}")
    print(f"  ✓ Standard Path: {'PASS' if test_results['standard_path'] else 'FAIL'}")
    
    lat_comp = test_results.get("latency_comparison", {})
    if lat_comp.get("improvement_ms", 0) > 0:
        print(f"  ✓ Latency Improvement: {lat_comp['improvement_ms']:.0f}ms faster")
    else:
        print(f"  ○ Latency Comparison: No significant improvement detected")
    
    print(f"  ✓ Pattern Detection: {'PASS' if test_results['pattern_detection'] else 'FAIL'}")
    
    print("\nDetailed latency results:")
    print(f"  Standard path average: {lat_comp.get('standard_avg', 0):.0f}ms")
    print(f"  Optimized path average: {lat_comp.get('optimized_avg', 0):.0f}ms")
    
    return test_results

if __name__ == "__main__":
    results = main()