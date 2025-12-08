#!/usr/bin/env python3
"""
TTS Sarvam Optimization Test Suite

Tests the performance improvements made to the TTS Sarvam service:
- Connection warmup latency
- Sequential vs parallel synthesis comparison
- Time-to-first-audio measurement
- WebSocket streaming modes

Usage:
    python test_tts_optimizations.py [--url URL]
"""

import asyncio
import httpx
import websockets
import json
import time
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List

# Configuration
DEFAULT_TTS_URL = "http://localhost:8025"
DEFAULT_WS_URL = "ws://localhost:8025/api/v1/stream"

class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")


def print_result(label: str, value: Any, unit: str = "", success: bool = True):
    """Print result with color"""
    color = Colors.GREEN if success else Colors.RED
    print(f"  {label}: {color}{value}{unit}{Colors.END}")


def print_comparison(label: str, sequential: float, parallel: float):
    """Print comparison with improvement percentage"""
    improvement = ((sequential - parallel) / sequential) * 100 if sequential > 0 else 0
    color = Colors.GREEN if improvement > 0 else Colors.RED
    print(f"  {label}:")
    print(f"    Sequential: {Colors.YELLOW}{sequential:.0f}ms{Colors.END}")
    print(f"    Parallel:   {Colors.CYAN}{parallel:.0f}ms{Colors.END}")
    print(f"    Improvement: {color}{improvement:.1f}%{Colors.END}")


# Test sentences - a mix of short and medium length
TEST_SENTENCES = [
    "Hello, how can I help you today?",
    "Welcome to our customer service.",
    "Please hold on while I check that for you.",
    "Is there anything else I can help you with?",
    "Thank you for calling. Have a great day!"
]

MULTI_SENTENCE_TEXT = " ".join(TEST_SENTENCES)


async def test_health(url: str) -> bool:
    """Test service health"""
    print_header("1. HEALTH CHECK")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                print_result("Status", data.get("status", "unknown"))
                print_result("Provider", data.get("provider", "unknown"))
                print_result("Cache", data.get("cache", "unknown"))
                print_result("Active Sessions", data.get("active_sessions", 0))
                return data.get("status") == "healthy"
            else:
                print_result("Health check failed", f"HTTP {response.status_code}", success=False)
                return False
    except Exception as e:
        print_result("Error", str(e), success=False)
        return False


async def test_warmup_metrics(url: str) -> Dict[str, Any]:
    """Test and analyze warmup behavior"""
    print_header("2. WARMUP / COLD START ANALYSIS")
    
    results = {"cold_start_ms": 0, "warm_ms": 0}
    test_text = "Hello"
    
    # Cold start - first request after potential service restart
    print(f"  Testing cold start (first request)...")
    async with httpx.AsyncClient() as client:
        start = time.time()
        response = await client.post(
            f"{url}/api/v1/synthesize",
            json={"text": test_text, "parallel": False},
            timeout=30.0
        )
        cold_ms = (time.time() - start) * 1000
        results["cold_start_ms"] = cold_ms
        
        if response.status_code == 200:
            print_result("Cold Start", f"{cold_ms:.0f}", "ms")
        else:
            print_result("Cold Start", f"FAILED ({response.status_code})", success=False)
    
    # Warm requests
    print(f"\n  Testing warm requests (subsequent)...")
    warm_times = []
    for i in range(3):
        async with httpx.AsyncClient() as client:
            start = time.time()
            response = await client.post(
                f"{url}/api/v1/synthesize",
                json={"text": test_text, "parallel": False},
                timeout=30.0
            )
            warm_ms = (time.time() - start) * 1000
            warm_times.append(warm_ms)
            print_result(f"Warm Request {i+1}", f"{warm_ms:.0f}", "ms")
    
    avg_warm = sum(warm_times) / len(warm_times)
    results["warm_ms"] = avg_warm
    
    print(f"\n  {Colors.BOLD}Summary:{Colors.END}")
    print_result("Avg Warm Request", f"{avg_warm:.0f}", "ms")
    improvement = ((cold_ms - avg_warm) / cold_ms) * 100 if cold_ms > 0 else 0
    print_result("Warmup Improvement", f"{improvement:.1f}", "%")
    
    return results


async def test_sequential_synthesis(url: str) -> Dict[str, Any]:
    """Test sequential synthesis mode"""
    print_header("3. SEQUENTIAL SYNTHESIS")
    
    results = {"total_ms": 0, "per_sentence_ms": []}
    
    async with httpx.AsyncClient() as client:
        start = time.time()
        response = await client.post(
            f"{url}/api/v1/synthesize",
            json={"text": MULTI_SENTENCE_TEXT, "parallel": False},
            timeout=60.0
        )
        total_ms = (time.time() - start) * 1000
        results["total_ms"] = total_ms
        
        if response.status_code == 200:
            data = response.json()
            print_result("Total Time", f"{total_ms:.0f}", "ms")
            print_result("Sentences", data.get("sentences", 0))
            print_result("Duration", f"{data.get('duration_ms', 0):.0f}", "ms audio")
            print_result("Audio Size", len(data.get("audio_data", "")), " chars (base64)")
        else:
            print_result("Sequential Test", "FAILED", success=False)
    
    return results


async def test_parallel_synthesis(url: str) -> Dict[str, Any]:
    """Test parallel synthesis mode"""
    print_header("4. PARALLEL SYNTHESIS")
    
    results = {"total_ms": 0, "per_sentence_ms": []}
    
    async with httpx.AsyncClient() as client:
        start = time.time()
        response = await client.post(
            f"{url}/api/v1/synthesize",
            json={"text": MULTI_SENTENCE_TEXT, "parallel": True},
            timeout=60.0
        )
        total_ms = (time.time() - start) * 1000
        results["total_ms"] = total_ms
        
        if response.status_code == 200:
            data = response.json()
            print_result("Total Time", f"{total_ms:.0f}", "ms")
            print_result("Sentences", data.get("sentences", 0))
            print_result("Duration", f"{data.get('duration_ms', 0):.0f}", "ms audio")
            print_result("Audio Size", len(data.get("audio_data", "")), " chars (base64)")
        else:
            print_result("Parallel Test", "FAILED", success=False)
    
    return results


async def test_websocket_parallel(ws_url: str) -> Dict[str, Any]:
    """Test WebSocket with parallel mode"""
    print_header("5. WEBSOCKET PARALLEL STREAMING")
    
    results = {
        "total_ms": 0,
        "ttfa_ms": 0,
        "chunks_received": 0
    }
    
    session_id = f"test_session_{int(time.time())}"
    try:
        async with websockets.connect(f"{ws_url}?session_id={session_id}") as ws:
            start = time.time()
            first_audio_time = None
            
            # Send synthesis request with parallel mode
            await ws.send(json.dumps({
                "type": "synthesize",
                "text": MULTI_SENTENCE_TEXT,
                "parallel": True
            }))
            
            # Receive messages
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "audio":
                        results["chunks_received"] += 1
                        if first_audio_time is None:
                            first_audio_time = time.time()
                            results["ttfa_ms"] = (first_audio_time - start) * 1000
                            print_result("Time to First Audio", f"{results['ttfa_ms']:.0f}", "ms")
                    
                    elif msg_type == "complete":
                        results["total_ms"] = (time.time() - start) * 1000
                        print_result("Total Time", f"{results['total_ms']:.0f}", "ms")
                        print_result("Audio Chunks", results["chunks_received"])
                        if "parallel_batches" in data:
                            print_result("Parallel Batches", data["parallel_batches"])
                        if "time_to_first_audio_ms" in data:
                            print_result("Server TTFA", f"{data['time_to_first_audio_ms']:.0f}", "ms")
                        break
                    
                    elif msg_type == "error":
                        print_result("Error", data.get("message", "Unknown"), success=False)
                        break
                        
                except asyncio.TimeoutError:
                    print_result("Timeout", "No response in 30s", success=False)
                    break
                    
    except Exception as e:
        print_result("WebSocket Error", str(e), success=False)
    
    return results


async def test_websocket_sequential(ws_url: str) -> Dict[str, Any]:
    """Test WebSocket with sequential mode"""
    print_header("6. WEBSOCKET SEQUENTIAL STREAMING")
    
    results = {
        "total_ms": 0,
        "ttfa_ms": 0,
        "chunks_received": 0
    }
    
    session_id = f"test_session_{int(time.time())}"
    try:
        async with websockets.connect(f"{ws_url}?session_id={session_id}") as ws:
            start = time.time()
            first_audio_time = None
            
            # Send synthesis request without parallel mode
            await ws.send(json.dumps({
                "type": "synthesize",
                "text": MULTI_SENTENCE_TEXT,
                "parallel": False
            }))
            
            # Receive messages
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "audio":
                        results["chunks_received"] += 1
                        if first_audio_time is None:
                            first_audio_time = time.time()
                            results["ttfa_ms"] = (first_audio_time - start) * 1000
                            print_result("Time to First Audio", f"{results['ttfa_ms']:.0f}", "ms")
                    
                    elif msg_type == "complete":
                        results["total_ms"] = (time.time() - start) * 1000
                        print_result("Total Time", f"{results['total_ms']:.0f}", "ms")
                        print_result("Audio Chunks", results["chunks_received"])
                        break
                    
                    elif msg_type == "error":
                        print_result("Error", data.get("message", "Unknown"), success=False)
                        break
                        
                except asyncio.TimeoutError:
                    print_result("Timeout", "No response in 30s", success=False)
                    break
                    
    except Exception as e:
        print_result("WebSocket Error", str(e), success=False)
    
    return results


async def run_suite(url: str, ws_url: str):
    """Run complete test suite"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           TTS SARVAM OPTIMIZATION TEST SUITE                        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    print(f"  Target: {url}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    if not await test_health(url):
        print(f"\n{Colors.RED}Service not healthy - aborting tests{Colors.END}")
        return
    
    warmup_results = await test_warmup_metrics(url)
    seq_results = await test_sequential_synthesis(url)
    par_results = await test_parallel_synthesis(url)
    ws_par_results = await test_websocket_parallel(ws_url)
    ws_seq_results = await test_websocket_sequential(ws_url)
    
    # Summary
    print_header("SUMMARY - OPTIMIZATION COMPARISON")
    
    print(f"  {Colors.BOLD}HTTP API:{Colors.END}")
    print_comparison("  Synthesis (5 sentences)", seq_results["total_ms"], par_results["total_ms"])
    
    print(f"\n  {Colors.BOLD}WebSocket Streaming:{Colors.END}")
    print_comparison("  Total Time", ws_seq_results["total_ms"], ws_par_results["total_ms"])
    print_comparison("  Time to First Audio", ws_seq_results["ttfa_ms"], ws_par_results["ttfa_ms"])
    
    print(f"\n  {Colors.BOLD}Warmup Effect:{Colors.END}")
    print_comparison("  First vs Warm Request", warmup_results["cold_start_ms"], warmup_results["warm_ms"])
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All tests completed!{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(description="TTS Sarvam Optimization Test Suite")
    parser.add_argument("--url", default=DEFAULT_TTS_URL, help="TTS service base URL")
    parser.add_argument("--ws-url", default=None, help="WebSocket URL (default: derived from --url)")
    args = parser.parse_args()
    
    ws_url = args.ws_url or args.url.replace("http://", "ws://") + "/api/v1/stream"
    
    try:
        asyncio.run(run_suite(args.url, ws_url))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()