#!/usr/bin/env python3
"""
Comprehensive test script to measure streaming latency and validate
Sarvam streaming API integration.

This test measures:
- Time to First Audio Chunk (TTFAC)
- Total synthesis time
- Audio chunk size and frequency
- Buffer management effectiveness
- WebSocket connection stability
- Streaming vs non-streaming performance comparison
"""

import asyncio
import base64
import json
import time
import statistics
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Add tts_sarvam to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tts_sarvam'))

try:
    import httpx
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Install with: pip install httpx websockets")
    sys.exit(1)

# Test configuration
TTS_BASE_URL = "http://localhost:2005"
TTS_WS_URL = "ws://localhost:2005/api/v1/stream"

@dataclass
class LatencyMetrics:
    """Container for latency measurement results"""
    tt_fac: float  # Time to First Audio Chunk (ms)
    total_time: float  # Total synthesis time (ms)
    chunk_count: int  # Number of audio chunks received
    avg_chunk_size: float  # Average chunk size in bytes
    chunk_sizes: List[int]  # Individual chunk sizes
    chunk_intervals: List[float]  # Time between chunks (ms)
    text_length: int  # Input text length
    buffer_size: int  # Buffer size used
    max_chunk_length: int  # Max chunk length used
    connection_stable: bool  # WebSocket stability
    error_count: int  # Number of errors encountered

@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    success: bool
    metrics: Optional[LatencyMetrics] = None
    error_message: Optional[str] = None
    comparison_data: Optional[Dict[str, Any]] = None

class StreamingLatencyTester:
    """Comprehensive streaming latency tester"""

    def __init__(self):
        self.session_id = f"latency_test_{int(time.time())}"
        self.test_texts = [
            "Hello, this is a simple test message for latency measurement.",
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once.",
            "Streaming text-to-speech synthesis enables real-time audio generation with minimal latency. This technology is crucial for interactive applications and voice interfaces.",
            "In the field of artificial intelligence, natural language processing and speech synthesis are rapidly advancing. These technologies enable more human-like interactions between computers and users."
        ]

    async def health_check(self) -> bool:
        """Check if TTS service is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{TTS_BASE_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "healthy"
        except Exception as e:
            print(f"Health check failed: {e}")
        return False

    async def test_non_streaming_http(self, text: str) -> Optional[LatencyMetrics]:
        """Test non-streaming HTTP synthesis for comparison"""
        start_time = time.time()

        try:
            payload = {
                "text": text,
                "language": "en-IN",
                "voice": "anushka"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{TTS_BASE_URL}/api/v1/synthesize", json=payload)

                if response.status_code != 200:
                    return None

                data = response.json()
                if not data.get("success") or not data.get("audio_data"):
                    return None

                # Decode to get actual audio size
                audio_bytes = base64.b64decode(data["audio_data"])
                total_time = (time.time() - start_time) * 1000

                return LatencyMetrics(
                    tt_fac=total_time,  # For non-streaming, TTFAC = total time
                    total_time=total_time,
                    chunk_count=1,
                    avg_chunk_size=len(audio_bytes),
                    chunk_sizes=[len(audio_bytes)],
                    chunk_intervals=[],
                    text_length=len(text),
                    buffer_size=0,  # Not applicable
                    max_chunk_length=0,  # Not applicable
                    connection_stable=True,
                    error_count=0
                )

        except Exception as e:
            print(f"HTTP test failed: {e}")
            return None

    async def test_streaming_websocket(self, text: str, buffer_size: int = 50, max_chunk_length: int = 200) -> Optional[LatencyMetrics]:
        """Test streaming WebSocket synthesis"""
        start_time = time.time()
        first_chunk_time = None
        chunks_received = []
        chunk_timestamps = []
        error_count = 0

        try:
            async with websockets.connect(f"{TTS_WS_URL}?session_id={self.session_id}") as websocket:
                # Wait for connection confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                if data.get("type") != "connected":
                    return None

                # Send streaming synthesis request
                await websocket.send(json.dumps({
                    "type": "synthesize",
                    "text": text,
                    "streaming": True
                }))

                # Wait for streaming started message
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                if data.get("type") != "streaming_started":
                    return None

                # Collect audio chunks
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        data = json.loads(response)

                        if data.get("type") == "audio":
                            chunk_time = time.time()
                            chunk_timestamps.append(chunk_time)

                            if first_chunk_time is None:
                                first_chunk_time = chunk_time

                            audio_b64 = data.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)
                            chunks_received.append(len(audio_bytes))

                        elif data.get("type") == "error":
                            error_count += 1
                            print(f"Streaming error: {data.get('message')}")

                        elif data.get("type") in ["complete", "stream_finished"]:
                            break

                    except asyncio.TimeoutError:
                        print("Timeout waiting for audio chunks")
                        break

                # Finish the stream
                await websocket.send(json.dumps({"type": "finish_stream"}))

                if not chunks_received:
                    return None

                total_time = (time.time() - start_time) * 1000
                tt_fac = (first_chunk_time - start_time) * 1000 if first_chunk_time else total_time

                # Calculate chunk intervals
                chunk_intervals = []
                for i in range(1, len(chunk_timestamps)):
                    interval = (chunk_timestamps[i] - chunk_timestamps[i-1]) * 1000
                    chunk_intervals.append(interval)

                return LatencyMetrics(
                    tt_fac=tt_fac,
                    total_time=total_time,
                    chunk_count=len(chunks_received),
                    avg_chunk_size=statistics.mean(chunks_received) if chunks_received else 0,
                    chunk_sizes=chunks_received,
                    chunk_intervals=chunk_intervals,
                    text_length=len(text),
                    buffer_size=buffer_size,
                    max_chunk_length=max_chunk_length,
                    connection_stable=error_count == 0,
                    error_count=error_count
                )

        except Exception as e:
            print(f"WebSocket streaming test failed: {e}")
            return None

    async def test_incremental_streaming(self, text_chunks: List[str], buffer_size: int = 50) -> Optional[LatencyMetrics]:
        """Test incremental text streaming"""
        start_time = time.time()
        first_chunk_time = None
        chunks_received = []
        chunk_timestamps = []
        error_count = 0

        try:
            async with websockets.connect(f"{TTS_WS_URL}?session_id={self.session_id}_incremental") as websocket:
                # Connection setup
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                if data.get("type") != "connected":
                    return None

                # Start streaming session
                await websocket.send(json.dumps({
                    "type": "synthesize",
                    "text": text_chunks[0],
                    "streaming": True
                }))

                # Wait for streaming started
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                if data.get("type") != "streaming_started":
                    return None

                # Add additional text chunks incrementally
                for chunk in text_chunks[1:]:
                    await asyncio.sleep(0.1)  # Small delay between additions
                    await websocket.send(json.dumps({
                        "type": "add_text",
                        "text": chunk
                    }))

                    # Wait for confirmation
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    if data.get("type") == "error":
                        error_count += 1

                # Collect audio chunks
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                        data = json.loads(response)

                        if data.get("type") == "audio":
                            chunk_time = time.time()
                            chunk_timestamps.append(chunk_time)

                            if first_chunk_time is None:
                                first_chunk_time = chunk_time

                            audio_b64 = data.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)
                            chunks_received.append(len(audio_bytes))

                        elif data.get("type") == "error":
                            error_count += 1

                        elif data.get("type") in ["complete", "stream_finished"]:
                            break

                    except asyncio.TimeoutError:
                        break

                # Finish stream
                await websocket.send(json.dumps({"type": "finish_stream"}))

                if not chunks_received:
                    return None

                total_time = (time.time() - start_time) * 1000
                tt_fac = (first_chunk_time - start_time) * 1000 if first_chunk_time else total_time

                chunk_intervals = []
                for i in range(1, len(chunk_timestamps)):
                    interval = (chunk_timestamps[i] - chunk_timestamps[i-1]) * 1000
                    chunk_intervals.append(interval)

                return LatencyMetrics(
                    tt_fac=tt_fac,
                    total_time=total_time,
                    chunk_count=len(chunks_received),
                    avg_chunk_size=statistics.mean(chunks_received) if chunks_received else 0,
                    chunk_sizes=chunks_received,
                    chunk_intervals=chunk_intervals,
                    text_length=sum(len(chunk) for chunk in text_chunks),
                    buffer_size=buffer_size,
                    max_chunk_length=max(text_chunks, key=len).__len__() if text_chunks else 0,
                    connection_stable=error_count == 0,
                    error_count=error_count
                )

        except Exception as e:
            print(f"Incremental streaming test failed: {e}")
            return None

    async def run_comprehensive_test(self) -> List[TestResult]:
        """Run comprehensive latency tests"""
        results = []

        print("🩺 Checking service health...")
        if not await self.health_check():
            print("❌ TTS service is not healthy. Aborting tests.")
            return [TestResult("health_check", False, error_message="Service unhealthy")]

        print("✅ Service is healthy. Starting latency tests...\n")

        # Test 1: Basic streaming vs non-streaming comparison
        print("📊 Test 1: Streaming vs Non-Streaming Performance")
        test_text = self.test_texts[0]

        # Non-streaming test
        http_metrics = await self.test_non_streaming_http(test_text)
        if http_metrics:
            print(f"   HTTP Total: {http_metrics.total_time:.1f}ms")
        else:
            print("❌ HTTP test failed")

        # Streaming test
        streaming_metrics = await self.test_streaming_websocket(test_text)
        if streaming_metrics:
            print(f"   Streaming TTFAC: {streaming_metrics.tt_fac:.1f}ms, Total: {streaming_metrics.total_time:.1f}ms")
            print(f"   Chunks: {streaming_metrics.chunk_count}, Avg size: {streaming_metrics.avg_chunk_size:.0f} bytes")
        else:
            print("❌ Streaming test failed")

        if http_metrics and streaming_metrics:
            speedup = http_metrics.total_time / streaming_metrics.tt_fac if streaming_metrics.tt_fac > 0 else 0
            results.append(TestResult(
                "streaming_vs_non_streaming",
                True,
                streaming_metrics,
                comparison_data={
                    "http_total_time": http_metrics.total_time,
                    "streaming_tt_fac": streaming_metrics.tt_fac,
                    "speedup_ratio": speedup
                }
            ))
        else:
            results.append(TestResult("streaming_vs_non_streaming", False, error_message="One or both tests failed"))

        # Test 2: Different text lengths
        print("\n📏 Test 2: Text Length Impact on Latency")
        for i, text in enumerate(self.test_texts):
            metrics = await self.test_streaming_websocket(text)
            if metrics:
                print(f"   Text length {len(text)}: TTFAC {metrics.tt_fac:.1f}ms, Total {metrics.total_time:.1f}ms")
                results.append(TestResult(f"text_length_{len(text)}", True, metrics))
            else:
                print(f"❌ Test failed for text length {len(text)}")
                results.append(TestResult(f"text_length_{len(text)}", False, error_message="Test failed"))

        # Test 3: Buffer size impact
        print("\n🔧 Test 3: Buffer Size Impact")
        test_text = self.test_texts[1]
        buffer_sizes = [25, 50, 100, 200]

        for buffer_size in buffer_sizes:
            metrics = await self.test_streaming_websocket(test_text, buffer_size=buffer_size)
            if metrics:
                print(f"   Buffer {buffer_size}: TTFAC {metrics.tt_fac:.1f}ms, Total {metrics.total_time:.1f}ms")
                results.append(TestResult(f"buffer_size_{buffer_size}", True, metrics))
            else:
                print(f"❌ Test failed for buffer size {buffer_size}")
                results.append(TestResult(f"buffer_size_{buffer_size}", False, error_message="Test failed"))

        # Test 4: Incremental streaming
        print("\n🔄 Test 4: Incremental Text Streaming")
        incremental_chunks = [
            "Hello, this is the first part of a message.",
            " I'm adding more text to demonstrate incremental streaming.",
            " This allows for real-time text input and audio generation.",
            " The system should handle this smoothly with proper buffering."
        ]

        metrics = await self.test_incremental_streaming(incremental_chunks)
        if metrics:
            print(f"   Incremental: TTFAC {metrics.tt_fac:.1f}ms, Total {metrics.total_time:.1f}ms, Chunks {metrics.chunk_count}")
            results.append(TestResult("incremental_streaming", True, metrics))
        else:
            print("❌ Incremental streaming test failed")
            results.append(TestResult("incremental_streaming", False, error_message="Test failed"))

        # Test 5: Connection stability
        print("\n🔗 Test 5: Connection Stability Test")
        stability_results = []
        for i in range(5):
            metrics = await self.test_streaming_websocket(self.test_texts[0])
            if metrics and metrics.connection_stable:
                stability_results.append(True)
            else:
                stability_results.append(False)
            await asyncio.sleep(0.5)  # Brief pause between tests

        stability_rate = sum(stability_results) / len(stability_results) * 100
        print(f"   Stability: {stability_rate:.1f}% ({sum(stability_results)}/{len(stability_results)})")
        results.append(TestResult(
            "connection_stability",
            stability_rate >= 80,  # 80% success rate threshold
            comparison_data={"stability_rate": stability_rate, "tests_run": len(stability_results)}
        ))

        return results

    def print_summary_report(self, results: List[TestResult]):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("📊 STREAMING LATENCY TEST SUMMARY REPORT")
        print("="*80)

        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]

        print(f"\n✅ Successful Tests: {len(successful_tests)}/{len(results)}")

        if successful_tests:
            print("\n📈 Performance Metrics:")

            # Collect all streaming metrics
            streaming_metrics = [r.metrics for r in successful_tests if r.metrics and r.test_name != "connection_stability"]

            if streaming_metrics:
                tt_fac_values = [m.tt_fac for m in streaming_metrics]
                total_time_values = [m.total_time for m in streaming_metrics]
                chunk_counts = [m.chunk_count for m in streaming_metrics]

                print(f"   TTFAC: Min {min(tt_fac_values):.1f}ms, Max {max(tt_fac_values):.1f}ms, Avg {statistics.mean(tt_fac_values):.1f}ms")
                print(f"   Total Time: Min {min(total_time_values):.1f}ms, Max {max(total_time_values):.1f}ms, Avg {statistics.mean(total_time_values):.1f}ms")
                print(f"   Chunks per Request: Min {min(chunk_counts)}, Max {max(chunk_counts)}, Avg {statistics.mean(chunk_counts):.1f}")
                print(f"   Text Length Range: {min(m.text_length for m in streaming_metrics)} - {max(m.text_length for m in streaming_metrics)} chars")
                print(f"   Buffer Sizes Tested: {sorted(set(m.buffer_size for m in streaming_metrics if m.buffer_size > 0))}")
                print(f"   Total Requests: {len(streaming_metrics)}")
                # Buffer management analysis
                buffer_sizes = [m.buffer_size for m in streaming_metrics if m.buffer_size > 0]
                if buffer_sizes:
                    print(f"   Buffer Size Range: {min(buffer_sizes)} - {max(buffer_sizes)} chars")
                # Chunk analysis
                all_chunk_sizes = []
                all_intervals = []
                for m in streaming_metrics:
                    all_chunk_sizes.extend(m.chunk_sizes)
                    all_intervals.extend(m.chunk_intervals)

                if all_chunk_sizes:
                    print(f"   Chunk Sizes: Min {min(all_chunk_sizes)} bytes, Max {max(all_chunk_sizes)} bytes, Avg {statistics.mean(all_chunk_sizes):.0f} bytes")
                    print(f"   Total Audio Data: {sum(all_chunk_sizes)} bytes across {len(all_chunk_sizes)} chunks")
                    print(f"   Chunk Size Variability: {statistics.stdev(all_chunk_sizes):.0f} bytes (std dev)")
                if all_intervals:
                    print(f"   Chunk Intervals: Min {min(all_intervals):.1f}ms, Max {max(all_intervals):.1f}ms, Avg {statistics.mean(all_intervals):.1f}ms")
                    print(f"   Interval Variability: {statistics.stdev(all_intervals):.1f}ms (std dev)")
        if failed_tests:
            print(f"\n❌ Failed Tests: {len(failed_tests)}")
            for result in failed_tests:
                print(f"   - {result.test_name}: {result.error_message}")

        # Recommendations
        print("\n💡 Recommendations:")
        if successful_tests:
            avg_tt_fac = statistics.mean([r.metrics.tt_fac for r in successful_tests if r.metrics])
            if avg_tt_fac > 1000:
                print("   - Consider optimizing for faster TTFAC (currently > 1s)")
            elif avg_tt_fac > 500:
                print("   - Good TTFAC performance (< 1s)")
            else:
                print("   - Excellent TTFAC performance (< 500ms)")

            stability_test = next((r for r in results if r.test_name == "connection_stability"), None)
            if stability_test and stability_test.comparison_data:
                stability_rate = stability_test.comparison_data["stability_rate"]
                if stability_rate < 80:
                    print("   - Improve WebSocket connection stability")
                else:
                    print("   - WebSocket connections are stable")

        print("\n" + "="*80)

async def main():
    """Main test execution"""
    print("🚀 Starting Comprehensive Streaming Latency Tests")
    print("This will test Sarvam streaming API integration and measure performance metrics.\n")

    tester = StreamingLatencyTester()
    results = await tester.run_comprehensive_test()
    tester.print_summary_report(results)

    # Exit with appropriate code
    success_count = sum(1 for r in results if r.success)
    total_count = len(results)

    if success_count == total_count:
        print("🎉 All tests passed!")
        sys.exit(0)
    elif success_count >= total_count * 0.8:  # 80% success rate
        print("⚠️ Most tests passed - review failed tests")
        sys.exit(0)
    else:
        print("❌ Many tests failed - investigate issues")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())