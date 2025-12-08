# RAG Streaming Cache Fix

## Issue
While the standard `/query` endpoint was successfully caching responses (~18ms), the `/stream_query` endpoint used by the Orchestrator was **bypassing the cache** entirely. This resulted in high latency (~1.2s) for voice interactions, even for common queries like "contact information".

## Fix Implementation
Modified `rag/app.py` to implement caching for the streaming endpoint:
1.  **Cache Lookup**: Checks Redis for the query key before processing.
2.  **Simulated Streaming**: If a cache hit occurs, the full cached response is yielded in small chunks to mimic the streaming behavior expected by the client.
3.  **Cache Write**: If a cache miss occurs, the streamed response is accumulated and saved to Redis for future requests.

## Verification
**Before Fix:**
- Latency: ~1.2s
- Cache Hit: No

**After Fix:**
- Latency: **~0.088s** (88ms)
- Cache Hit: Yes (verified via logs and timing)

## Commands to Verify
```bash
# 1. First request (Miss + Cache Write)
curl -X POST "http://localhost:2003/api/v1/stream_query" \
     -H "Content-Type: application/json" \
     -d '{"query": "contact information", "context": {"language": "te"}}'

# 2. Second request (Hit + Instant Stream)
time curl -X POST "http://localhost:2003/api/v1/stream_query" \
     -H "Content-Type: application/json" \
     -d '{"query": "contact information", "context": {"language": "te"}}'
```
