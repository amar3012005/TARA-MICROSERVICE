# RAG Service - curl Test Commands

## Quick Test (Complete Response)

```bash
curl -w "\n\n⏱️  Time to First Byte: %{time_starttransfer}s\n⏱️  Total Time: %{time_total}s\n" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is task",
    "context": {
      "language": "te-mixed",
      "organization": "T.A.S.K"
    },
    "enable_streaming": true
  }' \
  http://localhost:2003/api/v1/stream_query
```

## Test with Custom Query

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question here",
    "context": {
      "language": "te-mixed",
      "organization": "T.A.S.K"
    },
    "enable_streaming": true
  }' \
  http://localhost:2003/api/v1/stream_query
```

## Test Scripts

- `./test_first_chunk.sh` - Shows first chunk + complete response + timing
- `./test_cold_query.sh` - Tests with unique query (no cache) + complete response
- `./test_full_response.sh` - Shows complete streaming response with all chunks formatted

## Response Format

The streaming endpoint returns NDJSON (newline-delimited JSON) with chunks:

```json
{"text": "First chunk of text", "is_final": false}
{"text": "Second chunk", "is_final": false}
{"text": "", "is_final": true}
```

Each line is a JSON object. The complete response is all `text` fields concatenated.




