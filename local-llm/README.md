# Local LLM Service (rag-llm-service)

This folder bundles everything required to run the Leibniz RAG completion service with a GPU-backed vLLM server. Two compose files are provided:

- `docker-compose.yml` – tuned for **single-user, ultra-low latency** on a local RTX-class GPU.
- `docker-compose.cloud.yml` – ready for **multi-session cloud deployments** (L4 / A10 / A100 class GPUs) with health checks and telemetry hooks.

## 1. Local, single-user workflow

1. Export your Hugging Face token (if the model requires authentication):
   ```bash
   export HUGGING_FACE_HUB_TOKEN=hf_xxx
   ```
2. Start the stack:
   ```bash
   docker compose up -d
   ```
   This launches `rag-llm-service` (Qwen2.5-1.5B-Instruct-AWQ) plus Redis on `network_mode: host`.
3. Health check & sample call:
   ```bash
   curl -s http://localhost:8081/health
   curl -s http://localhost:8081/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct-AWQ","messages":[{"role":"user","content":"Say hello."}]}'
   ```

**Why this config is fast for one user**
- `VLLM_MAX_BATCH_SIZE=4`, `VLLM_MAX_NUM_SEQS=6`, and `max_num_tokens=1024` keep the GPU scheduler from queuing large batches, so the 0.6–7 B model responds in ~500 ms.
- `--quantization awq --kv-cache-dtype fp8` keeps VRAM under ~7 GB on an RTX 4060 while still sounding human.

## 2. Cloud deployment workflow

1. Provision a GPU VM (see “Recommended cloud config” below).
2. Copy this directory to the VM and set the following environment variables as needed:
   - `HUGGING_FACE_HUB_TOKEN` – private HF access token.
   - `MODEL_CACHE_DIR` – persistent path for downloaded weights (default `/var/lib/llm-cache`).
   - `LLM_MODEL_ID` – override to a larger checkpoint (e.g., `Qwen/Qwen2.5-7B-Instruct-AWQ`).
   - `TENSOR_PARALLEL_SIZE` – set to `2` if you attach two GPUs.
3. Launch with the cloud profile:
   ```bash
   docker compose -f docker-compose.cloud.yml up -d
   ```
4. Verify:
   ```bash
   curl -s http://<public-ip>:8081/health
   ```

**Cloud compose highlights**
- Uses bridge networking with `8081:8081` so you can place it behind a load balancer / API gateway.
- Enables higher concurrency (`VLLM_MAX_BATCH_SIZE=32`, `max_num_seqs=32`) and chunked prefill for better GPU utilization.
- Adds a health check and optional OTLP endpoint for OpenTelemetry traces.
- Persists the Hugging Face cache so large checkpoints don’t redownload on each restart.

## 3. Load testing & observability

- Quick latency probe:
  ```bash
  python3 scripts/load_test.py  # adapt from snippets in docs or reuse the inline script from README
  ```
- Watch resource usage:
  ```bash
  docker stats
  nvidia-smi dmon -s pucm
  curl -s http://localhost:8081/metrics | grep vllm
  ```
- For Redis insight:
  ```bash
  docker exec local-llm-redis-1 redis-cli info memory
  ```

## 4. Recommended cloud configuration

| Tier | Suggested Provider / SKU | Notes |
| --- | --- | --- |
| **Primary** | **AWS g6e.xlarge (1× NVIDIA L4 24 GB)** | 24 GB VRAM keeps Qwen2.5-7B-AWQ in memory, supports ~30 concurrent sessions with `docker-compose.cloud.yml`. Attach 200 GB gp3 volume for model cache. |
| Alternative | GCP A2-highgpu-1g (1× A100 40 GB) | More headroom for FP16 models or longer contexts (8–16 k tokens). |
| Budget | Lambda Cloud 1×A10 24 GB | Works well with AWQ checkpoints; pair with Redis Stack or AWS ElastiCache for production persistence. |

**Operational tips**
- Front the service with an HTTPS ingress (AWS ALB, Nginx, or Cloudflare Tunnel) and enforce authentication at the edge.
- Keep `MODEL_CACHE_DIR` on SSD/NVMe to minimize cold-start pulls.
- For deterministic latency, always append `/no_think` in your prompt contract unless the RAG orchestrator explicitly asks for deep reasoning.

## 5. Next steps

- Wire your existing RAG microservice to call `http://rag-llm-service:8081/v1/chat/completions` with the structured prompt contract.
- Store per-request metrics (tokens, latency, cache hits) so you can autoscale the GPU backend after migrating to the cloud profile.
