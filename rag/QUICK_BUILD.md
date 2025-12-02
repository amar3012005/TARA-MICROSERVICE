# Quick Build Guide - RAG Service

## Fast Build (Without Heavy Packages)

The RAG service now builds quickly by excluding heavy ML packages (~1.5GB).

### Build Time Comparison:
- **Before**: ~40 minutes (with torch, sentence-transformers)
- **After**: ~2-3 minutes (lightweight packages only)

## Step 1: Build Container (Fast)

```bash
cd /home/prometheus/leibniz_agent/services
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml build rag-service-tara
```

This will complete in ~2-3 minutes!

## Step 2: Start Container

```bash
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d rag-service-tara
```

## Step 3: Install Heavy Packages (Inside Container)

### Option A: Using the install script (Recommended)

```bash
# Enter container
docker exec -it tara-rag-service bash

# Run install script
bash /app/install_heavy_packages.sh
```

### Option B: Manual install

```bash
# Enter container
docker exec -it tara-rag-service bash

# Install heavy packages
pip install --default-timeout=1000 --user --no-cache-dir -r /app/leibniz_agent/services/rag/requirements_after.txt
```

**Installation time**: ~5-10 minutes (depends on network speed)

## Step 4: Verify Installation

```bash
# Inside container
python -c "import sentence_transformers; print('✅ OK')"
python -c "import torch; print(f'✅ torch {torch.__version__}')"
```

## What Gets Installed?

The `requirements_after.txt` includes:
- `sentence-transformers` - Embedding models (~500MB)
- `torch` - PyTorch framework (~900MB)
- `tokenizers` - Fast tokenization (~3MB)
- `transformers` - HuggingFace transformers (~100MB)
- `huggingface-hub` - Model downloading (~50MB)

**Total**: ~1.5GB download, ~3GB disk space

## Benefits

✅ **Fast builds** - Build completes in minutes, not hours  
✅ **Flexible** - Install heavy packages when needed  
✅ **Development-friendly** - Faster iteration cycles  
✅ **Production-ready** - Install once, use many times  

## Troubleshooting

### Container won't start without heavy packages?

The RAG service will start but won't be able to process queries until heavy packages are installed. The health check will pass, but actual queries will fail.

### Want automatic installation during build?

Edit `Dockerfile` line 28:
```dockerfile
RUN pip install --default-timeout=1000 --user --no-cache-dir -r requirements.txt -r requirements_after.txt
```

This will install everything during build (slower but automatic).

