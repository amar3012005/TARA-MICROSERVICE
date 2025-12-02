# Installing Heavy Packages After Container Build

The RAG service Dockerfile has been optimized to build quickly by excluding heavy ML packages. These packages need to be installed manually after the container is running.

## Quick Install (Inside Container)

### Option 1: Using the install script

```bash
# Enter the container
docker exec -it tara-rag-service bash

# Run the install script
bash /app/leibniz_agent/services/rag/install_heavy_packages.sh
```

### Option 2: Manual install

```bash
# Enter the container
docker exec -it tara-rag-service bash

# Install heavy packages
pip install --default-timeout=1000 --user --no-cache-dir -r /app/leibniz_agent/services/rag/requirements_after.txt
```

## What Gets Installed

The heavy packages include:
- **sentence-transformers** (~500MB) - Embedding models
- **torch** (~900MB) - PyTorch framework
- **tokenizers** (~3MB) - Fast tokenization
- **transformers** (~100MB) - HuggingFace transformers
- **huggingface-hub** (~50MB) - Model downloading

**Total size**: ~1.5GB download, ~3GB disk space

## Verify Installation

After installation, verify it works:

```bash
python -c "import sentence_transformers; print('✅ sentence-transformers OK')"
python -c "import torch; print(f'✅ torch OK - version {torch.__version__}')"
```

## Why This Approach?

1. **Faster builds**: Docker build completes in ~2 minutes instead of ~40 minutes
2. **Flexibility**: You can choose when to install heavy packages
3. **Development**: Faster iteration during development
4. **Production**: Install heavy packages once, then use the container

## Alternative: Install During Build

If you want to install heavy packages during build (slower but automatic), modify `Dockerfile`:

```dockerfile
# Install ALL dependencies including heavy ones
RUN pip install --default-timeout=1000 --user --no-cache-dir -r requirements.txt -r requirements_after.txt
```

## Troubleshooting

### Out of disk space
```bash
# Check disk usage
df -h

# Clean pip cache
pip cache purge
```

### Network timeout
```bash
# Increase timeout
pip install --default-timeout=2000 --user --no-cache-dir -r requirements_after.txt
```

### Permission errors
```bash
# Use --user flag (already included)
pip install --user --no-cache-dir -r requirements_after.txt
```




