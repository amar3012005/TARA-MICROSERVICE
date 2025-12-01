#!/usr/bin/env python3
"""
RAG Service - Docker Development Workflow Verification Test

This script verifies that the new development-optimized Docker build works correctly.
It tests both development and production modes to ensure all features function as expected.

Usage:
    python verify_docker_build.py [--mode dev|prod|all]
    
    --mode dev   : Test development workflow only
    --mode prod  : Test production workflow only  
    --mode all   : Test both workflows (default)
"""

import argparse
import subprocess
import sys
import time
from typing import Tuple, Optional
import requests

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def log_info(msg: str):
    print(f"{BLUE}ℹ️  {msg}{NC}")

def log_success(msg: str):
    print(f"{GREEN} {msg}{NC}")

def log_warning(msg: str):
    print(f"{YELLOW}️  {msg}{NC}")

def log_error(msg: str):
    print(f"{RED} {msg}{NC}")

def run_command(cmd: str, check: bool = True) -> Tuple[int, str, str]:
    """Run shell command and return (returncode, stdout, stderr)"""
    log_info(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if check and result.returncode != 0:
        log_error(f"Command failed: {cmd}")
        log_error(f"Stderr: {result.stderr}")
        sys.exit(1)
    
    return result.returncode, result.stdout, result.stderr

def check_container_running(container_name: str) -> bool:
    """Check if container is running"""
    _, stdout, _ = run_command(f"docker ps --filter name={container_name} --format '{{{{.Names}}}}'", check=False)
    return container_name in stdout

def wait_for_health(port: int = 8003, timeout: int = 60) -> bool:
    """Wait for service to become healthy"""
    log_info(f"Waiting for service on port {port} to become healthy (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                log_success(f"Service healthy! Response: {response.json()}")
                return True
        except Exception as e:
            pass
        
        time.sleep(2)
    
    log_error(f"Service did not become healthy within {timeout}s")
    return False

def verify_dev_mode():
    """Verify development mode workflow"""
    log_info("=" * 60)
    log_info("Testing Development Mode Workflow")
    log_info("=" * 60)
    
    # Step 1: Build image
    log_info("Step 1: Building lightweight image (should take ~2 min)...")
    start_time = time.time()
    run_command("docker-compose -f docker-compose.leibniz.yml build rag")
    build_time = time.time() - start_time
    log_success(f"Build completed in {build_time:.1f}s")
    
    if build_time > 300:  # 5 minutes
        log_warning(f"Build took longer than expected ({build_time:.1f}s). Expected ~120s.")
    
    # Step 2: Start container
    log_info("Step 2: Starting container in development mode...")
    run_command("docker-compose -f docker-compose.leibniz.yml up -d rag")
    
    time.sleep(5)  # Wait for container startup
    
    if not check_container_running("leibniz-rag"):
        log_error("Container not running!")
        sys.exit(1)
    log_success("Container running")
    
    # Step 3: Verify setup script exists
    log_info("Step 3: Verifying setup script exists...")
    returncode, _, _ = run_command(
        "docker exec leibniz-rag test -f /app/setup_heavy_deps.sh",
        check=False
    )
    if returncode != 0:
        log_error("Setup script not found!")
        sys.exit(1)
    log_success("Setup script exists")
    
    # Step 4: Check lightweight deps installed
    log_info("Step 4: Checking lightweight dependencies...")
    returncode, stdout, _ = run_command(
        "docker exec leibniz-rag python -c 'import fastapi, redis, faiss, numpy; print(\"OK\")'",
        check=False
    )
    if returncode != 0:
        log_error("Lightweight dependencies not installed!")
        sys.exit(1)
    log_success("Lightweight dependencies installed")
    
    # Step 5: Verify torch NOT installed yet
    log_info("Step 5: Verifying torch not installed (expected behavior)...")
    returncode, _, _ = run_command(
        "docker exec leibniz-rag python -c 'import torch'",
        check=False
    )
    if returncode == 0:
        log_warning("Torch is already installed (unexpected in dev mode)")
    else:
        log_success("Torch not installed (correct - install interactively)")
    
    # Step 6: Run setup script
    log_info("Step 6: Running setup script (may take 3-5 minutes)...")
    log_warning("This installs torch and builds FAISS index - please wait...")
    start_time = time.time()
    returncode, stdout, stderr = run_command(
        "docker exec leibniz-rag /app/setup_heavy_deps.sh",
        check=False
    )
    setup_time = time.time() - start_time
    
    if returncode != 0:
        log_error(f"Setup script failed! Stderr: {stderr}")
        sys.exit(1)
    log_success(f"Setup completed in {setup_time:.1f}s")
    
    # Step 7: Verify torch installed
    log_info("Step 7: Verifying torch installed...")
    returncode, stdout, _ = run_command(
        "docker exec leibniz-rag python -c 'import torch; print(torch.__version__)'",
        check=False
    )
    if returncode != 0:
        log_error("Torch not installed after setup!")
        sys.exit(1)
    log_success(f"Torch installed: {stdout.strip()}")
    
    # Step 8: Verify FAISS index built
    log_info("Step 8: Verifying FAISS index built...")
    returncode, stdout, _ = run_command(
        "docker exec leibniz-rag ls -l /app/index/",
        check=False
    )
    if returncode != 0 or "faiss_index" not in stdout:
        log_error("FAISS index not built!")
        sys.exit(1)
    log_success("FAISS index exists")
    
    # Step 9: Start service
    log_info("Step 9: Starting service...")
    run_command(
        "docker exec -d leibniz-rag uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003",
        check=False
    )
    
    if not wait_for_health(port=8003, timeout=30):
        log_error("Service failed to start!")
        # Show logs for debugging
        run_command("docker logs leibniz-rag", check=False)
        sys.exit(1)
    log_success("Service started successfully")
    
    # Step 10: Test health endpoint
    log_info("Step 10: Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        response.raise_for_status()
        log_success(f"Health check passed: {response.json()}")
    except Exception as e:
        log_error(f"Health check failed: {e}")
        sys.exit(1)
    
    # Cleanup
    log_info("Cleaning up development mode test...")
    run_command("docker-compose -f docker-compose.leibniz.yml down", check=False)
    
    log_success("=" * 60)
    log_success("Development Mode Verification: PASSED ")
    log_success("=" * 60)

def verify_prod_mode():
    """Verify production mode workflow"""
    log_info("=" * 60)
    log_info("Testing Production Mode Workflow")
    log_info("=" * 60)
    
    # Step 1: Start production service
    log_info("Step 1: Starting production service (auto-setup enabled)...")
    log_warning("First start will take ~8 minutes (installs torch + builds index)...")
    start_time = time.time()
    run_command("docker-compose -f docker-compose.leibniz.yml --profile production up -d rag-prod")
    
    # Step 2: Wait for setup to complete and service to start
    log_info("Step 2: Waiting for auto-setup and service startup...")
    if not wait_for_health(port=8003, timeout=600):  # 10 min timeout
        log_error("Production service failed to start!")
        run_command("docker logs leibniz-rag-prod", check=False)
        sys.exit(1)
    
    startup_time = time.time() - start_time
    log_success(f"Production service started in {startup_time:.1f}s")
    
    # Step 3: Verify setup marker created
    log_info("Step 3: Verifying setup marker exists...")
    returncode, _, _ = run_command(
        "docker exec leibniz-rag-prod test -f /app/.setup_complete",
        check=False
    )
    if returncode != 0:
        log_error("Setup marker not found!")
        sys.exit(1)
    log_success("Setup marker exists")
    
    # Step 4: Test health endpoint
    log_info("Step 4: Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        response.raise_for_status()
        log_success(f"Health check passed: {response.json()}")
    except Exception as e:
        log_error(f"Health check failed: {e}")
        sys.exit(1)
    
    # Step 5: Test restart (should be fast now)
    log_info("Step 5: Testing restart (should be fast with cached setup)...")
    run_command("docker restart leibniz-rag-prod")
    
    if not wait_for_health(port=8003, timeout=30):
        log_error("Service failed to restart!")
        sys.exit(1)
    log_success("Restart successful (fast)")
    
    # Cleanup
    log_info("Cleaning up production mode test...")
    run_command("docker-compose -f docker-compose.leibniz.yml --profile production down", check=False)
    
    log_success("=" * 60)
    log_success("Production Mode Verification: PASSED ")
    log_success("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Verify RAG Docker development workflow")
    parser.add_argument(
        "--mode",
        choices=["dev", "prod", "all"],
        default="all",
        help="Test mode: dev (development only), prod (production only), or all (both)"
    )
    args = parser.parse_args()
    
    log_info("RAG Service Docker Build Verification")
    log_info("This will test the new development-optimized Docker workflow")
    log_info("")
    
    try:
        if args.mode in ["dev", "all"]:
            verify_dev_mode()
        
        if args.mode in ["prod", "all"]:
            verify_prod_mode()
        
        log_success("")
        log_success(" All verification tests passed!")
        log_success("")
        log_success("Development workflow is ready to use:")
        log_success("  PowerShell: .\\rag-dev.ps1 build; .\\rag-dev.ps1 start; .\\rag-dev.ps1 setup; .\\rag-dev.ps1 run")
        log_success("  Bash: ./rag-dev.sh build; ./rag-dev.sh start; ./rag-dev.sh setup; ./rag-dev.sh run")
        
    except KeyboardInterrupt:
        log_warning("\nVerification interrupted by user")
        log_info("Cleaning up...")
        run_command("docker-compose -f docker-compose.leibniz.yml down", check=False)
        run_command("docker-compose -f docker-compose.leibniz.yml --profile production down", check=False)
        sys.exit(1)
    except Exception as e:
        log_error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
