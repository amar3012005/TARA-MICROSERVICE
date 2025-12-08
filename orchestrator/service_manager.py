"""
Service Manager for Orchestrator
Manages lifecycle of dependent microservices (start, stop, health check)
"""

import asyncio
import logging
import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple
import aiohttp

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages Docker containers for TARA microservices"""
    
    def __init__(self, docker_compose_file: str = None, docker_context: str = "desktop-linux"):
        """
        Initialize Service Manager
        
        Args:
            docker_compose_file: Path to docker-compose file (optional, uses docker commands if None)
            docker_context: Docker context to use (default: desktop-linux)
        """
        self.docker_compose_file = docker_compose_file
        self.docker_context = docker_context
        # Detect project name from environment or use default
        project_name = os.getenv("DOCKER_PROJECT_NAME", "tara-task")
        container_prefix = f"{project_name.replace('_', '-')}"
        
        self.services_config = {
            "redis": {
                "container": f"{container_prefix}-redis",
                "health_url": None,  # Redis doesn't have HTTP health endpoint
                "port": 6000,
                "start_order": 1
            },
            "stt": {
                "container": f"{container_prefix}-stt-vad",
                "health_url": "http://localhost:6001/health",
                "port": 6001,
                "start_order": 2,
                "fastrtc_url": "http://localhost:6012"
            },
            "rag": {
                "container": f"{container_prefix}-rag",
                "health_url": "http://localhost:6003/health",
                "port": 6003,
                "start_order": 3
            },
            "tts": {
                "container": f"{container_prefix}-tts-sarvam",
                "health_url": "http://localhost:6005/health",
                "port": 6005,
                "start_order": 4,
                "fastrtc_url": "http://localhost:6005/fastrtc"
            }
        }
    
    async def _run_docker_command(self, command: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Run docker command with context"""
        try:
            # Set docker context first if specified (only once per session)
            if self.docker_context:
                # Use subprocess for context switching (synchronous)
                import subprocess
                subprocess.run(["docker", "context", "use", self.docker_context], 
                             capture_output=True, timeout=5)
            
            # Run actual docker command
            full_command = ["docker"] + command
            
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            output = stdout.decode() + stderr.decode()
            success = process.returncode == 0
            
            return success, output
        except asyncio.TimeoutError:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)
    
    async def is_container_running(self, container_name: str) -> bool:
        """Check if container is running"""
        success, output = await self._run_docker_command(
            ["ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
        )
        return container_name in output
    
    async def start_container(self, container_name: str) -> bool:
        """Start a Docker container"""
        if await self.is_container_running(container_name):
            logger.info(f"✅ {container_name} is already running")
            return True
        
        logger.info(f"🚀 Starting {container_name}...")
        
        # Try docker start first (if container exists)
        success, output = await self._run_docker_command(["start", container_name])
        if success:
            logger.info(f"✅ Started existing container: {container_name}")
            return True
        
        # If docker-compose file is provided, use docker-compose up
        if self.docker_compose_file:
            logger.info(f"📦 Using docker-compose to start {container_name}...")
            success, output = await self._run_docker_command(
                ["compose", "-f", self.docker_compose_file, "up", "-d", container_name]
            )
            if success:
                logger.info(f"✅ Started {container_name} via docker-compose")
                return True
        
        logger.warning(f"⚠️ Could not start {container_name}: {output}")
        return False
    
    async def check_service_health(self, service_name: str, health_url: Optional[str] = None, max_retries: int = 30, retry_delay: int = 2) -> bool:
        """Check service health with retries"""
        if not health_url:
            # For Redis, check if container is running
            container_name = self.services_config[service_name]["container"]
            return await self.is_container_running(container_name)
        
        logger.info(f"🏥 Checking health of {service_name} at {health_url}...")
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            logger.info(f"✅ {service_name} is healthy (attempt {attempt + 1}/{max_retries})")
                            return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"⏳ {service_name} not ready yet (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(f"❌ {service_name} health check failed after {max_retries} attempts: {e}")
        
        return False
    
    async def start_all_services(self, skip_services: List[str] = None) -> Dict[str, bool]:
        """Start all services in order"""
        skip_services = skip_services or []
        results = {}
        
        # Sort services by start_order
        sorted_services = sorted(
            self.services_config.items(),
            key=lambda x: x[1]["start_order"]
        )
        
        for service_name, config in sorted_services:
            if service_name in skip_services:
                logger.info(f"⏭️ Skipping {service_name} (configured to skip)")
                results[service_name] = True
                continue
            
            container_name = config["container"]
            
            # Start container
            started = await self.start_container(container_name)
            if not started:
                results[service_name] = False
                logger.error(f"❌ Failed to start {service_name}")
                continue
            
            # Wait a bit for container to initialize
            await asyncio.sleep(3)
            
            # Check health
            health_url = config.get("health_url")
            healthy = await self.check_service_health(service_name, health_url)
            results[service_name] = healthy
            
            if not healthy:
                logger.error(f"❌ {service_name} failed health check")
            else:
                logger.info(f"✅ {service_name} is ready")
        
        return results
    
    def get_service_urls(self) -> Dict[str, str]:
        """Get FastRTC URLs for STT and TTS"""
        urls = {}
        for service_name, config in self.services_config.items():
            if "fastrtc_url" in config:
                urls[service_name] = config["fastrtc_url"]
        return urls
    
    async def ensure_network(self, network_name: str = None) -> bool:
        """Ensure Docker network exists"""
        if not network_name:
            project_name = os.getenv("DOCKER_PROJECT_NAME", "tara-task")
            network_name = f"{project_name.replace('_', '-')}-network"
        
        logger.info(f"🌐 Ensuring network {network_name} exists...")
        
        # Check if network exists
        success, output = await self._run_docker_command(["network", "ls", "--format", "{{.Name}}"])
        if network_name in output:
            logger.info(f"✅ Network {network_name} already exists")
            return True
        
        # Create network
        logger.info(f"📦 Creating network {network_name}...")
        success, output = await self._run_docker_command(["network", "create", network_name])
        if success:
            logger.info(f"✅ Created network {network_name}")
            return True
        else:
            logger.warning(f"⚠️ Could not create network: {output}")
            return False

