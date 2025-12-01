#!/usr/bin/env python3
"""
Deployment Script for Leibniz Appointment FSM Microservice

Deploys the appointment service alongside existing services.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


class AppointmentServiceDeployer:
    """Handles deployment of the appointment service"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.service_dir = Path(__file__).parent
        self.env_file = self.project_root / "leibniz_agent" / ".env.leibniz"

    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print(" Checking dependencies...")

        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
            print(" Python 3.9+ required")
            return False
        print(f" Python {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Check required packages
        required_packages = [
            'fastapi', 'uvicorn', 'redis', 'pydantic',
            'pytest', 'httpx', 'pytest-asyncio'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f" {package}")
            except ImportError:
                missing_packages.append(package)
                print(f" {package}")

        if missing_packages:
            print(f"\n️ Missing packages: {', '.join(missing_packages)}")
            print("Run: pip install -r requirements.txt")
            return False

        return True

    def check_redis(self):
        """Check if Redis is available"""
        print("\n Checking Redis...")
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            print(" Redis connected")
            return True
        except Exception as e:
            print(f" Redis not available: {e}")
            print("Please start Redis server on localhost:6379")
            return False

    def check_config(self):
        """Check if configuration files exist"""
        print("\n Checking configuration...")

        if not self.env_file.exists():
            print(f" Environment file not found: {self.env_file}")
            return False

        print(f" Environment file: {self.env_file}")
        return True

    def run_tests(self):
        """Run the test suite"""
        print("\n Running tests...")

        try:
            # Run the test script
            result = subprocess.run([
                sys.executable, 'test_service.py'
            ], cwd=self.service_dir, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(" Basic functionality test passed")
                return True
            else:
                print(" Basic functionality test failed")
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print(" Test timed out")
            return False
        except Exception as e:
            print(f" Test error: {e}")
            return False

            if result.returncode == 0:
                print(" Basic functionality test passed")
                return True
            else:
                print(" Basic functionality test failed")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print(" Test timed out")
            return False
        except Exception as e:
            print(f" Test error: {e}")
            return False

    def start_service(self):
        """Start the appointment service"""
        print("\n Starting appointment service...")

        try:
            # Start service in background
            cmd = [sys.executable, '-m', 'uvicorn', 'app:app',
                   '--host', '0.0.0.0', '--port', '8001',
                   '--reload', '--log-level', 'info']

            print(f"Command: {' '.join(cmd)}")
            print("Service will be available at: http://localhost:8001")
            print("API docs: http://localhost:8001/docs")
            print("Health check: http://localhost:8001/health")
            print("\nStarting service...")

            # Start the service
            process = subprocess.Popen(cmd, cwd=self.service_dir)

            # Wait a moment for startup
            time.sleep(3)

            # Check if process is still running
            if process.poll() is None:
                print(" Service started successfully")
                print(f"PID: {process.pid}")
                return process
            else:
                print(" Service failed to start")
                return None

        except Exception as e:
            print(f" Failed to start service: {e}")
            return None

    def test_integration(self):
        """Test integration with the running service"""
        print("\n Testing service integration...")

        try:
            import requests

            # Test health endpoint
            response = requests.get('http://localhost:8001/health', timeout=5)
            if response.status_code == 200:
                print(" Health endpoint responding")
            else:
                print(f" Health endpoint failed: {response.status_code}")
                return False

            # Test session creation
            response = requests.post('http://localhost:8001/api/v1/session/create', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'session_id' in data:
                    print(" Session creation working")
                    session_id = data['session_id']

                    # Test session processing
                    response = requests.post(
                        f'http://localhost:8001/api/v1/session/{session_id}/process',
                        json={'user_input': 'John Doe'},
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(" Session processing working")
                        return True
                    else:
                        print(f" Session processing failed: {response.status_code}")
                        return False
                else:
                    print(" Invalid session creation response")
                    return False
            else:
                print(f" Session creation failed: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f" Integration test failed: {e}")
            return False
        except ImportError:
            print("️ requests library not available, skipping integration test")
            return True

    def deploy(self):
        """Complete deployment process"""
        print(" Leibniz Appointment FSM Service Deployment")
        print("=" * 50)

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Check Redis
        if not self.check_redis():
            return False

        # Check config
        if not self.check_config():
            return False

        # Run tests
        if not self.run_tests():
            return False

        # Start service
        process = self.start_service()
        if not process:
            return False

        # Test integration
        if not self.test_integration():
            print("️ Integration test failed, but service may still be running")
            return False

        print("\n DEPLOYMENT SUCCESSFUL!")
        print("=" * 30)
        print("Service is running at: http://localhost:8001")
        print("API Documentation: http://localhost:8001/docs")
        print("Health Check: http://localhost:8001/health")
        print("\nTo stop the service, press Ctrl+C or kill the process")

        try:
            # Keep running
            process.wait()
        except KeyboardInterrupt:
            print("\n Stopping service...")
            process.terminate()
            process.wait()
            print(" Service stopped")

        return True


def main():
    """Main deployment function"""
    deployer = AppointmentServiceDeployer()
    success = deployer.deploy()

    if success:
        print("\n Deployment completed successfully!")
        sys.exit(0)
    else:
        print("\n Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()