#!/usr/bin/env python3
"""
Docker Compose Cleanup Script for Leibniz Microservices

Removes duplicate RAG service entries and optimizes docker-compose.yml
for individual service deployment and testing.

Usage:
    python leibniz_agent/services/cleanup_docker_compose.py

This script will:
1. Remove duplicate RAG service entries
2. Ensure clean service definitions
3. Validate the resulting docker-compose.yml
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_docker_compose(file_path: str) -> Dict[str, Any]:
    """Load docker-compose.yml file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f" File not found: {file_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f" YAML parsing error: {e}")
        sys.exit(1)


def save_docker_compose(file_path: str, data: Dict[str, Any]):
    """Save docker-compose.yml file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f" Saved cleaned docker-compose.yml to {file_path}")
    except Exception as e:
        print(f" Failed to save file: {e}")
        sys.exit(1)


def find_duplicate_services(services: Dict[str, Any]) -> list:
    """Find services that appear multiple times with different names"""
    service_names = {}
    duplicates = []

    for service_name, service_config in services.items():
        # Check if this service config matches any other service
        for other_name, other_config in services.items():
            if service_name != other_name and service_config == other_config:
                if service_name not in duplicates:
                    duplicates.append(service_name)
                if other_name not in duplicates:
                    duplicates.append(other_name)

    return duplicates


def remove_duplicate_rag_services(services: Dict[str, Any]) -> Dict[str, Any]:
    """Remove duplicate RAG service entries, keeping the canonical one"""
    cleaned_services = {}
    rag_services = {}

    # First pass: identify all RAG services
    for service_name, service_config in services.items():
        if 'rag' in service_name.lower():
            rag_services[service_name] = service_config
        else:
            cleaned_services[service_name] = service_config

    # If we have multiple RAG services, keep only 'rag'
    if len(rag_services) > 1:
        print(f" Found {len(rag_services)} RAG services: {list(rag_services.keys())}")

        # Keep the canonical 'rag' service if it exists
        if 'rag' in rag_services:
            cleaned_services['rag'] = rag_services['rag']
            print(" Keeping canonical 'rag' service")
        else:
            # If no 'rag', keep the first one alphabetically
            canonical_name = sorted(rag_services.keys())[0]
            cleaned_services[canonical_name] = rag_services[canonical_name]
            print(f" Keeping first RAG service: {canonical_name}")

        removed_count = len(rag_services) - 1
        print(f"️  Removed {removed_count} duplicate RAG service(s)")

    elif len(rag_services) == 1:
        # Only one RAG service, keep it
        service_name = list(rag_services.keys())[0]
        cleaned_services[service_name] = rag_services[service_name]
        print(f" Single RAG service found: {service_name}")

    return cleaned_services


def validate_services(services: Dict[str, Any]) -> bool:
    """Validate that all required services are present and properly configured"""
    required_services = ['intent', 'appointment', 'tts', 'stt-vad', 'rag']  # Note: stt-vad has hyphen
    missing_services = []

    for service in required_services:
        if service not in services:
            missing_services.append(service)

    if missing_services:
        print(f"️  Missing services: {missing_services}")
        return False

    print(" All required services present")
    return True


def optimize_service_config(service_config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize individual service configuration"""
    optimized = service_config.copy()

    # Ensure build context is correct
    if 'build' in optimized:
        build_config = optimized['build']
        if isinstance(build_config, str):
            # Convert string build context to dict
            optimized['build'] = {'context': build_config}
        elif isinstance(build_config, dict):
            # Ensure context is set correctly
            if 'context' not in build_config:
                build_config['context'] = '.'

    # Ensure ports are properly formatted
    if 'ports' in optimized:
        ports = optimized['ports']
        if isinstance(ports, list):
            # Validate port format
            for port in ports:
                if isinstance(port, str):
                    if ':' not in port:
                        print(f"️  Invalid port format: {port}")
                elif not isinstance(port, int):
                    print(f"️  Invalid port type: {port}")

    return optimized


def main():
    """Main cleanup function"""
    print(" Docker Compose Cleanup for Leibniz Microservices")
    print("=" * 60)

    # File paths
    compose_file = Path("c:/Users/AMAR/SINDHv2/SINDH-Orchestra-Complete/docker-compose.leibniz.yml")

    if not compose_file.exists():
        print(f" Docker compose file not found: {compose_file}")
        sys.exit(1)

    print(f" Processing: {compose_file}")

    # Load docker-compose.yml
    compose_data = load_docker_compose(str(compose_file))

    if 'services' not in compose_data:
        print(" No services section found in docker-compose.yml")
        sys.exit(1)

    services = compose_data['services']
    print(f" Found {len(services)} services: {list(services.keys())}")

    # Find duplicates
    duplicates = find_duplicate_services(services)
    if duplicates:
        print(f" Found duplicate services: {duplicates}")

    # Remove duplicate RAG services
    cleaned_services = remove_duplicate_rag_services(services)

    # Optimize service configurations
    optimized_services = {}
    for service_name, service_config in cleaned_services.items():
        optimized_services[service_name] = optimize_service_config(service_config)

    # Update compose data
    compose_data['services'] = optimized_services

    # Validate
    is_valid = validate_services(optimized_services)

    # Save cleaned file
    save_docker_compose(str(compose_file), compose_data)

    # Summary
    print("\n CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Original services: {len(services)}")
    print(f"Cleaned services: {len(optimized_services)}")
    print(f"Removed: {len(services) - len(optimized_services)} duplicate(s)")

    if is_valid:
        print(" Docker compose file is valid and ready for use")
    else:
        print("️  Docker compose file has missing services")

    print("\n Next Steps:")
    print("1. Review the cleaned docker-compose.leibniz.yml")
    print("2. Test individual services: python leibniz_agent/services/test_service_health.py")
    print("3. Build and run services using BUILD_INDIVIDUAL_SERVICES.md guide")


if __name__ == "__main__":
    main()