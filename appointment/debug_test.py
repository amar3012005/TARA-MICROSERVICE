#!/usr/bin/env python3
"""
Debug script to check actual FSM behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import AppointmentConfig
from fsm_manager import AppointmentFSMManager
from models import AppointmentState

def main():
    config = AppointmentConfig(
        redis_url="redis://localhost:6379",
        session_ttl=1800,
        max_retries=3,
        max_confirmation_attempts=2
    )

    fsm = AppointmentFSMManager(config)

    print("Initial state:", fsm.state)
    print("Initial data:", fsm.data)

    # Test init handler
    response = fsm._handle_init()
    print("Init response:", response[:100])
    print("State after init:", fsm.state)

    # Test name input
    result = fsm.process_input("John Doe")
    print("Name input result keys:", list(result.keys()))
    print("Name input result state:", result["state"])
    print("FSM state after name:", fsm.state)
    print("FSM data name:", fsm.data.name)

if __name__ == "__main__":
    main()