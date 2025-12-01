#!/usr/bin/env python3
"""
Simple Load Testing Script for Leibniz Appointment FSM Service

Tests service performance under concurrent load without full leibniz_agent imports.
"""

import asyncio
import time
import statistics
import sys
import os

# Add only appointment service path
sys.path.insert(0, os.path.dirname(__file__))

# Import only appointment service components
from app import app
from config import AppointmentConfig
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


class SimpleLoadTester:
    """Simple load testing for appointment service"""

    def __init__(self, num_clients=5, sessions_per_client=2):
        self.num_clients = num_clients
        self.sessions_per_client = sessions_per_client
        self.config = AppointmentConfig()

        # Mock Redis
        self.mock_redis = AsyncMock()
        self.stored_data = {}

        async def mock_get(key):
            return self.stored_data.get(key)

        async def mock_setex(key, ttl, value):
            self.stored_data[key] = value
            return True

        self.mock_redis.get.side_effect = mock_get
        self.mock_redis.setex.side_effect = mock_setex
        self.mock_redis.delete.return_value = 1
        self.mock_redis.incr.return_value = 1
        self.mock_redis.ping.return_value = True

    def create_client(self):
        """Create test client with mocked dependencies"""
        with patch('app.redis_client', self.mock_redis), \
             patch('app.config', self.config):
            return TestClient(app)

    async def simulate_session(self, client_id):
        """Simulate a complete appointment booking"""
        try:
            client = self.create_client()
            start_time = time.time()

            # Create session
            response = client.post('/api/v1/session/create')
            if response.status_code != 200:
                return {'success': False, 'error': 'session creation failed'}

            session_id = response.json()['session_id']

            # Quick booking flow (skip confirmations for speed)
            steps = [
                ('John Doe', 'collect_name'),
                ('yes', 'collect_email'),
                ('john@example.com', 'confirm_email'),
                ('yes', 'collect_phone'),
                ('555-1234', 'confirm_phone'),
                ('yes', 'collect_department'),
                ('Academic Advising', 'confirm_department'),
                ('yes', 'collect_appointment_type'),
                ('Course selection', 'confirm_appointment_type'),
                ('yes', 'collect_datetime'),
                ('tomorrow 2pm', 'confirm_datetime'),
                ('yes', 'collect_purpose'),
                ('Help with courses', 'confirm_purpose'),
                ('yes', 'confirm')  # Final confirmation
            ]

            for user_input, expected_state in steps:
                response = client.post(f'/api/v1/session/{session_id}/process',
                                     json={'user_input': user_input})
                if response.status_code != 200:
                    return {'success': False, 'error': f'step failed: {user_input}'}

                data = response.json()
                if data.get('complete') and expected_state == 'confirm':
                    break  # Successfully completed

            end_time = time.time()
            duration = end_time - start_time

            return {
                'client_id': client_id,
                'success': True,
                'duration': duration
            }

        except Exception as e:
            return {
                'client_id': client_id,
                'success': False,
                'error': str(e)
            }

    async def run_test(self):
        """Run the load test"""
        print(f" Running load test: {self.num_clients} clients × {self.sessions_per_client} sessions")

        start_time = time.time()
        tasks = []

        # Create all tasks
        for client_id in range(self.num_clients):
            for session_num in range(self.sessions_per_client):
                task = asyncio.create_task(self.simulate_session(f"{client_id}_{session_num}"))
                tasks.append(task)

        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # Process results
        successful = 0
        failed = 0
        durations = []

        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif result['success']:
                successful += 1
                durations.append(result['duration'])
            else:
                failed += 1

        total_sessions = len(results)
        success_rate = (successful / total_sessions) * 100 if total_sessions > 0 else 0
        avg_duration = statistics.mean(durations) if durations else 0
        throughput = total_sessions / total_time if total_time > 0 else 0

        print("
 RESULTS"        print(f"Sessions: {total_sessions}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(".1f"        print(".2f"        print(".1f"        print(".3f"
        return {
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'throughput': throughput
        }


async def main():
    """Main test function"""
    print(" Appointment Service Load Test")

    tester = SimpleLoadTester(num_clients=3, sessions_per_client=2)
    results = await tester.run_test()

    # Evaluate results
    if results['success_rate'] >= 95 and results['avg_duration'] <= 3.0:
        print(" LOAD TEST PASSED")
    else:
        print("️ LOAD TEST NEEDS ATTENTION")

    print("\n Load testing completed!")


if __name__ == "__main__":
    asyncio.run(main())