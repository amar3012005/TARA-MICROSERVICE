#!/usr/bin/env python3
"""
Load Testing Script for Leibniz Appointment FSM Service

Tests service performance under concurrent load.
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from leibniz_agent.services.appointment.app import app
from leibniz_agent.services.appointment.config import AppointmentConfig
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


class LoadTester:
    """Load testing utilities for appointment service"""

    def __init__(self, num_clients=10, requests_per_client=5):
        self.num_clients = num_clients
        self.requests_per_client = requests_per_client
        self.config = AppointmentConfig()

        # Mock Redis for load testing
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
        """Create a test client with mocked dependencies"""
        with patch('leibniz_agent.services.appointment.app.redis_client', self.mock_redis), \
             patch('leibniz_agent.services.appointment.app.config', self.config):
            return TestClient(app)

    async def simulate_user_session(self, client_id):
        """Simulate a complete user appointment booking session"""
        client = self.create_client()

        try:
            # Create session
            start_time = time.time()
            response = client.post('/api/v1/session/create')
            session_id = response.json()['session_id']

            # Process name
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': f'User {client_id}'})

            # Confirm name
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Process email
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': f'user{client_id}@example.com'})

            # Confirm email
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Process phone
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': '+1-555-0123'})

            # Confirm phone
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Process department
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'Academic Advising'})

            # Confirm department
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Process appointment type
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'Course selection'})

            # Confirm appointment type
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Process datetime
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'tomorrow at 2pm'})

            # Confirm datetime
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Process purpose
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'Need help with course selection'})

            # Confirm purpose
            client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})

            # Final confirmation
            response = client.post(f'/api/v1/session/{session_id}/process', json={'user_input': 'yes'})
            end_time = time.time()

            success = response.status_code == 200 and response.json().get('complete') == True
            duration = end_time - start_time

            return {
                'client_id': client_id,
                'success': success,
                'duration': duration,
                'session_id': session_id
            }

        except Exception as e:
            return {
                'client_id': client_id,
                'success': False,
                'duration': 0,
                'error': str(e)
            }

    async def run_load_test(self):
        """Run concurrent load test"""
        print(f" Starting load test with {self.num_clients} clients, {self.requests_per_client} sessions each")
        print("=" * 60)

        start_time = time.time()
        results = []

        # Run clients concurrently
        tasks = []
        for client_id in range(self.num_clients):
            for session_num in range(self.requests_per_client):
                task = asyncio.create_task(self.simulate_user_session(f"{client_id}_{session_num}"))
                tasks.append(task)

        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_sessions = 0
        failed_sessions = 0
        durations = []

        for result in completed_results:
            if isinstance(result, Exception):
                failed_sessions += 1
                print(f" Exception: {result}")
            else:
                results.append(result)
                if result['success']:
                    successful_sessions += 1
                    durations.append(result['duration'])
                else:
                    failed_sessions += 1

        total_time = time.time() - start_time
        total_sessions = len(results)

        # Calculate statistics
        success_rate = (successful_sessions / total_sessions) * 100 if total_sessions > 0 else 0
        avg_duration = statistics.mean(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        throughput = total_sessions / total_time if total_time > 0 else 0

        # Print results
        print("\n LOAD TEST RESULTS")
        print("=" * 40)
        print(f"Total Sessions: {total_sessions}")
        print(f"Successful: {successful_sessions}")
        print(f"Failed: {failed_sessions}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Avg Duration: {avg_duration:.2f}s")
        print(f"Min Duration: {min_duration:.2f}s")
        print(f"Max Duration: {max_duration:.2f}s")
        print(f"Throughput: {throughput:.1f} req/s")
        print("\n PERFORMANCE METRICS")
        print("=" * 40)
        print(f"Average Response Time: {avg_duration:.3f}s")
        print(f"Throughput: {throughput:.1f} req/s")
        print(f"Total Test Time: {total_time:.3f}s")
        # Success criteria
        if success_rate >= 95:
            print(" SUCCESS RATE: EXCELLENT (≥95%)")
        elif success_rate >= 90:
            print("️ SUCCESS RATE: GOOD (≥90%)")
        else:
            print(" SUCCESS RATE: NEEDS IMPROVEMENT (<90%)")

        if avg_duration <= 2.0:
            print(" RESPONSE TIME: EXCELLENT (≤2.0s)")
        elif avg_duration <= 5.0:
            print("️ RESPONSE TIME: ACCEPTABLE (≤5.0s)")
        else:
            print(" RESPONSE TIME: TOO SLOW (>5.0s)")

        if throughput >= 10:
            print(" THROUGHPUT: EXCELLENT (≥10 req/s)")
        elif throughput >= 5:
            print("️ THROUGHPUT: GOOD (≥5 req/s)")
        else:
            print(" THROUGHPUT: LOW (<5 req/s)")

        return {
            'total_sessions': total_sessions,
            'successful': successful_sessions,
            'failed': failed_sessions,
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'throughput': throughput,
            'total_time': total_time
        }


async def main():
    """Main load testing function"""
    print(" Leibniz Appointment FSM Load Tester")
    print("Testing service performance under concurrent load\n")

    # Test configurations
    test_configs = [
        (5, 2, "Small Load Test"),
        (10, 3, "Medium Load Test"),
        (20, 5, "Heavy Load Test")
    ]

    all_results = []

    for num_clients, sessions_per_client, test_name in test_configs:
        print(f"\n {test_name}")
        print("-" * 30)

        tester = LoadTester(num_clients=num_clients, requests_per_client=sessions_per_client)
        results = await tester.run_load_test()
        all_results.append((test_name, results))

        # Brief pause between tests
        await asyncio.sleep(1)

    # Summary
    print("\n LOAD TESTING SUMMARY")
    print("=" * 50)
    for test_name, results in all_results:
        print(f"{test_name}: {results['success_rate']:.1f}% success, {results['throughput']:.1f} req/s")

    print("\n Load testing completed!")


if __name__ == "__main__":
    asyncio.run(main())