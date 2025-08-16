# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import requests
import argparse
from pprint import pprint

class APIClient:
    """A simple client for interacting with the AndroidEnv HTTP server."""

    def __init__(self, host='localhost', port=5000):
        self.base_url = f"http://{host}:{port}/v1"

    def _make_request(self, method, endpoint, **kwargs):
        """Helper function to make HTTP requests."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            if response.status_code == 204:
                return None
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            if e.response is not None:
                print(f"Response body: {e.response.text}")
            return None
        except ValueError: # If response is not JSON
            return response.text

    def get_server_status(self):
        """Fetches the current status of the server."""
        print("\n" + "="*20 + " Fetching Server Status " + "="*20)
        status = self._make_request('GET', '/server_status')
        if status:
            pprint(status)
        return status

    def get_environment(self, config_name: str):
        """Requests a new or reused environment."""
        print(f"\n---> Requesting environment with config: '{config_name}'...")
        return self._make_request('POST', '/envs', json={'config_name': config_name})

    def release_environment(self, env_id: str):
        """Releases an environment back to the idle pool."""
        print(f"\n---> Releasing environment {env_id}...")
        return self._make_request('POST', f'/envs/{env_id}/release')

    def close_environment(self, env_id: str):
        """Forcibly closes and removes an environment."""
        print(f"\n---> Closing environment {env_id}...")
        return self._make_request('DELETE', f'/envs/{env_id}')

    def step(self, env_id: str):
        """Performs a dummy step action."""
        print(f"\n---> Performing a step in {env_id}...")
        # This is a dummy action. A real agent would generate a meaningful action.
        # dummy_action = {"action_type": 0, "touch_position": [0.5, 0.5]}
        dummy_action = {}
        return self._make_request('POST', f'/envs/{env_id}/step', json={'action': dummy_action})

    def get_observation_spec(self, env_id: str):
        """Gets the observation spec for the environment."""
        print(f"\n---> Getting observation spec for {env_id}...")
        return self._make_request('GET', f'/envs/{env_id}/observation_spec')


def run_test_scenario(host, port):
    """Executes a sequence of tests against the server."""
    client = APIClient(host, port)

    # 1. Check initial server status
    client.get_server_status()

    # 2. Request a new environment using the "default" config
    response = client.get_environment("default")
    if not response or 'env_id' not in response:
        print("Failed to get environment. Exiting test.")
        return
    env_id_1 = response['env_id']
    print(f"Successfully got environment: {env_id_1}")
    client.get_server_status()

    # 3. Get observation spec and check for screen dimensions
    spec = client.get_observation_spec(env_id_1)
    print(spec)
    if spec:
        pixels_spec = spec.get('pixels', {})
        shape = pixels_spec.get('shape')
        if shape and len(shape) >= 2:
            print(f"SUCCESS: Found screen dimensions in spec: {shape[1]}x{shape[0]}")
        else:
            print("FAILURE: Could not find screen dimensions in observation spec.")
    else:
        print("FAILURE: Call to get observation spec failed.")

    # 4. Perform a step
    step_result = client.step(env_id_1)
    if step_result:
        print("Step successful.")
        print(f"Action success status: {step_result['action_success']}")

    # 5. Release the environment (put it to sleep)
    client.release_environment(env_id_1)
    client.get_server_status()

    # 6. Request the same config again, expecting to reuse the previous env
    response = client.get_environment("default")
    if not response or 'env_id' not in response:
        print("Failed to get environment on second attempt. Exiting test.")
        return
    
    env_id_2 = response['env_id']
    reused = response.get('reused', False)
    print(f"Got environment {env_id_2}. Reused: {reused}")
    if env_id_1 == env_id_2 and reused:
        print("SUCCESS: Environment was correctly reused.")
    else:
        print("FAILURE: Environment was not reused as expected.")
    
    client.get_server_status()

    # 7. Close the environment for real this time
    client.close_environment(env_id_2)
    client.get_server_status()

    print("\n" + "="*20 + " Test Scenario Complete " + "="*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for AndroidEnv HTTP Server")
    parser.add_argument('--host', type=str, default='localhost', help='Host of the server.')
    parser.add_argument('--port', type=int, default=5000, help='Port of the server.')
    
    args = parser.parse_args()
    
    run_test_scenario(args.host, args.port)
