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

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image
from pprint import pprint
import readline # For better input editing
import base64
import io

# -----------------------------------------------------------------------------
# Utilities for saving observation screenshots
# -----------------------------------------------------------------------------

_DEFAULT_SCREENSHOT_DIR = "/mnt/data/oyj/screenshots"

def _maybe_save_observation_image(observation: Optional[dict], env_id: Optional[str], label: str = "screenshot") -> Optional[str]:
    """Save observation['pixels'] to a PNG file if available.

    Supports pixels provided as:
      - numpy array / array-like
      - base64 string (optionally with data:image/*;base64, prefix)
      - bytes / bytearray
      - PIL.Image.Image

    Returns the absolute path of the saved file, or None if not saved.
    """
    if not observation or 'pixels' not in observation:
        return None

    pixels = observation.get('pixels')

    image: Optional[Image.Image] = None

    # Case 1: Already a PIL Image
    if isinstance(pixels, Image.Image):
        image = pixels

    # Case 2: Base64-encoded string (with or without data URL prefix)
    elif isinstance(pixels, str):
        try:
            b64_str = pixels
            if ',' in b64_str and b64_str.lower().startswith('data:image/'):
                b64_str = b64_str.split(',', 1)[1]
            decoded = base64.b64decode(b64_str)
            image = Image.open(io.BytesIO(decoded))
        except Exception:
            image = None

    # Case 3: Raw bytes that represent an encoded image
    elif isinstance(pixels, (bytes, bytearray)):
        try:
            image = Image.open(io.BytesIO(pixels))
        except Exception:
            image = None

    # Case 4: Array-like
    if image is None:
        try:
            arr = np.asarray(pixels)

            # Determine mode
            if arr.ndim == 2:
                mode = 'L'
            elif arr.ndim == 3 and arr.shape[2] == 4:
                mode = 'RGBA'
            else:
                mode = 'RGB'

            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            image = Image.fromarray(arr, mode=mode)
        except Exception:
            return None

    os.makedirs(_DEFAULT_SCREENSHOT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    env_tag = env_id[:8] if env_id else 'noenv'
    filename = f"{env_tag}_{label}_{ts}.png"
    out_path = os.path.join(_DEFAULT_SCREENSHOT_DIR, filename)

    try:
        image.save(out_path)
        print(f"<-- Saved screenshot: {out_path}")
        return out_path
    except Exception as e:
        print(f"---! Failed to save screenshot: {e}")
        return None

class InteractiveClient:
    """A client for interactively testing actions on a remote AndroidEnv."""

    def __init__(self, host, port):
        self.base_url = f"http://{host}:{port}/v1"
        self.env_id = None
        self.step_count = 0

    def _make_request(self, method, endpoint, **kwargs):
        """Helper function to make HTTP requests."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=60, **kwargs)
            response.raise_for_status()
            if response.status_code == 204:
                return {"status": "success", "message": "No content returned."}
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"\n---! Error making request to {url}: {e}")
            if e.response is not None:
                print(f"---! Response body: {e.response.text}")
            return None

    def start_session(self, config_name):
        """Starts a new session by acquiring an environment."""
        print(f"--> Requesting environment with config: '{config_name}'...")
        response = self._make_request('POST', '/envs', json={'config_name': config_name})
        if response and response.get('env_id'):
            self.env_id = response['env_id']
            self.step_count = 0
            print(f"<-- Successfully got environment: {self.env_id}")
            if response.get('reused'):
                print("<-- Reused an existing idle environment.")
            return response
        else:
            print("<--! Failed to get an environment. Aborting.")
            return False

    def step(self, action_dict):
        """Sends a step action to the environment."""
        print(f"--> Sending action to {self.env_id}...")
        return self._make_request('POST', f'/envs/{self.env_id}/step', json={'action': action_dict})

    def reset(self):
        """Resets the current environment."""
        print(f"--> Resetting environment {self.env_id}...")
        return self._make_request('POST', f'/envs/{self.env_id}/reset')

    def get_status(self):
        """Gets the full server status."""
        return self._make_request('GET', '/server_status')

    def get_observation_spec(self):
        """Gets the observation spec for the current environment."""
        if not self.env_id:
            print("---! No active environment. Start a session first.")
            return None
        print(f"--> Fetching observation_spec for {self.env_id}...")
        return self._make_request('GET', f'/envs/{self.env_id}/observation_spec')

    def release(self):
        """Releases the environment back to the pool."""
        print(f"--> Releasing environment {self.env_id}...")
        response = self._make_request('POST', f'/envs/{self.env_id}/release')
        if response:
            self.env_id = None
        return response

    def close(self):
        """Forcibly closes the environment."""
        if not self.env_id:
            return
        print(f"--> Closing environment {self.env_id}...")
        self._make_request('DELETE', f'/envs/{self.env_id}')
        self.env_id = None
    
    def print_help(self):
        print("\n" + "="*20 + " HELP " + "="*20)
        print("Enter an action as a JSON string. Special commands:")
        print("  help              - Show this help message.")
        print("  status            - Get the current server status.")
        print("  obs_spec          - Print observation_spec of the current environment.")
        print("  reset             - Reset the current environment.")
        print("  release           - Release the env (sleep) and get a new one.")
        print("  quit / exit       - Close the environment and exit the script.")
        print("\nAction Examples:")
        print('  {"action": "click", "coordinate": [540, 1170]}')
        print('  {"action": "long_press", "coordinate": [540, 1170], "time": 2.0}')
        print('  {"action": "swipe", "coordinate": [540, 1500], "coordinate2": [540, 500]}')
        print('  {"action": "type", "text": "hello from test script"}')
        print('  {"action": "system_button", "button": "Home"}')
        print("="*46 + "\n")


def main_loop(client):
    """The main interactive loop."""
    client.print_help()
    while True:
        try:
            user_input = input(f"Action for env [{client.env_id[:8]}]> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                break
            
            elif user_input.lower() == 'help':
                client.print_help()
                continue
            
            elif user_input.lower() == 'status':
                pprint(client.get_status())
                continue

            elif user_input.lower() in ['obs_spec', 'spec']:
                spec = client.get_observation_spec()
                if spec is not None:
                    pprint(spec)
                continue

            elif user_input.lower() == 'reset':
                reset_resp = client.reset()
                if reset_resp:
                    print("<-- Environment reset.")
                    _maybe_save_observation_image(reset_resp.get('observation'), client.env_id, label='reset')
                else:
                    pprint(reset_resp)
                continue
            
            elif user_input.lower() == 'release':
                client.release()
                print("<-- Environment released. You need to request a new one to continue.")
                # For simplicity, we exit. The user can restart for a new env.
                break

            try:
                action_dict = json.loads(user_input)
                response = client.step(action_dict)
                if response:
                    client.step_count += 1
                    _maybe_save_observation_image(response.get('observation'), client.env_id, label=f'step{client.step_count:04d}')

            except json.JSONDecodeError:
                print(f"---! Invalid JSON. Please try again. Type 'help' for examples.")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive client for AndroidEnv HTTP Server")
    parser.add_argument('--host', type=str, default='localhost', help='Host of the server.')
    parser.add_argument('--port', type=int, default=5000, help='Port of the server.')
    parser.add_argument('--config', type=str, default='default', help='Environment config profile to use.')
    
    args = parser.parse_args()
    
    client = InteractiveClient(args.host, args.port)

    session_resp = client.start_session(args.config)
    if session_resp:
        # Save initial observation screenshot if available
        _maybe_save_observation_image(
            session_resp.get('observation'), client.env_id, label='initial'
        )
        try:
            main_loop(client)
        finally:
            print("\nCleaning up...")
            client.close() # Ensure env is closed on exit
            print("Session closed.")
