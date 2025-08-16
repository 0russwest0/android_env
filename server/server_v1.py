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
import atexit
import threading
import uuid
import time
import psutil
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from android_env import loader
from android_env.components import config_classes
from android_env.wrappers.qwen25vl_wrapper import Qwen25VLWrapper
from absl import logging
from PIL import Image

app = Flask(__name__)

# ###########################################################################
# Global Server Configuration and State
# ###########################################################################

MAX_ENVS = 32
ENV_CONFIGS = {
    "default": [
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_2_xl_android_10',
            # 'run_headless': True,
            # 'adb_path': '/root/Android/Sdk/platform-tools/adb',
            # 'emulator_path': '/root/Android/Sdk/emulator/emulator',
            # 'android_sdk_root': '/root/Android/Sdk',
            # 'android_avd_home': '/root/.android/avd'
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_2_xl_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_2_xl_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_2_xl_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3a_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3a_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3a_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_3a_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_xl_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_xl_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_xl_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4_xl_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4a_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4a_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4a_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_4a_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_5_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_5_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_5_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_5_android_13',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_6_android_10',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_6_android_11',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_6_android_12',
        },
        {
            'task_config_path': '/root/android_env/tasks/dummy.textproto',
            'avd_name': 'pixel_6_android_13',
        },   
    ]
}

envs = {}
envs_lock = threading.Lock()
active_avds = set()
# NOTE: active_avds_lock is not strictly needed anymore as all modifications
# to active_avds now happen inside the protection of envs_lock.
# It is kept for clarity but could be removed.
active_avds_lock = threading.Lock()

class EnvInfo:
    """Holds information about a single environment instance."""
    def __init__(self, env, config_name, full_config):
        self.env = env  # Can be None if status is 'CREATING'
        self.config_name = config_name
        self.full_config = full_config
        self.status = 'CREATING'  # 'CREATING', 'BUSY', or 'IDLE'
        self.created_at = time.time()
        self.last_active = self.created_at
        self.num_resets = 0
        self.num_steps = 0

    def to_dict(self):
        """Serializes the object to a dictionary for JSON responses."""
        return {
            'config_name': self.config_name,
            'avd_name': self.full_config.get('avd_name'),
            'status': self.status,
            'created_at': self.created_at,
            'last_active': self.last_active,
            'num_resets': self.num_resets,
            'num_steps': self.num_steps
        }

def make_env(env_config: dict):
    """Creates a new AndroidEnv instance. This is a time-consuming operation."""
    # Default paths can be set here if not provided in the config
    defaults = {
        'emulator_path': '~/Android/Sdk/emulator/emulator',
        'android_sdk_root': '~/Android/Sdk',
        'android_avd_home': '~/.android/avd',
        'adb_path': '~/Android/Sdk/platform-tools/adb',
        'run_headless': True
    }
    config = {**defaults, **env_config}

    env_config_obj = config_classes.AndroidEnvConfig(
        task=config_classes.FilesystemTaskConfig(path=config['task_config_path']),
        simulator=config_classes.EmulatorConfig(
            emulator_launcher=config_classes.EmulatorLauncherConfig(
                emulator_path=config['emulator_path'],
                android_sdk_root=config['android_sdk_root'],
                android_avd_home=config['android_avd_home'],
                avd_name=config['avd_name'],
                run_headless=config['run_headless'],
            ),
            adb_controller=config_classes.AdbControllerConfig(adb_path=config['adb_path']),
        ),
    )
    env = loader.load(env_config_obj)
    env = Qwen25VLWrapper(env)
    return env

# ###########################################################################
# Core API Endpoints for Environment Management
# ###########################################################################

@app.route('/v1/envs', methods=['POST'])
def create_env():
    """
    Gets or creates an environment. This is the main entry point for clients.
    It tries to reuse an idle environment before creating a new one.
    """
    data = request.get_json()
    config_name = data.get('config_name')
    if not config_name or config_name not in ENV_CONFIGS:
        return jsonify({'error': 'A valid `config_name` must be provided.'}), 400

    # Part 1: Quick, locked decision phase
    reused_env_id = None
    reused_info = None
    with envs_lock:
        # Try to find an idle (sleeping) environment to reuse
        for env_id, info in envs.items():
            if info.config_name == config_name and info.status == 'IDLE':
                info.status = 'BUSY'
                info.last_active = time.time()
                logging.info(f"Reusing idle env {env_id} for config {config_name}.")
                reused_env_id = env_id
                reused_info = info
                break
    if reused_env_id is not None and reused_info is not None:
        # Do the slow reset OUTSIDE the lock to avoid serializing concurrent calls
        timestep = reused_info.env.reset()
        return jsonify({
            'env_id': reused_env_id,
            'reused': True,
            'observation': _encode_observation_pixel(timestep.observation),
        })

    # If no idle envs, check capacity before creating a new one
    if len(envs) >= MAX_ENVS:
        logging.warning("Max environment capacity reached. Request rejected.")
        return jsonify({'error': 'Server at max capacity. Please try again later.'}), 503

    # Find an available AVD and create a placeholder
    chosen_config = None
    with active_avds_lock:
        for config_option in ENV_CONFIGS[config_name]:
            avd_name = config_option.get('avd_name')
            if avd_name and avd_name not in active_avds:
                chosen_config = config_option
                active_avds.add(avd_name)
                break
    
    if not chosen_config:
        logging.warning(f"No available AVDs for config {config_name}.")
        return jsonify({'error': f'All AVDs for config `{config_name}` are currently in use.'}), 503

    # Create and store the placeholder
    env_id = str(uuid.uuid4())
    placeholder_info = EnvInfo(env=None, config_name=config_name, full_config=chosen_config)
    envs[env_id] = placeholder_info
    logging.info(f"Placeholder created for env {env_id} with AVD {chosen_config['avd_name']}. Starting creation...")

    # Part 2: Slow, unlocked creation phase
    try:
        new_env = make_env(chosen_config)
        initial_timestep = new_env.reset() # Get initial observation right away
    except Exception as e:
        logging.error(f"Failed to create environment {env_id}: {e}", exc_info=True)
        # Part 4a: Cleanup on failure
        with envs_lock, active_avds_lock:
            envs.pop(env_id, None)
            active_avds.discard(chosen_config.get('avd_name'))
        return jsonify({'error': f'Failed to create environment: {str(e)}'}), 500

    # Part 3: Quick, locked update phase to fill the placeholder
    with envs_lock:
        info = envs.get(env_id)
        if info:
            info.env = new_env
            info.status = 'BUSY' # Start in BUSY state
            info.num_resets = 1 # The reset has been done
            info.last_active = time.time()
            logging.info(f"Successfully created env {env_id}. It is now BUSY.")
        else:
            # This case is unlikely but possible if env was closed during creation
            logging.warning(f"Env {env_id} was closed during its creation. Cleaning up.")
            new_env.close()
            with active_avds_lock:
                active_avds.discard(chosen_config.get('avd_name'))
            return jsonify({'error': 'Environment was closed during creation.'}), 409

    return jsonify({
        'env_id': env_id, 
        'reused': False, 
        'status': 'READY',
        'observation': _encode_observation_pixel(initial_timestep.observation)
    })


def _get_env_info_and_check_status(env_id):
    """A helper to get env info and verify it's not in a CREATING state."""
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return None, (jsonify({'error': 'env_id not found'}), 404)
    if info.status == 'CREATING':
        return None, (jsonify({'error': 'Environment is still being created.'}), 425) # 425 Too Early
    if info.env is None: # Should not happen if status is not CREATING
        return None, (jsonify({'error': 'Internal server error: env object is missing.'}), 500)
    return info, None

@app.route('/v1/envs/<env_id>/release', methods=['POST'])
def release_env(env_id):
    """Releases an environment back to the pool, making it idle (sleep)."""
    info, error = _get_env_info_and_check_status(env_id)
    if error: return error

    if info.status == 'IDLE':
        return jsonify({'message': 'Environment was already idle.'}), 200

    info.env.reset()
    with envs_lock:
        info.status = 'IDLE'
        info.last_active = time.time()
    logging.info(f"Environment {env_id} released and is now IDLE.")
    return '', 204

@app.route('/v1/envs/<env_id>', methods=['DELETE'])
def close_env(env_id):
    """Forcibly closes and removes an environment."""
    with envs_lock:
        info = envs.pop(env_id, None)

    if info:
        avd_name_to_release = info.full_config.get('avd_name')
        logging.info(f"Closing environment {env_id} with AVD {avd_name_to_release}.")
        
        with active_avds_lock:
            active_avds.discard(avd_name_to_release)

        if info.env: # If the env was fully created
            info.env.close()
        return '', 204
    else:
        return jsonify({'error': 'env_id not found'}), 404

# ###########################################################################
# Interaction Endpoints - Now with status checks
# ###########################################################################

@app.route('/v1/envs/<env_id>/reset', methods=['POST'])
def reset_env(env_id):
    info, error = _get_env_info_and_check_status(env_id)
    if error: return error
    if info.status == 'IDLE':
         return jsonify({'error': 'Cannot reset an idle environment. Get it first.'}), 400

    timestep = info.env.reset()
    with envs_lock:
        info.last_active = time.time()
        info.num_resets += 1
    
    return jsonify({
        'observation': _encode_observation_pixel(timestep.observation),
        'step_type': timestep.step_type.name,
    })

@app.route('/v1/envs/<env_id>/step', methods=['POST'])
def step_env(env_id):
    info, error = _get_env_info_and_check_status(env_id)
    if error: return error
    if info.status == 'IDLE':
        return jsonify({'error': 'Cannot step an idle environment. Get it first.'}), 400
    
    data = request.get_json()
    action = data['action']
    timestep = info.env.step(_from_jsonable(action))
    
    with envs_lock:
        info.last_active = time.time()
        info.num_steps += 1
    
    # Get action execution status from wrapper stats
    action_success = True
    wrapper_stats = {}
    try:
        wrapper_stats = info.env.stats()
        print("Wrapper Stats: ", wrapper_stats)
        action_success = wrapper_stats.get('last_action_success', True)
    except Exception as e:
        logging.warning(f"Could not get wrapper stats: {e}")
    
    # Convert dm_env.TimeStep to standard format
    return jsonify({
        'observation': _encode_observation_pixel(timestep.observation),
        'reward': float(timestep.reward) if timestep.reward is not None else 0.0,
        'done': timestep.step_type.name in ['LAST', 'TERMINATION'],
        'step_type': timestep.step_type.name,
        'discount': float(timestep.discount) if timestep.discount is not None else 1.0,
        'action_success': action_success,
        'info': {
            'wrapper_stats': wrapper_stats
        }
    })

@app.route('/v1/envs/<env_id>/observation_spec', methods=['GET'])
def get_observation_spec(env_id):
    """Returns the observation spec for a given environment."""
    info, error = _get_env_info_and_check_status(env_id)
    if error: return error
    
    spec = info.env.observation_spec()
    return jsonify(_spec_to_dict(spec))

# ###########################################################################
# Server Status and Monitoring - Now with CREATING state
# ###########################################################################

@app.route('/v1/server_status', methods=['GET'])
def server_status():
    """Provides a comprehensive status of the server and all environments."""
    with envs_lock:
        envs_info = {eid: info.to_dict() for eid, info in envs.items()}
    
    status_counts = {'BUSY': 0, 'IDLE': 0, 'CREATING': 0}
    for info in envs_info.values():
        if info['status'] in status_counts:
            status_counts[info['status']] += 1
    
    return jsonify({
        'server_capacity': {
            'max_envs': MAX_ENVS,
            'total_managed': len(envs_info),
            **status_counts
        },
        'active_avds': list(active_avds),
        'environments': envs_info,
        'timestamp': time.time()
    })

# ###########################################################################
# Utility and Cleanup
# ###########################################################################

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_jsonable(x) for x in obj]
    return obj

def _from_jsonable(obj):
    return obj

def _spec_to_dict(spec):
    """Converts a dm_env spec to a JSON-serializable dictionary."""
    if isinstance(spec, dict):
        return {k: _spec_to_dict(v) for k, v in spec.items()}
    
    d = {'name': type(spec).__name__}
    for attr in ['shape', 'dtype', 'minimum', 'maximum', 'num_values']:
        if hasattr(spec, attr):
            val = getattr(spec, attr)
            if hasattr(val, 'name'):
                d[attr] = val.name # Convert dtype to string name
            elif hasattr(val, 'tolist'):
                d[attr] = val.tolist() # Convert numpy array to list
            else:
                d[attr] = val
    return d

# -------------------------------
# Observation encoding utilities
# -------------------------------


def _encode_ndarray_as_png_base64(array: np.ndarray):
    # Only attempt for plausible images: 2D (grayscale) or 3D with channel last
    if array.dtype != np.uint8:
        return None
    if array.ndim == 2:
        mode = 'L'
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        mode = 'RGB' if array.shape[2] == 3 else 'RGBA'
    else:
        return None
    try:
        img = Image.fromarray(array, mode=mode)
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return b64
    except Exception:
        return None


def _encode_observation_pixel(observation):
    # Replace only the 'pixels' field with base64; keep other fields intact
    if not isinstance(observation, dict):
        return _to_jsonable(observation)
    encoded: dict = {}
    for key, value in observation.items():
        if key == 'pixels' and isinstance(value, np.ndarray):
            b64 = _encode_ndarray_as_png_base64(value)
            encoded[key] = b64 if b64 is not None else _to_jsonable(value)
        else:
            encoded[key] = _to_jsonable(value)
    return encoded

def _cleanup_all_envs():
    """Function to be called on server shutdown to close all envs."""
    logging.info("Server shutting down. Cleaning up all environments...")
    with envs_lock:
        env_ids = list(envs.keys())
        for env_id in env_ids:
            info = envs.pop(env_id, None)
            if info and info.env:
                logging.info(f"Closing env: {env_id}")
                info.env.close()
    logging.info("Cleanup complete.")

atexit.register(_cleanup_all_envs)

# ###########################################################################
# Main Execution
# ###########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AndroidEnv HTTP Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on.')
    parser.add_argument('--max-envs', type=int, default=32, help='Maximum number of concurrent environments.')
    
    args = parser.parse_args()
    MAX_ENVS = args.max_envs
    
    logging.set_verbosity('info')
    logging.set_stderrthreshold('info')
    
    logging.info(f"Starting server on {args.host}:{args.port} with max_envs={MAX_ENVS}")
    app.run(host=args.host, port=args.port, threaded=True)