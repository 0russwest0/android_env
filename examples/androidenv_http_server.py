import threading
import uuid
from flask import Flask, request, jsonify
from android_env import loader
from android_env.components import config_classes
from android_env.wrappers.gym_wrapper import GymInterfaceWrapper
import numpy as np
import time
import psutil
from android_env.proto import adb_pb2, state_pb2
from google.protobuf.json_format import MessageToDict, ParseDict

app = Flask(__name__)

# 全局环境实例池
class EnvInfo:
    def __init__(self, env):
        self.env = env
        self.created_at = time.time()
        self.last_active = self.created_at
        self.num_resets = 0
        self.num_steps = 0

    def to_dict(self):
        return {
            'created_at': self.created_at,
            'last_active': self.last_active,
            'num_resets': self.num_resets,
            'num_steps': self.num_steps
        }

envs = {}
envs_lock = threading.Lock()

def make_env(config_path):
    config = config_classes.AndroidEnvConfig(
        task=config_classes.FilesystemTaskConfig(path=config_path)
    )
    env = loader.load(config)
    return GymInterfaceWrapper(env)

@app.route('/v1/envs', methods=['POST'])
def create_env():
    data = request.get_json()
    config_path = data.get('config_path', '')
    env = make_env(config_path)
    env_id = str(uuid.uuid4())
    with envs_lock:
        envs[env_id] = EnvInfo(env)
    return jsonify({'env_id': env_id})

@app.route('/v1/envs/<env_id>/reset', methods=['POST'])
def reset_env(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    obs = info.env.reset()
    info.last_active = time.time()
    info.num_resets += 1
    return jsonify({'observation': _to_jsonable(obs)})

@app.route('/v1/envs/<env_id>/step', methods=['POST'])
def step_env(env_id):
    data = request.get_json()
    action = data['action']
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    obs, reward, done, info_dict = info.env.step(_from_jsonable(action))
    info.last_active = time.time()
    info.num_steps += 1
    return jsonify({
        'observation': _to_jsonable(obs),
        'reward': reward,
        'done': done,
        'info': info_dict
    })

@app.route('/v1/envs/<env_id>', methods=['DELETE'])
def close_env(env_id):
    with envs_lock:
        info = envs.pop(env_id, None)
    if info is not None:
        info.env.close()
        return '', 204
    else:
        return jsonify({'error': 'env_id not found'}), 404

@app.route('/v1/envs/<env_id>/action_space', methods=['GET'])
def get_action_space(env_id):
    with envs_lock:
        env = envs.get(env_id)
    if env is None:
        return jsonify({'error': 'env_id not found'}), 404
    return jsonify(_space_to_dict(env.env.action_space))

@app.route('/v1/envs/<env_id>/observation_space', methods=['GET'])
def get_observation_space(env_id):
    with envs_lock:
        env = envs.get(env_id)
    if env is None:
        return jsonify({'error': 'env_id not found'}), 404
    return jsonify(_space_to_dict(env.env.observation_space))

@app.route('/v1/envs/<env_id>/task_extras', methods=['GET'])
def get_task_extras(env_id):
    latest_only = request.args.get('latest_only', 'true').lower() == 'true'
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        extras = info.env.task_extras(latest_only=latest_only)
        return jsonify(_to_jsonable(extras))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/envs/<env_id>/stats', methods=['GET'])
def get_stats(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        stats = info.env.stats()
        return jsonify(_to_jsonable(stats))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/envs/<env_id>/raw_action', methods=['GET'])
def get_raw_action(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        raw_action = info.env.raw_action
        return jsonify(_to_jsonable(raw_action))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/envs/<env_id>/raw_observation', methods=['GET'])
def get_raw_observation(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        raw_obs = info.env.raw_observation
        return jsonify(_to_jsonable(raw_obs))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/envs/<env_id>/execute_adb_call', methods=['POST'])
def execute_adb_call(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        adb_req_dict = request.get_json()
        adb_req = adb_pb2.AdbRequest()
        ParseDict(adb_req_dict, adb_req)
        resp = info.env.execute_adb_call(adb_req)
        return jsonify(MessageToDict(resp, preserving_proto_field_name=True))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/envs/<env_id>/save_state', methods=['POST'])
def save_state(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        req_dict = request.get_json() or {}
        req = state_pb2.SaveStateRequest()
        ParseDict(req_dict, req)
        resp = info.env.save_state(req)
        return jsonify(MessageToDict(resp, preserving_proto_field_name=True))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/envs/<env_id>/load_state', methods=['POST'])
def load_state(env_id):
    with envs_lock:
        info = envs.get(env_id)
    if info is None:
        return jsonify({'error': 'env_id not found'}), 404
    try:
        req_dict = request.get_json() or {}
        req = state_pb2.LoadStateRequest()
        ParseDict(req_dict, req)
        resp = info.env.load_state(req)
        return jsonify(MessageToDict(resp, preserving_proto_field_name=True))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/server_status', methods=['GET'])
def server_status():
    with envs_lock:
        env_ids = list(envs.keys())
        envs_info = {eid: envs[eid].to_dict() for eid in env_ids}
    # 获取进程资源占用
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    return jsonify({
        'num_envs': len(env_ids),
        'env_ids': env_ids,
        'envs_info': envs_info,
        'server_resource': {
            'memory_rss_MB': mem_info.rss // 1024 // 1024,
            'memory_vms_MB': mem_info.vms // 1024 // 1024,
            'cpu_percent': cpu_percent
        }
    })

def _to_jsonable(obj):
    # numpy数组转list，dict递归
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj

def _from_jsonable(obj):
    # 可根据需要实现反序列化
    return obj

def _space_to_dict(space):
    # 简单描述gym.Space
    d = {'name': type(space).__name__}
    for attr in ['shape', 'dtype', 'low', 'high', 'n', 'spaces']:
        if hasattr(space, attr):
            val = getattr(space, attr)
            if isinstance(val, np.ndarray):
                val = val.tolist()
            d[attr] = val
    if hasattr(space, 'spaces') and isinstance(space.spaces, dict):
        d['spaces'] = {k: _space_to_dict(v) for k, v in space.spaces.items()}
    elif hasattr(space, 'spaces') and isinstance(space.spaces, (list, tuple)):
        d['spaces'] = [_space_to_dict(v) for v in space.spaces]
    return d

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True) 