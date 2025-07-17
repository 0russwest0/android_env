import requests
import numpy as np

class RemoteEnv:
    def __init__(self, server_url, config_path=None, env_id=None):
        self.server_url = server_url.rstrip('/')
        if env_id is not None:
            self.env_id = env_id
        else:
            assert config_path is not None, 'config_path must be provided if env_id is None'
            resp = requests.post(f'{self.server_url}/v1/envs', json={'config_path': config_path})
            resp.raise_for_status()
            self.env_id = resp.json()['env_id']
        self._action_space = None
        self._observation_space = None

    def action_space(self):
        if self._action_space is None:
            resp = requests.get(f'{self.server_url}/v1/envs/{self.env_id}/action_space')
            resp.raise_for_status()
            self._action_space = resp.json()
        return self._action_space

    def observation_space(self):
        if self._observation_space is None:
            resp = requests.get(f'{self.server_url}/v1/envs/{self.env_id}/observation_space')
            resp.raise_for_status()
            self._observation_space = resp.json()
        return self._observation_space

    def reset(self):
        resp = requests.post(f'{self.server_url}/v1/envs/{self.env_id}/reset')
        resp.raise_for_status()
        return self._from_jsonable(resp.json()['observation'])

    def step(self, action):
        resp = requests.post(f'{self.server_url}/v1/envs/{self.env_id}/step', json={'action': self._to_jsonable(action)})
        resp.raise_for_status()
        result = resp.json()
        obs = self._from_jsonable(result['observation'])
        reward = result['reward']
        done = result['done']
        info = result['info']
        return obs, reward, done, info

    def close(self):
        requests.delete(f'{self.server_url}/v1/envs/{self.env_id}')

    @staticmethod
    def _to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: RemoteEnv._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RemoteEnv._to_jsonable(x) for x in obj]
        return obj

    @staticmethod
    def _from_jsonable(obj):
        if isinstance(obj, list):
            return np.array(obj)
        if isinstance(obj, dict):
            return {k: RemoteEnv._from_jsonable(v) for k, v in obj.items()}
        return obj 