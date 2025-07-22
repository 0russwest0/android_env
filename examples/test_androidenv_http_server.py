import requests
import pytest

BASE_URL = 'http://127.0.0.1:5000/v1'

def test_create_env():
    resp = requests.post(f'{BASE_URL}/envs', json={
        'task_config_path': '/root/android_env/tasks/mdp_0000.textproto',
        'avd_name': 'TestAvd'
    })
    assert resp.status_code == 200
    env_id = resp.json()['env_id']
    return env_id

def test_env_lifecycle():
    env_id = test_create_env()
    print('Created env:', env_id)

    # Reset
    resp = requests.post(f'{BASE_URL}/envs/{env_id}/reset')
    assert resp.status_code == 200
    assert 'observation' in resp.json()

    # Get action space
    resp = requests.get(f'{BASE_URL}/envs/{env_id}/action_space')
    assert resp.status_code == 200
    assert 'name' in resp.json()

    # Get observation space
    resp = requests.get(f'{BASE_URL}/envs/{env_id}/observation_space')
    assert resp.status_code == 200
    assert 'name' in resp.json()

    # Step (dummy action, may need to adjust based on your env)
    action = {}  # TODO: fill with a valid action for your env
    resp = requests.post(f'{BASE_URL}/envs/{env_id}/step', json={'action': action})
    # Accept 200 or 400 if action is invalid
    assert resp.status_code in (200, 400)

    # Close env
    resp = requests.delete(f'{BASE_URL}/envs/{env_id}')
    assert resp.status_code in (204, 200)

if __name__ == '__main__':
    test_env_lifecycle()
    print('Lifecycle test passed.')
