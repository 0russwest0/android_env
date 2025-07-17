from androidenv_http_remote_env import RemoteEnv
import numpy as np
import sys

def sample_action(action_space):
    action = {}
    for k, v in action_space.items():
        if v['name'] == 'Discrete':
            action[k] = int(np.random.randint(0, v['n']))
        elif v['name'] == 'Box':
            low = np.array(v['low'])
            high = np.array(v['high'])
            shape = tuple(v['shape'])
            action[k] = np.random.uniform(low, high, size=shape).astype(np.float32)
        else:
            raise NotImplementedError(f"Unknown action space type: {v['name']}")
    return action

if __name__ == '__main__':
    # 用法: python androidenv_http_random_agent.py <server_url> <config_path> <num_episodes>
    if len(sys.argv) < 4:
        print("用法: python androidenv_http_random_agent.py <server_url> <config_path> <num_episodes>")
        sys.exit(1)
    server_url = sys.argv[1].rstrip('/')
    config_path = sys.argv[2]
    num_episodes = int(sys.argv[3])

    env = RemoteEnv(server_url, config_path)
    action_space = env.action_space()

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        while not done:
            action = sample_action(action_space)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            print(f'Episode {ep+1}, Step {step_count}, Reward: {reward}, Done: {done}')
        print(f'Episode {ep+1} finished, Total reward: {total_reward}, Steps: {step_count}')

    env.close()
    print('Env closed.') 