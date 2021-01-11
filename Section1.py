import gym
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import csv
import os
from ResultSaving import save_results


# todo: choose env
env_name = 'CartPole-v1'  # state_size=4, action_size=2
# env_name = 'Acrobot-v1'  # state_size=6, action_size=3
# env_name = 'MountainCarContinuous-v0'  # state_size=2, action_size=continuous

render = False
eps_to_render = 10
params = {
    'CartPole-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.01,
        'action_space': 'discrete', 'use_memory': True},
    'Acrobot-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.001,
        'action_space': 'discrete', 'use_memory': False},
    'MountainCarContinuous-v0': {
        'policy_hidden_layers': [48], 'baseline_hidden_layers': [480], 'policy_lr': 0.0001, 'baseline_lr': 0.01,
        'action_space': 'continuous', 'use_memory': False},
}
global_state_size, global_action_size = 6, 3
env = gym.make(env_name)
max_steps = env.spec.max_episode_steps + 1
reward_threshold = env.spec.reward_threshold
print('env: %s, max_steps=%s, reward_threshold=%s\n' % (env_name, max_steps, reward_threshold))
agent = ActorCritic(env_name, env, global_state_size, global_action_size,
                    render=render, eps_to_render=eps_to_render,
                    # save_final_model=True,
                    **params[env_name])
results = agent.train()
save_results('train history section 1', f'{env_name}', results)
