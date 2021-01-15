import gym
import math
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import csv
import os
from StateTranslations import *
import tensorflow as tf
from ResultSaving import save_results


# todo: choose sources and target env
source_env_names, target_env_name = ['Acrobot-v1', 'MountainCarContinuous-v0'], 'CartPole-v1'
# source_env_names, target_env_name = ['CartPole-v1', 'Acrobot-v1'], 'MountainCarContinuous-v0'

render = False
eps_to_render = 1
env_params = {
    'CartPole-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.001, 'baseline_lr': 0.01,
        'action_space': 'discrete',
        # 'optimizer': tf.compat.v1.train.RMSPropOptimizer
    },
    'Acrobot-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.001,
        'action_space': 'discrete'
    },
    'MountainCarContinuous-v0': {
        'policy_hidden_layers': [48], 'baseline_hidden_layers': [480], 'policy_lr': 0.0001, 'baseline_lr': 0.01,
        'action_space': 'continuous'
    },
}
global_state_size, global_action_size = 6, 3

target_env = gym.make(target_env_name)
max_steps = target_env.spec.max_episode_steps + 1
reward_threshold = target_env.spec.reward_threshold
print(f'progressive: {source_env_names}>{target_env_name}, max_steps={max_steps}, reward_threshold={reward_threshold}')
print()

source_agents = []
for source_env_name in source_env_names:
    source_params = env_params[source_env_name]
    source_params['state_translation'] = state_translations[f'{target_env_name}>{source_env_name}']
    saved_path = 'saved_models/%s/model' % source_env_name
    source_agents.append(ActorCritic(source_env_name, gym.make(source_env_name), global_state_size, global_action_size,
                                     saved_path=saved_path, is_source=True, **source_params))
target_params = env_params[target_env_name]
target_agent = ActorCritic(target_env_name, target_env, global_state_size, global_action_size, render=render,
                           eps_to_render=eps_to_render, source_agents=source_agents, **target_params)
for i in range(10):  # cross validation
    results = target_agent.train(reinit_output_weights_each_ep=True)
    save_results('train history section 3', f'{source_env_names} to {target_env_name}', results, i)
