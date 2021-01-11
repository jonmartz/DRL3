import gym
import math
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import csv
import os
from StateTranslations import *
import tensorflow as tf
from ResultSaving import save_results


render = False
# render = True
eps_to_render = 1

# todo: choose source and target env
# source_env_name, target_env_name = 'Acrobot-v1', 'CartPole-v1'
source_env_name, target_env_name = 'CartPole-v1', 'MountainCarContinuous-v0'

# NOT REQUIRED:
# source_env_name, target_env_name = 'MountainCarContinuous-v0', 'CartPole-v1'
# source_env_name, target_env_name = 'Acrobot-v1', 'MountainCarContinuous-v0'

env_params = {
    'CartPole-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.01,
        'action_space': 'discrete', 'use_memory': True, 'reinit_output_weights_each_ep': False,
        'optimizer': tf.compat.v1.train.RMSPropOptimizer
    },
    'Acrobot-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.001,
        'action_space': 'discrete', 'reinit_output_weights_each_ep': False
    },
    'MountainCarContinuous-v0': {
        'policy_hidden_layers': [48], 'baseline_hidden_layers': [480], 'policy_lr': 0.0004, 'baseline_lr': 0.001,
        'action_space': 'continuous', 'reinit_output_weights_each_ep': True,
        'optimizer': tf.compat.v1.train.RMSPropOptimizer
    },
}
global_state_size, global_action_size = 6, 3

saved_path = 'saved_models/%s/model' % source_env_name
target_env = gym.make(target_env_name)
max_steps = target_env.spec.max_episode_steps + 1
reward_threshold = target_env.spec.reward_threshold
print(f'fine_tunning: {source_env_name}>{target_env_name}, max_steps={max_steps}, reward_threshold={reward_threshold}')
print()

mixed_params = env_params[target_env_name]
source_params = env_params[source_env_name]
mixed_params['policy_hidden_layers'] = source_params['policy_hidden_layers']
mixed_params['baseline_hidden_layers'] = source_params['baseline_hidden_layers']
mixed_params['state_translation'] = state_translations[f'{target_env_name}>{source_env_name}']

reinit_output_weights_each_ep = mixed_params.pop('reinit_output_weights_each_ep')

agent = ActorCritic(source_env_name, target_env, global_state_size, global_action_size, render=render,
                    eps_to_render=eps_to_render,
                    saved_path=saved_path,
                    **mixed_params)
results = agent.train(
    fine_tune=True,
    freeze_hidden=True,
    reinit_output_weights_each_ep=reinit_output_weights_each_ep,
)
task_name = f'{source_env_name} to {target_env_name}'
save_results('train history section 2', f'{source_env_name} to {target_env_name}', results)
