import gym
import math
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import csv
import os


def translate_cartpole_to_acrobot(cartpole_state):
    """
    Translates a cartpole state to an acrobot state, so the
    pre-trained acrobot model can extract some meaning from it.
    :param cartpole_state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    :return: acrobot state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]

             where theta1 and theta2 are the angles of the inner and outer joints and thetaDot_i is angular velocity
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = cartpole_state
    return [1, 0, 1, 0, -1/(pole_ang_vel + pole_angle + cart_vel), 0]


def translate_mountaincar_to_cartpole(mountaincart_state):
    """
    Translates a mountaincar state to an cartpole state, so the
    pre-trained cartpole model can extract some meaning from it.
    :param mountaincart_state: [Car Position, Car Velocity]
    :return: cartpole state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    """
    car_pos, car_vel = mountaincart_state
    return [car_pos, car_vel, -car_pos, 1/car_vel]


# render = False
render = True
eps_to_render = 1

# todo: choose source and target env
source_env_name, target_env_name = 'Acrobot-v1', 'CartPole-v1'
# source_env_name, target_env_name = 'CartPole-v1', 'MountainCarContinuous-v0'

env_params = {
    'CartPole-v1': {
        'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.01,
        'action_space': 'discrete', 'use_memory': False, 'unfreeze_at_ep': 2
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
state_translations = {
    'CartPole-v1>Acrobot-v1': translate_cartpole_to_acrobot,
    'MountainCarContinuous-v0>CartPole-v1': translate_mountaincar_to_cartpole,
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

agent = ActorCritic(target_env_name, target_env, global_state_size, global_action_size, render=render,
                    eps_to_render=eps_to_render, saved_path=saved_path,**mixed_params)
results = agent.train(fine_tune=True)
task_name = f'{source_env_name} to {target_env_name}'

dir_name = 'train history section 2'
if not os.path.exists(f'{dir_name}'):
    os.makedirs(f'{dir_name}')

# plot
avg_rewards = results[0]
plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
plt.xlabel('episode')
plt.ylabel('last 100 eps. average reward')
plt.savefig(f'{dir_name}/{task_name}.png', bbox_inches='tight')
plt.show()

# save to csv
solving_ep = results[-2]
time_to_solve = results[-1]
with open(f'{dir_name}/{task_name}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['episode', 'avg 100 rewards', 'avg 100 policy loss',
              'avg 100 baseline loss', 'time to solve']
    writer.writerow(header)
    episode = 0
    for reward, policy_loss, baseline_loss in zip(*results[:-2]):
        episode += 1
        row = [episode, reward, policy_loss, baseline_loss, time_to_solve]
        writer.writerow(row)
