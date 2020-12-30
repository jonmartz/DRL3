import gym
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import csv


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
        'policy_hidden_layers': [64], 'baseline_hidden_layers': [512], 'policy_lr': 0.0001, 'baseline_lr': 0.01,
        'action_space': 'continuous', 'use_memory': False},
}
global_state_size, global_action_size = 6, 3
env = gym.make(env_name)
max_steps = env.spec.max_episode_steps + 1
reward_threshold = env.spec.reward_threshold
print('env: %s, max_steps=%s, reward_threshold=%s\n' % (env_name, max_steps, reward_threshold))
agent = ActorCritic(env_name, env, global_state_size, global_action_size,
                    render=render, eps_to_render=eps_to_render, **params[env_name])
results = agent.train()

# plot
avg_rewards = results[0]
plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
plt.xlabel('episode')
plt.ylabel('last 100 eps. average reward')
plt.savefig('train history/%s.png' % env_name, bbox_inches='tight')
plt.show()

# save to csv
solving_ep = results[-2]
time_to_solve = results[-1]
with open('train history/%s.csv' % env_name, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['episode', 'avg 100 rewards', 'avg 100 policy loss',
              'avg 100 baseline loss', 'time to solve']
    writer.writerow(header)
    episode = 0
    for reward, policy_loss, baseline_loss in zip(*results[:-2]):
        episode += 1
        row = [episode, reward, policy_loss, baseline_loss, time_to_solve]
        writer.writerow(row)