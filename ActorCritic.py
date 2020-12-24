import gym
import numpy as np
import tensorflow as tf
from Models import NeuralNetwork
from time import time
import matplotlib.pyplot as plt
from collections import deque


class ActorCritic:

    def __init__(self, env, state_size, action_size, continuous, policy_hidden_layers, baseline_hidden_layers,
                 policy_net=None, baseline_net=None, policy_learning_rate=0.0004, baseline_learning_rate=0.01,
                 discount_factor=0.99, max_episodes=1000, memory_size=5000, n_to_activate_memory=5, n_to_use_memory=2,
                 n_fails_to_reset_memory=3, use_memory=True,
                 stop_at_solved=False, verbose=True, render=False, eps_to_render=10):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.continuous = continuous
        self.policy_hidden_layers = policy_hidden_layers
        self.baseline_hidden_layers = baseline_hidden_layers
        self.policy_lr = policy_learning_rate
        self.baseline_lr = baseline_learning_rate
        self.discount_factor = discount_factor
        self.max_episodes = max_episodes
        self.use_memory = use_memory
        self.stop_at_solved = stop_at_solved
        self.verbose = verbose
        self.memory_size = memory_size
        self.n_to_activate_memory = n_to_activate_memory
        self.n_to_use_memory = n_to_use_memory
        self.n_fails_to_reset_memory = n_fails_to_reset_memory
        self.render = render
        self.eps_to_render = eps_to_render
        self.policy_net = policy_net
        self.baseline_net = baseline_net

        # derived
        self.max_steps = env.spec.max_episode_steps + 1
        self.reward_threshold = env.spec.reward_threshold
        self.env_state_size = env.observation_space.shape[0]
        if self.continuous:
            self.env_action_size = 2
            self.min_action = env.action_space.low[0]
            self.max_action = env.action_space.high[0]
        else:
            self.env_action_size = env.action_space.n
        self.state_size_delta = self.state_size - self.env_state_size
        self.action_size_delta = self.action_size - self.env_action_size

    def train(self, epsilon=0.0000001):
        """
        Train the agent with the Advantage Actor-Critic algorithm.
        :return: [average_rewards_total, average_policy_losses_total, average_baseline_losses_total, time_to_solve]
                 where "average" is the moving average of the last 100 episodes and the "losses" are averaged across
                 each episode's steps
        """
        np.random.seed(1)

        memory = deque(maxlen=self.memory_size)
        if self.use_memory:
            memory_activated = False
            n_got_great_reward = 0
            n_got_less_than_great_reward = 0
            n_memory_failed = 0

        # Initialize the policy network
        if self.policy_net is None:
            tf.compat.v1.reset_default_graph()
            self.policy_net = NeuralNetwork(self.state_size, self.action_size, self.policy_lr,
                                            self.policy_hidden_layers, 'policy_net')
            self.baseline_net = NeuralNetwork(self.state_size, 1, self.baseline_lr,
                                              self.baseline_hidden_layers, 'baseline_net')
            self.policy_net.set_policy_loss().set_train_step()
            self.baseline_net.set_baseline_loss().set_train_step()

        time_to_solve = None
        start_time = int(round(time() * 1000))

        # Start training the agent with REINFORCE algorithm
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            solved = False
            episode_rewards = np.zeros(self.max_episodes)
            avg_ep_policy_losses = np.zeros(self.max_episodes)
            avg_ep_baseline_losses = np.zeros(self.max_episodes)
            average_rewards = 0.0

            average_rewards_total = []
            average_policy_losses_total = []
            average_baseline_losses_total = []

            for episode in range(self.max_episodes):
                state = self.env.reset()
                state = np.concatenate([state, np.zeros(self.state_size_delta)])
                state = state.reshape([1, self.state_size])

                episode_policy_losses = []
                episode_baseline_losses = []

                td_errors = []

                for step in range(self.max_steps):
                    actions_distribution = sess.run(self.policy_net.actions_distribution,
                                                    {self.policy_net.state: state})[:self.env_action_size]
                    actions_distribution = actions_distribution / (np.sum(actions_distribution) + epsilon)
                    if self.continuous:  # normalize first action from [0,1] to the env's range
                        action = [self.min_action + (self.max_action - self.min_action) * actions_distribution[0]]
                        action_vector = np.concatenate([actions_distribution, np.zeros(self.action_size_delta)])
                    else:
                        action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                        action_vector = np.zeros(self.action_size)
                        action_vector[action] = 1
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.concatenate([next_state, np.zeros(self.state_size_delta)])
                    next_state = next_state.reshape([1, self.state_size])

                    if self.render and episode % self.eps_to_render == 0:
                        self.env.render()

                    # Compute TD-error and update the network's weights
                    if done:
                        baseline_next = 0
                    else:
                        baseline_next = sess.run(self.baseline_net.output, {self.baseline_net.state: next_state})
                    baseline_target = reward + self.discount_factor * baseline_next

                    baseline_current = sess.run(self.baseline_net.output, {self.baseline_net.state: state})
                    feed_dict = {self.baseline_net.state: state, self.baseline_net.target: baseline_target}
                    _, baseline_loss = sess.run([self.baseline_net.train_step, self.baseline_net.loss], feed_dict)

                    td_error = baseline_target - baseline_current
                    td_errors.append(float(td_error))

                    feed_dict = {self.policy_net.state: state, self.policy_net.target: td_error,
                                 self.policy_net.action: action_vector}
                    _, policy_loss = sess.run([self.policy_net.train_step, self.policy_net.loss], feed_dict)

                    if self.use_memory and not memory_activated:
                        memory.append((state, baseline_target, td_error, action_vector))

                    episode_rewards[episode] += reward
                    episode_policy_losses.append(policy_loss)
                    episode_baseline_losses.append(baseline_loss)

                    if done:
                        average_rewards_total.append(np.mean(episode_rewards[max(0, episode - 99):episode + 1]))
                        avg_ep_policy_losses[episode] = np.mean(episode_policy_losses)
                        avg_ep_baseline_losses[episode] = np.mean(episode_baseline_losses)
                        average_policy_losses_total.append(
                            np.mean(avg_ep_policy_losses[max(0, episode - 99):episode + 1]))
                        average_baseline_losses_total.append(
                            np.mean(avg_ep_baseline_losses[max(0, episode - 99):episode + 1]))
                        if episode > 98:
                            average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                        if average_rewards > self.reward_threshold:
                            solved = True
                        break

                    state = next_state

                if self.verbose:
                    print("\tep: %d \t reward: %.1f \t(100mean: %.1f) \tbaseline_loss: %.2f \t(100mean: %.1f)" % (
                        episode, episode_rewards[episode], float(average_rewards_total[episode]),
                        avg_ep_baseline_losses[episode], float(average_baseline_losses_total[-1])))

                if self.use_memory:
                    # MEMORY REPLAY BEHAVIOR:
                    if episode_rewards[episode] > self.reward_threshold:
                        n_got_great_reward += 1
                        n_got_less_than_great_reward = 0
                        n_memory_failed = 0
                    else:
                        n_got_great_reward = 0
                        n_got_less_than_great_reward += 1
                    if n_got_great_reward >= self.n_to_activate_memory and not memory_activated:
                        if self.verbose:
                            print('\tMEMORY ACTIVATED')
                        memory_activated = True
                    if memory_activated and n_got_less_than_great_reward >= self.n_to_use_memory:
                        if self.verbose:
                            print('\tMEMORY USED')
                        n_got_less_than_great_reward = 0
                        n_memory_failed += 1
                        for state, baseline_target, td_error, action in memory:
                            sess.run(self.baseline_net.train_step,
                                     {self.baseline_net.state: state, self.baseline_net.target: baseline_target})
                            feed_dict = {self.policy_net.state: state, self.policy_net.target: td_error,
                                         self.policy_net.action: action}
                            sess.run(self.policy_net.train_step, feed_dict)
                    if memory_activated and n_memory_failed > self.n_fails_to_reset_memory:
                        if self.verbose:
                            print('\tMEMORY RESET')
                        memory_activated = False
                        n_got_great_reward = 0
                        n_got_less_than_great_reward = 0
                        n_memory_failed = 0

                if solved:
                    if time_to_solve is None:
                        time_to_solve = (round(time() * 1000) - start_time) / 1000
                    if self.stop_at_solved:
                        break

        if time_to_solve is None:
            time_to_solve = (round(time() * 1000) - start_time) / 1000

        return average_rewards_total, average_policy_losses_total, average_baseline_losses_total, time_to_solve


if __name__ == "__main__":
    # todo: choose env
    env_name = 'CartPole-v1'  # state_size=4, action_size=2
    # env_name = 'Acrobot-v1'  # state_size=6, action_size=3
    # env_name = 'MountainCarContinuous-v0'  # state_size=2, action_size=continuous

    render = False
    params = {
        'CartPole-v1': {
            'policy_hidden_layers': [12], 'baseline_hidden_layers': [12],
            'policy_learning_rate': 0.0004, 'baseline_learning_rate': 0.01,
            'continuous': False, 'use_memory': True},
        'Acrobot-v1': {
            'policy_hidden_layers': [12], 'baseline_hidden_layers': [12],
            'policy_learning_rate': 0.0004, 'baseline_learning_rate': 0.001,
            'continuous': False, 'use_memory': False},
        'MountainCarContinuous-v0': {
            'policy_hidden_layers': [12], 'baseline_hidden_layers': [12],
            'policy_learning_rate': 0.001, 'baseline_learning_rate': 0.01,
            'continuous': True, 'use_memory': False},
    }
    global_state_size, global_action_size = 4, 2
    env = gym.make(env_name)
    max_steps = env.spec.max_episode_steps + 1
    reward_threshold = env.spec.reward_threshold
    print('env: %s, max_steps=%s, reward_threshold=%s\n' % (env_name, max_steps, reward_threshold))
    agent = ActorCritic(env, global_state_size, global_action_size, render=render, **params[env_name])
    result = agent.train()
    avg_rewards = result[0]
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
    plt.xlabel('episode')
    plt.ylabel('last 100 eps. average reward')
    plt.show()
