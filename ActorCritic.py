import gym
import numpy as np
import tensorflow as tf
from Models import NeuralNetwork
from time import time
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import StandardScaler
import csv
import os
import pandas as pd


class ActorCritic:

    def __init__(self, name, env, state_size, action_size, action_space, policy_hidden_layers, baseline_hidden_layers,
                 policy_net=None, baseline_net=None, policy_lr=0.0004, baseline_lr=0.01, discount_factor=0.99,
                 max_episodes=1000, memory_size=5000, n_to_activate_memory=5, n_to_use_memory=2,
                 n_fails_to_reset_memory=3, use_memory=True, stop_at_solved=True, verbose=True, render=False,
                 eps_to_render=10, saved_path=None, state_translation=None, unfreeze_at_ep=-1):
        self.name = name
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.policy_hidden_layers = policy_hidden_layers
        self.baseline_hidden_layers = baseline_hidden_layers
        self.policy_net = policy_net
        self.baseline_net = baseline_net
        self.policy_lr = policy_lr
        self.baseline_lr = baseline_lr
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
        self.saved_path = saved_path
        self.state_translation = state_translation
        self.unfreeze_at_ep = unfreeze_at_ep

        # derived
        self.max_steps = env.spec.max_episode_steps + 1
        self.reward_threshold = env.spec.reward_threshold
        self.env_state_size = env.observation_space.shape[0]
        if self.action_space == 'discrete':
            self.env_action_size = env.action_space.n
        elif self.action_space == 'continuous':
            self.env_action_size = 2
            self.min_action = env.action_space.low[0]
            self.max_action = env.action_space.high[0]
            # fit normal scaler
            self.scaler = StandardScaler()
            samples = np.array([env.observation_space.sample() for _ in range(10000)])
            self.scaler.fit(samples)

    def train(self, epsilon=0.0000001, fine_tune=False, save_hist=False, fit=True):
        """
        Train the agent with the Advantage Actor-Critic algorithm.
        :return: [average_rewards_total, average_policy_losses_total, average_baseline_losses_total, time_to_solve]
                 where "average" is the moving average of the last 100 episodes and the "losses" are averaged across
                 each episode's steps
        """
        np.random.seed(1)
        tf.compat.v1.reset_default_graph()

        if not fit:
            self.use_memory = False

        with tf.compat.v1.Session() as sess:

            memory = None
            if self.use_memory:
                memory = deque(maxlen=self.memory_size)
                memory_activated = False
                n_got_great_reward = 0
                n_got_less_than_great_reward = 0
                n_memory_failed = 0

            self.policy_net = NeuralNetwork(self.state_size, self.env_action_size, self.policy_lr,
                                            self.policy_hidden_layers, 'policy_net')
            self.baseline_net = NeuralNetwork(self.state_size, 1, self.baseline_lr,
                                              self.baseline_hidden_layers, 'baseline_net')
            # prepare fine tuning:
            prefix = ''
            trainable_layers = None  # train all layers
            if fine_tune:
                prefix = 'target_'  # to avoid loading a pre-trained output layer
                # trainable_layers = 1  # freeze all except output layer
                var_list = tf.compat.v1.global_variables()  # load only non-output vars

            self.policy_net.set_output(self.env_action_size, self.action_space, prefix=prefix).set_policy_loss(
                self.action_space).set_train_step(trainable_layers)
            self.baseline_net.set_output(prefix=prefix).set_baseline_loss().set_train_step(trainable_layers)
            # self.baseline_net.set_output(prefix=prefix).set_baseline_loss().set_train_step()

            if not fine_tune:  # load all vars
                var_list = tf.compat.v1.global_variables()

            sess.run(tf.compat.v1.global_variables_initializer())

            if self.saved_path is not None:
                saver = tf.compat.v1.train.Saver(var_list=var_list)
                saver.restore(sess, self.saved_path)

            solving_ep, time_to_solve = None, None
            start_time = int(round(time() * 1000))

            solved = False
            episode_rewards = np.zeros(self.max_episodes)
            avg_ep_policy_losses = np.zeros(self.max_episodes)
            avg_ep_baseline_losses = np.zeros(self.max_episodes)

            avg_rewards_total = []
            avg_policy_losses_total = []
            avg_baseline_losses_total = []

            if save_hist:
                columns = ['ep', 'step'] + ['s%s' % i for i in range(self.state_size)] + ['action']
                pd.DataFrame(columns=columns).to_csv('hist_%s.csv' % self.name, index=False)

            for episode in range(self.max_episodes):

                # if episode == self.unfreeze_at_ep:  # unfreeze all layers
                #     print('\tUNFREEZE ALL LAYERS')
                #     self.policy_net.set_train_step()
                #     self.baseline_net.set_train_step()

                state = self.process_state(self.env.reset())
                episode_policy_losses = []
                episode_baseline_losses = []

                td_errors = []

                rows = []

                for step in range(self.max_steps):
                    action_vector = np.zeros(self.env_action_size)
                    feed_dict = {self.policy_net.state: state}
                    if self.action_space == 'discrete':
                        actions_distribution = sess.run(self.policy_net.actions_distribution, feed_dict)
                        actions_distribution = actions_distribution[:self.env_action_size]
                        actions_distribution = actions_distribution / (np.sum(actions_distribution) + epsilon)
                        action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                        action_vector[action] = 1
                    elif self.action_space == 'continuous':
                        action = sess.run(self.policy_net.sampled_action, feed_dict)
                        action = max(self.min_action, min(self.max_action, np.squeeze(action)))  # bound action
                        action_vector[0] = action
                        action = [action]

                    # print(f's={list(state[0])} a={action}')
                    if save_hist:
                        rows.append([episode, step] + list(state.reshape(-1)) + [action])

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.process_state(next_state)

                    if self.render and episode % self.eps_to_render == 0:
                        self.env.render()

                    if fit:
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
                    else:
                        baseline_loss, policy_loss = 0, 0

                    if self.use_memory and not memory_activated:
                        memory.append((state, baseline_target, td_error, action_vector))

                    episode_rewards[episode] += reward
                    episode_policy_losses.append(policy_loss)
                    episode_baseline_losses.append(baseline_loss)

                    if done:
                        avg_rewards_total.append(np.mean(episode_rewards[max(0, episode - 99):episode + 1]))
                        avg_ep_policy_losses[episode] = np.mean(episode_policy_losses)
                        avg_ep_baseline_losses[episode] = np.mean(episode_baseline_losses)
                        avg_policy_losses_total.append(
                            np.mean(avg_ep_policy_losses[max(0, episode - 99):episode + 1]))
                        avg_baseline_losses_total.append(
                            np.mean(avg_ep_baseline_losses[max(0, episode - 99):episode + 1]))
                        if episode > 98:
                            average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                            if average_rewards > self.reward_threshold:
                                solved = True
                        if save_hist:
                            pd.DataFrame(rows).to_csv('hist_%s.csv' % self.name, index=False, mode='a', header=False)
                        break

                    state = next_state

                if self.verbose:
                    print("\tep: %d \t reward: %.1f \t(100mean: %.1f) \tbaseline_loss: %.2f \t(100mean: %.1f)" % (
                        episode, episode_rewards[episode], float(avg_rewards_total[episode]),
                        avg_ep_baseline_losses[episode], float(avg_baseline_losses_total[-1])))

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
                        print('SOLVED!!!')
                        time_to_solve = (round(time() * 1000) - start_time) / 1000
                        solving_ep = self.max_episodes
                        if self.saved_path is None and not fine_tune:
                            saver = tf.compat.v1.train.Saver()
                            save_path = saver.save(sess, 'saved_models/%s/model' % self.name)
                            print("Model saved in path: %s" % save_path)

                    if self.stop_at_solved:
                        break

        if time_to_solve is None:
            time_to_solve = (round(time() * 1000) - start_time) / 1000
            solving_ep = episode

        return avg_rewards_total, avg_policy_losses_total, avg_baseline_losses_total, solving_ep, time_to_solve

    def process_state(self, state):
        if self.action_space == 'continuous':
            state = self.scaler.transform([state]).squeeze()
        if self.state_translation is not None:
            state = self.state_translation(state)
        state = np.concatenate([state, np.zeros(self.state_size - len(state))])
        return state.reshape(1, -1)


if __name__ == "__main__":
    # todo: choose env
    # env_name = 'CartPole-v1'  # state_size=4, action_size=2
    # env_name = 'Acrobot-v1'  # state_size=6, action_size=3
    env_name = 'MountainCarContinuous-v0'  # state_size=2, action_size=continuous

    # saved_path = None
    saved_path = 'saved_models/%s/model' % env_name
    params = {
        'CartPole-v1': {
            'policy_hidden_layers': [12], 'baseline_hidden_layers': [12], 'policy_lr': 0.0004, 'baseline_lr': 0.01,
            'action_space': 'discrete', 'use_memory': True,
        },
        'Acrobot-v1': {
            'policy_hidden_layers': [12], 'baseline_hidden_layers': [12],
            # 'policy_hidden_layers': [12, 12, 3], 'baseline_hidden_layers': [12, 12, 3],
            'policy_lr': 0.0004, 'baseline_lr': 0.001,
            'action_space': 'discrete', 'use_memory': False,
        },
        'MountainCarContinuous-v0': {
            'policy_hidden_layers': [64], 'baseline_hidden_layers': [512], 'policy_lr': 0.0001, 'baseline_lr': 0.01,
            'action_space': 'continuous', 'use_memory': False,
        },
    }
    global_state_size, global_action_size = 6, 3
    env = gym.make(env_name)
    max_steps = env.spec.max_episode_steps + 1
    reward_threshold = env.spec.reward_threshold
    print('env: %s, max_steps=%s, reward_threshold=%s\n' % (env_name, max_steps, reward_threshold))
    agent = ActorCritic(env_name, env, global_state_size, global_action_size, render=False,
                        eps_to_render=1, saved_path=saved_path, stop_at_solved=False,
                        **params[env_name])
    results = agent.train(save_hist=True, fit=False)
