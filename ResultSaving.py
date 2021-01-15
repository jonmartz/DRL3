import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import numpy as np


class ModifiedTensorBoard(TensorBoard):
    """
    Modified tensorboard for displaying values from a csv, for the purpose of making plots
    averaged across many runs of the experiment.
    """
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.model = None
        self.TB_graph = tf.compat.v1.Graph()
        with self.TB_graph.as_default():
            self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=5000)
            self.writer.set_as_default()
            self.all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        self.TB_sess = tf.compat.v1.InteractiveSession(graph=self.TB_graph)
        self.TB_sess.run(self.writer.init())

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = self.log_dir + '\\train'

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # added for performance?
    def on_train_batch_end(self, _, __):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            self.TB_sess.run(self.all_summary_ops)
            if self.model is not None:
                name = f'{name}_{self.model.name}'
            self.TB_sess.run(tf.summary.scalar(name, value, step=index))
        self.model = None


def write_average_results(section, section_id, task):
    """
    For padding the different runs of a task to match the longest run, average results over runs
    and plotting the bar graphs shown in the report.
    :param section: section string
    :param section_id: section id
    :param task: string
    """
    df = pd.read_csv(f'{section}/{task}.csv')
    max_ep = df['episode'].max()
    solving_eps, solving_times = [], []
    rows = []
    for i, df_iteration in df.groupby('iteration'):
        for j, row in df_iteration.iterrows():
            rows.append(row)
        last_row = rows[-1]
        last_ep = last_row['episode']
        solving_eps.append(last_ep)
        solving_times.append(last_row['time to solve'])
        while last_ep < max_ep:
            last_ep += 1
            padded_row = last_row.copy()
            padded_row['episode'] = last_ep
            rows.append(padded_row)
    df_padded = pd.DataFrame(rows, columns=df.columns)
    df_padded.groupby('episode').mean().to_csv(f'{section}/{task}_avg.csv')

    # plot
    avg_rewards = df_padded.groupby('episode').mean()['avg 100 rewards']
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
    plt.xlabel('episode')
    plt.ylabel('last 100 eps. average reward')
    plt.show()

    # add to global statistics
    mode = 'w' if not os.path.exists('solving_stats.csv') else 'a'
    with open('solving_stats.csv', mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            header = ['section id', 'target env', 'solving episode avg', 'solving episode std',
                      'time to solve avg', 'time to solve std']
            writer.writerow(header)
        row = [section_id, task.split(' ')[-1], np.mean(solving_eps), np.std(solving_eps),
               np.mean(solving_times), np.std(solving_times)]
        writer.writerow(row)


def save_results(dir_name, file_name, results, iteration):
    """
    Script for saving results of an experimental run in a csv file.
    :param dir_name: to save in
    :param file_name:  to save as
    :param results: list returned by the ActorCritic.train(...) function
    :param iteration: iteration run
    """
    if not os.path.exists(f'{dir_name}'):
        os.makedirs(f'{dir_name}')

    # save to csv
    time_to_solve = results[-1]
    mode = 'w' if iteration == 0 else 'a'  # only append if iteration is not the first
    with open(f'{dir_name}/{file_name}.csv', mode=mode, newline='') as file:
        writer = csv.writer(file)
        if iteration == 0:
            header = ['iteration', 'episode', 'avg 100 rewards', 'avg 100 policy loss',
                      'avg 100 baseline loss', 'time to solve']
            writer.writerow(header)
        episode = 0
        for reward, policy_loss, baseline_loss in zip(*results[:-2]):
            episode += 1
            row = [iteration, episode, reward, policy_loss, baseline_loss, time_to_solve]
            writer.writerow(row)


def plot_final_comparison():
    """
    After running all the three sections of the assignment, to produce the bar graphs.
    """
    matplotlib.rcParams.update({'font.size': 12})
    df = pd.read_csv('solving_stats.csv')
    for metric in ['solving episode', 'time to solve']:
        for env, df_env in df.groupby('target env'):
            i = 0
            for _, row in df_env.iterrows():
                plt.bar(i, row[f'{metric} avg'], label=f'section {row["section id"]}',
                        yerr=row[f'{metric} std'], error_kw=dict(ecolor='black'))
                i += 1
            plt.xticks(range(3), [f'section {i}' for i in range(1, 4)])
            plt.ylabel(metric)
            plt.savefig(f'{env} {metric}.png', bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    section_id = 1
    tasks = [
        'CartPole-v1',
        'Acrobot-v1',
        'MountainCarContinuous-v0',
    ]

    # section_id = 2
    # tasks = [
    #     # 'Acrobot-v1 to CartPole-v1',
    #     'CartPole-v1 to MountainCarContinuous-v0',
    # ]

    # section_id = 3
    # tasks = [
    #     "['Acrobot-v1', 'MountainCarContinuous-v0'] to CartPole-v1",
    #     "['CartPole-v1', 'Acrobot-v1'] to MountainCarContinuous-v0",
    # ]

    section_dir = f'train history section {section_id}'
    for task in tasks:
        write_average_results(section_dir, section_id, task)
        print('alg: %s' % task)
        tb = ModifiedTensorBoard(log_dir=f'logs/{section_dir}/{task}')
        df = pd.read_csv(f'{section_dir}/{task}_avg.csv')
        for i, row in df.iterrows():
            ep = row['episode']
            print('\t\tep: %d/%d' % (ep, len(df)))
            avg_reward = row['avg 100 rewards']
            loss = row['avg 100 policy loss']
            tb.step = ep
            tb.update_stats(loss=loss, reward_steps=ep, reward_avg100=avg_reward)

    # plot_final_comparison()
