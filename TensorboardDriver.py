"""
Run this script after running GatheringResults.py to produce the plots using tensorboard.
We don't run tensorboard during GatheringResults.py because we wanted plots that are averaged
across experimental runs and tensorboard is not capable of this.
"""

import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard


class ModifiedTensorBoard(TensorBoard):

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


def write_average_results():
    df = pd.read_csv('results.csv')
    df_algs = df.groupby('algorithm')
    for alg in envs:
        df_alg = df_algs.get_group(alg)
        df_alg.groupby('episode').mean().to_csv('avg_results_%s.csv' % alg)


# section = 'train history section 1'
# envs = [
#     'CartPole-v1',
#     'Acrobot-v1',
#     'MountainCarContinuous-v0',
# ]

section = 'train history section 2'
envs = [
    'Acrobot-v1 to CartPole-v1',
    'CartPole-v1 to MountainCarContinuous-v0',
]

# write_average_results()
for env in envs:
    print('alg: %s' % env)
    tb = ModifiedTensorBoard(log_dir=f'logs/{section}/{env}')
    df = pd.read_csv(f'{section}/{env}.csv')
    for i, row in df.iterrows():
        ep = row['episode']
        print('\t\tep: %d/%d' % (ep, len(df)))
        avg_reward = row['avg 100 rewards']
        loss = row['avg 100 policy loss']
        tb.step = ep
        tb.update_stats(loss=loss, reward_steps=ep, reward_avg100=avg_reward)
