import tensorflow as tf
from keras.layers import Dense, Input, Softmax
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.optimizers import Adam
from keras.models import Model, load_model
import keras.backend as K
# import tensorflow_probability as tfp
from keras.callbacks import ModelCheckpoint


class NeuralNetwork:
    """
    Network for predicting the best action given a state.
    """

    def __init__(self, name, state_size, output_size, hidden_layer_sizes):

        self.name = name
        self.state = Input(state_size, name='state')
        self.target = Input(1, name='target')
        self.action = Input(output_size, name='action')
        self.output_size = output_size
        self.output, self.actions_distribution = None, None

        self.hidden_layers = []
        last_layer = self.state
        for hidden_layer_size in hidden_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu')(last_layer)
            self.hidden_layers.append(hidden_layer)
            last_layer = hidden_layer

        # # TFv1:
        # self.state_size = state_size
        # self.output_size = output_size
        # self.learning_rate = learning_rate
        # self.hidden_layer_sizes = hidden_layer_sizes
        # self.name = name
        # self.graph = graph
        # self.fine_tune = fine_tune
        #
        # with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        #     self.scope = tf.compat.v1.get_variable_scope()
        #     self.state = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_size], name='state')
        #     self.target = tf.compat.v1.placeholder(tf.compat.v1.float32, name='target')
        #     self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        #     self.output, self.loss, self.train_step = None, None, None
        #
        #     # possibly unused
        #     self.action = tf.compat.v1.placeholder(tf.compat.v1.float32, [self.output_size], name='action')
        #     self.actions_distribution, self.sampled_action = None, None
        #
        #     self.Ws, self.bs, self.hidden_outputs = [], [], []
        #
        #     prev_layer_size, hidden_output = self.state_size, self.state
        #     # for i, layer_size in enumerate(hidden_layer_sizes + [self.output_size]):
        #     for i, layer_size in enumerate(hidden_layer_sizes):
        #         if graph is None:
        #             W = tf.compat.v1.get_variable('W%d' % i, [prev_layer_size, layer_size],
        #                                           initializer=tf.compat.v1.initializers.glorot_uniform(seed=0))
        #             b = tf.compat.v1.get_variable('b%d' % i, [layer_size],
        #                                           initializer=tf.compat.v1.zeros_initializer())
        #         else:
        #             W = [var for var in tf.compat.v1.global_variables() if var.op.name == '%s/W%d' % (name, i)][0]
        #             b = [var for var in tf.compat.v1.global_variables() if var.op.name == '%s/b%d' % (name, i)][0]
        #         hidden_output = tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W), b)
        #         # if i < len(hidden_layer_sizes):  # skip activation in last output layer
        #         #     hidden_output = tf.compat.v1.nn.relu(hidden_output)
        #         hidden_output = tf.compat.v1.nn.relu(hidden_output)
        #         self.Ws.append(W)
        #         self.bs.append(b)
        #         self.hidden_outputs.append(hidden_output)
        #         prev_layer_size = layer_size
        #     # self.output = self.outputs[-1]

    # def set_output(self, action_space='discrete'):
    def set_output(self, output_size=1, activation=None):
        self.output = Dense(output_size, activation=activation)(self.hidden_layers[-1])
        return self

        # if action_space == 'discrete':
        #     self.output = Dense(self.output_size)(self.hidden_layers[-1])
        # elif action_space == 'continuous':
        #     self.output = [Dense(1)(self.hidden_layers[-1]), Dense(1)(self.hidden_layers[-1])]
        # return self

        # # TFv1:
        # with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
        # # with tf.compat.v1.name_scope(self.scope.original_name_scope):
        #     if self.output is not None:  # changing existing output
        #         self.Ws = self.Ws[:-1]
        #         self.bs = self.bs[:-1]
        #     prev_layer_size = self.hidden_layer_sizes[-1]
        #     hidden_output = self.hidden_outputs[-1]
        #     if action_space == 'discrete':
        #         if self.graph is None or self.fine_tune:
        #             W = tf.compat.v1.get_variable('W_out_%s' % self.name, [prev_layer_size, output_size],
        #                                           initializer=tf.compat.v1.initializers.glorot_uniform(seed=0))
        #             b = tf.compat.v1.get_variable('b_out_%s' % self.name, [output_size],
        #                                           initializer=tf.compat.v1.zeros_initializer())
        #         else:
        #             W = [var for var in tf.compat.v1.global_variables() if var.op.name == 'W_out_%s' % self.name][0]
        #             b = [var for var in tf.compat.v1.global_variables() if var.op.name == 'b_out_%s' % self.name][0]
        #             # b = [var for var in tf.compat.v1.global_variables() if var.op.name == '%s/b_out_%s' % (self.name, self.name)][0]
        #             # W = self.graph.get_tensor_by_name('%s/W_out_%s:0' % (self.name, self.name))
        #             # b = self.graph.get_tensor_by_name('%s/b_out_%s:0' % (self.name, self.name))
        #         self.output = tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W), b, name='output_discrete')
        #     elif action_space == 'continuous':
        #         W, b, self.output = [], [], []
        #         for output_name in ['mean', 'stdev']:
        #             W.append(tf.compat.v1.get_variable('W_%s' % output_name, [prev_layer_size, 1],
        #                                                initializer=tf.compat.v1.initializers.glorot_uniform(seed=0)))
        #             b.append(tf.compat.v1.get_variable('b_%s' % output_name, [1],
        #                                                initializer=tf.compat.v1.zeros_initializer()))
        #             self.output.append(tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W[-1]), b[-1],
        #                                                 name='output_%s' % output_name))
        #     self.Ws.append(W)
        #     self.bs.append(b)
        # return self

    # def set_baseline_loss(self):
    #     # TFv1:
    #     with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
    #         v = tf.compat.v1.squeeze(self.output)
    #         self.loss = tf.compat.v1.math.squared_difference(v, self.target)
    #         self.loss = tf.compat.v1.reduce_mean(self.loss, name='loss')
    #     return self
    #
    # def set_policy_loss(self, action_space='discrete', epsilon=0.0000001):
    #     # with tf.compat.v1.name_scope(self.scope.original_name_scope):
    #     with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
    #         if action_space == 'discrete':
    #             self.actions_distribution = tf.compat.v1.squeeze(tf.compat.v1.nn.softmax(self.output),
    #                                                              name='actions_distribution')
    #             self.loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
    #                 logits=self.output, labels=self.action)
    #             self.loss = tf.compat.v1.reduce_mean(self.loss * self.target, name='loss')
    #         elif action_space == 'continuous':
    #             mean, stdev = self.output[0], self.output[1]
    #             stdev = tf.compat.v1.nn.softplus(stdev) + epsilon
    #             self.actions_distribution = tf.compat.v1.distributions.Normal(mean, stdev, name='actions_distribution')
    #             self.sampled_action = tf.compat.v1.squeeze(self.actions_distribution.sample(1), axis=0,
    #                                                        name='sampled_action')
    #             self.loss = tf.multiply(-tf.compat.v1.log(self.actions_distribution.prob(self.action) + epsilon),
    #                                     self.target, name='loss')
    #     return self

    def set_train_step(self, loss_type, lr):
        params = {'optimizer': Adam(lr)}
        if loss_type == 'baseline':
            params['loss'] = MeanSquaredError()
            model = Model(self.state, self.output, name=self.name)
        else:
            if loss_type == 'policy_discrete':
                params['loss'] = CategoricalCrossentropy(from_logits=True)
            elif loss_type == 'policy_continuous':
                mean, stddev = self.output[0], self.output[1]
                stddev = K.softplus(stddev) + 1e-5
                self.actions_distribution = tf.compat.v1.distributions.Normal(mean, stddev, name='actions_distribution')
                params['loss'] = -K.log(self.actions_distribution.prob(self.action) + 1e-5) * self.target
            # params['loss_weights'] = self.target
            model = Model([self.state, self.target], self.output, name=self.name)
        model.compile(**params)
        return model

        # TFv1:
        # with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
        #     if trainable_layers is None:
        #         var_list = None
        #     else:  # train only top <trainable_layers> layers
        #         var_list = self.Ws[-trainable_layers:] + self.bs[-trainable_layers:]
        #     self.train_step = self.optimizer.minimize(self.loss, var_list=var_list, name='train_step')


def loss_policy_discrete(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# class SavedModel:
#     def __init__(self, model_name, graph, learning_rate, action_space='discrete'):
#         # with tf.compat.v1.variable_scope(model_name + , reuse=tf.compat.v1.AUTO_REUSE):
#         self.model_name = model_name
#         self.graph = graph
#
#         self.var_names = [var_name for var_name in graph._names_in_use if model_name in var_name]
#         self.state = self.get_var('state')
#         self.target = self.get_var('target')
#         self.loss = self.get_var('loss')
#         # self.train_step = self.get_var('train_step')
#         if action_space == 'discrete':
#             self.output = self.get_var('output_discrete')
#         elif action_space == 'continuous':
#             self.sampled_action = self.get_var('sampled_action')
#             self.output = [self.get_var('output_mean'), self.get_var('output_stdev')]
#         try:
#             self.action = self.get_var('action')
#             self.actions_distribution = self.get_var('actions_distribution')
#         except:
#             pass
#
#         self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
#         self.train_step = self.optimizer.minimize(self.loss, name='train_step')
#
#     def get_var(self, target_name):
#         var_names = [name for name in self.var_names if target_name in name]
#         var_names = [name for name in var_names if name.count('/') == 1]
#         var_name = var_names[0] + ':0'
#         return self.graph.get_tensor_by_name(var_name)
