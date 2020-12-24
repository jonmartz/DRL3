import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class NeuralNetwork:
    """
    Network for predicting the best action given a state.
    """

    def __init__(self, state_size, output_size, learning_rate, hidden_layers, name):
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_size], name="state")
            self.target = tf.compat.v1.placeholder(tf.compat.v1.float32, name="target")
            self.loss, self.optimizer, self.train_step = None, None, None

            # possibly unused
            self.action = tf.compat.v1.placeholder(tf.compat.v1.int32, [self.output_size], name="action")
            self.actions_distribution = None

            self.Ws, self.bs = [], []

            prev_layer_size, self.output = self.state_size, self.state
            for i, layer_size in enumerate(hidden_layers + [self.output_size]):
                W = tf.compat.v1.get_variable("W%d" % (i + 1), [prev_layer_size, layer_size],
                                              initializer=tf.compat.v1.initializers.glorot_uniform(seed=0))
                b = tf.compat.v1.get_variable("b%d" % (i + 1), [layer_size],
                                              initializer=tf.compat.v1.zeros_initializer())
                self.Ws.append(W)
                self.bs.append(b)
                self.output = tf.compat.v1.add(tf.compat.v1.matmul(self.output, W), b)
                prev_layer_size = layer_size
                if i < len(hidden_layers):  # skip activation in last output layer
                    self.output = tf.compat.v1.nn.relu(self.output)

            # # Softmax probability distribution over actions
            # self.actions_distribution = tf.compat.v1.squeeze(tf.compat.v1.nn.softmax(self.output))
            # self.loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            # self.loss = tf.compat.v1.reduce_mean(self.loss * self.target)
            # self.train_step = self.optimizer.minimize(self.loss)

    def set_baseline_loss(self):
        with tf.compat.v1.variable_scope(self.name, reuse=True):
            self.loss = tf.compat.v1.math.squared_difference(self.output, self.target)
        return self

    def set_policy_loss(self, action_space='discrete'):
        with tf.compat.v1.variable_scope(self.name, reuse=True):
            if action_space == 'discrete':
                self.actions_distribution = tf.compat.v1.squeeze(tf.compat.v1.nn.softmax(self.output))
                self.loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.action)
            elif action_space == 'continuous':
                pass
        return self

    def set_train_step(self, trainable_layers=None):
        with tf.compat.v1.variable_scope(self.name, reuse=True):
            if self.optimizer is None:
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            if trainable_layers is None:
                var_list = None
            else:  # train only top <trainable_layers> layers
                var_list = self.Ws[-trainable_layers:] + self.bs[-trainable_layers:]
            self.train_step = self.optimizer.minimize(self.loss, var_list=var_list)


# class BaselineNetwork:
#     """
#     Network for predicting the target value given a state. ReLu is applied to the output to avoid silly
#     negative predictions, which is applicable in this problem (cartpole).
#     """
#
#     def __init__(self, state_size, learning_rate, hidden_layers, name='baseline_network'):
#         self.state_size = state_size
#         self.learning_rate = learning_rate
#
#         with tf.compat.v1.variable_scope(name):
#             self.state = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_size], name="state")
#             self.target = tf.compat.v1.placeholder(tf.compat.v1.float32, name="target")
#
#             prev_layer_size, self.output = self.state_size, self.state
#             for i, layer_size in enumerate(hidden_layers + [1]):
#                 W = tf.compat.v1.get_variable("W%d" % (i + 1), [prev_layer_size, layer_size],
#                                               initializer=tf.compat.v1.initializers.glorot_uniform(seed=0))
#                 b = tf.compat.v1.get_variable("b%d" % (i + 1), [layer_size],
#                                               initializer=tf.compat.v1.zeros_initializer())
#                 self.output = tf.compat.v1.add(tf.compat.v1.matmul(self.output, W), b)
#                 prev_layer_size = layer_size
#                 if i < len(hidden_layers):  # skip activation in last output layer
#                     self.output = tf.compat.v1.nn.relu(self.output)
#
#             self.loss = tf.compat.v1.math.squared_difference(self.output, self.target)
#             self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
