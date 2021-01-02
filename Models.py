import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class NeuralNetwork:
    """
    Network for predicting the best action given a state.
    """

    def __init__(self, state_size, output_size, learning_rate, hidden_layer_sizes, name):
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.name = name

        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            self.scope = tf.compat.v1.get_variable_scope()
            self.state = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_size], name='state')
            self.target = tf.compat.v1.placeholder(tf.compat.v1.float32, name='target')
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.output, self.loss, self.train_step = None, None, None

            # possibly unused
            self.action = tf.compat.v1.placeholder(tf.compat.v1.float32, [self.output_size], name='action')
            self.actions_distribution, self.sampled_action = None, None

            self.Ws, self.bs, self.hidden_outputs = [], [], []

            prev_layer_size, hidden_output = self.state_size, self.state
            for i, layer_size in enumerate(hidden_layer_sizes):
                W = tf.compat.v1.get_variable('W%d' % i, [prev_layer_size, layer_size],
                                              initializer=tf.compat.v1.initializers.glorot_uniform(seed=0))
                b = tf.compat.v1.get_variable('b%d' % i, [layer_size],
                                              initializer=tf.compat.v1.zeros_initializer())
                hidden_output = tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W), b)
                hidden_output = tf.compat.v1.nn.relu(hidden_output)
                self.Ws.append(W)
                self.bs.append(b)
                self.hidden_outputs.append(hidden_output)
                prev_layer_size = layer_size

    def set_output(self, output_size=1, action_space='discrete', prefix=''):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
        # with tf.compat.v1.name_scope(self.scope.original_name_scope):
            if self.output is not None:  # changing existing output
                self.Ws = self.Ws[:-1]
                self.bs = self.bs[:-1]
            prev_layer_size = self.hidden_layer_sizes[-1]
            hidden_output = self.hidden_outputs[-1]
            if action_space == 'discrete':
                W = tf.compat.v1.get_variable('%sW_out_%s' % (prefix, self.name), [prev_layer_size, output_size],
                                              initializer=tf.compat.v1.initializers.glorot_uniform(seed=0))
                b = tf.compat.v1.get_variable('%sb_out_%s' % (prefix, self.name), [output_size],
                                              initializer=tf.compat.v1.zeros_initializer())
                self.output = tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W), b,
                                               name='%soutput_discrete' % prefix)
            elif action_space == 'continuous':
                W, b, self.output = [], [], []
                for output_name in ['mean', 'stdev']:
                    W.append(tf.compat.v1.get_variable('%sW_%s' % (prefix, output_name), [prev_layer_size, 1],
                                                       initializer=tf.compat.v1.initializers.glorot_uniform(seed=0)))
                    b.append(tf.compat.v1.get_variable('%sb_%s' % (prefix, output_name), [1],
                                                       initializer=tf.compat.v1.zeros_initializer()))
                    self.output.append(tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W[-1]), b[-1],
                                                        name='%soutput_%s' % (prefix, output_name)))
            self.Ws.append(W)
            self.bs.append(b)
        return self

    def set_baseline_loss(self):
        # with tf.compat.v1.name_scope(self.scope.original_name_scope):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.squeeze(self.output)
            self.loss = tf.compat.v1.math.squared_difference(v, self.target)
            self.loss = tf.compat.v1.reduce_mean(self.loss, name='loss')
        return self

    def set_policy_loss(self, action_space='discrete', epsilon=0.0000001):
        # with tf.compat.v1.name_scope(self.scope.original_name_scope):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
            if action_space == 'discrete':
                self.actions_distribution = tf.compat.v1.squeeze(tf.compat.v1.nn.softmax(self.output),
                                                                 name='actions_distribution')
                self.loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.action)
                self.loss = tf.compat.v1.reduce_mean(self.loss * self.target, name='loss')
            elif action_space == 'continuous':
                mean, stdev = self.output[0], self.output[1]
                stdev = tf.compat.v1.nn.softplus(stdev) + epsilon
                self.actions_distribution = tf.compat.v1.distributions.Normal(mean, stdev, name='actions_distribution')
                self.sampled_action = tf.compat.v1.squeeze(self.actions_distribution.sample(1), axis=0,
                                                           name='sampled_action')
                self.loss = tf.multiply(-tf.compat.v1.log(self.actions_distribution.prob(self.action) + epsilon),
                                        self.target, name='loss')
        return self

    def set_train_step(self, trainable_layers=None):
        # with tf.compat.v1.name_scope(self.scope.original_name_scope):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
            if trainable_layers is None:
                var_list = None
            else:  # train only top <trainable_layers> layers
                # todo: deal with continous action space
                var_list = self.Ws[-trainable_layers:] + self.bs[-trainable_layers:]
            self.train_step = self.optimizer.minimize(self.loss, var_list=var_list, name='train_step')


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
