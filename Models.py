import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class NeuralNetwork:
    """
    Network for predicting the best action given a state.
    """

    def __init__(self, state_size, output_size, learning_rate, hidden_layer_sizes, name, source_nets=None,
                 optimizer=tf.compat.v1.train.AdamOptimizer):
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.name = name
        self.source_nets = source_nets

        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            self.scope = tf.compat.v1.get_variable_scope()
            self.state = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_size], name='state')
            self.target = tf.compat.v1.placeholder(tf.compat.v1.float32, name='target')
            self.optimizer = optimizer(learning_rate=self.learning_rate)
            self.output, self.loss, self.train_step = None, None, None

            # possibly unused
            self.action = tf.compat.v1.placeholder(tf.compat.v1.float32, [self.output_size], name='action')
            self.actions_distribution, self.sampled_action = None, None

            self.Ws, self.bs, self.Zs, self.As = [], [], [], []
            if source_nets is not None:
                self.lateral_Ws, self.lateral_bs = [], []

            prev_layer_size, A = self.state_size, self.state
            for i, layer_size in enumerate(hidden_layer_sizes):
                W = tf.compat.v1.get_variable('W%d' % i, [prev_layer_size, layer_size],
                                              initializer=tf.compat.v1.initializers.glorot_uniform())
                b = tf.compat.v1.get_variable('b%d' % i, [layer_size],
                                              initializer=tf.compat.v1.zeros_initializer())
                Z = tf.compat.v1.add(tf.compat.v1.matmul(A, W), b)

                if source_nets is not None:  # add hidden output of source nets before activation
                    for j, source_net in enumerate(source_nets):
                        source_Z = source_net.Zs[i]
                        source_layer_size = source_net.hidden_layer_sizes[i]
                        lateral_W = tf.compat.v1.get_variable(f'lateral_{j}_W%d' % i, [source_layer_size, layer_size],
                                                              initializer=tf.compat.v1.initializers.glorot_uniform())
                        lateral_b = tf.compat.v1.get_variable(f'lateral_{j}_b%d' % i, [layer_size],
                                                              initializer=tf.compat.v1.zeros_initializer())
                        lateral_Z = tf.compat.v1.add(tf.compat.v1.matmul(source_Z, lateral_W), lateral_b)
                        Z = tf.compat.v1.add(Z, lateral_Z)

                        self.lateral_Ws.append(lateral_W)
                        self.lateral_bs.append(lateral_b)

                A = tf.compat.v1.nn.relu(Z)
                self.Ws.append(W)
                self.bs.append(b)
                self.Zs.append(Z)
                self.As.append(A)
                prev_layer_size = layer_size

    def set_output(self, output_size=1, action_space='discrete', prefix=''):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
            # with tf.compat.v1.name_scope(self.scope.original_name_scope):
            if self.output is not None:  # changing existing output
                self.Ws = self.Ws[:-1]
                self.bs = self.bs[:-1]
            prev_layer_size = self.hidden_layer_sizes[-1]
            hidden_output = self.As[-1]
            if action_space == 'discrete':
                W = tf.compat.v1.get_variable('%sW_out_%s' % (prefix, self.name), [prev_layer_size, output_size],
                                              initializer=tf.compat.v1.initializers.glorot_uniform())
                b = tf.compat.v1.get_variable('%sb_out_%s' % (prefix, self.name), [output_size],
                                              initializer=tf.compat.v1.zeros_initializer())
                self.output = tf.compat.v1.add(tf.compat.v1.matmul(hidden_output, W), b,
                                               name='%soutput_discrete' % prefix)
            elif action_space == 'continuous':
                W, b, self.output = [], [], []
                for output_name in ['mean', 'stdev']:
                    W.append(tf.compat.v1.get_variable('%sW_%s' % (prefix, output_name), [prev_layer_size, 1],
                                                       initializer=tf.compat.v1.initializers.glorot_uniform()))
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

    def set_train_step(self, trainable_layers=0):
        # with tf.compat.v1.name_scope(self.scope.original_name_scope):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
            # if trainable_layers is None:
            #     var_list = None
            # else:  # train only top <trainable_layers> layers
            # todo: deal with continous action space
            var_list = self.Ws[-trainable_layers:] + self.bs[-trainable_layers:]
            if self.source_nets is not None:
                var_list += self.lateral_Ws + self.lateral_bs
            self.train_step = self.optimizer.minimize(self.loss, var_list=var_list, name='train_step')

    # todo: check if models are actually frozen during progressive network training
