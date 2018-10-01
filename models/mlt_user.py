import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score


class MLTUser:
    def __init__(self, options):
        self.batch_size = options.batch_size
        self.max_seq_len = options.seq_len
        self.state_size = options.state_size
        self.cell_type = options.cell_type
        self.learning_rate = options.learning_rate
        self.vocab_size = options.vocab_size
        self.emb_size = options.emb_size
        self.epochs = options.epochs
        self.arch = options.archi
        self.r_label_size = options.rumor_label
        self.s_label_size = options.stance_label
        self.user_feat_size = options.user_feat_size
        self.vae_latent_size = options.vae_latent_size
        self.vae_encoder_size = options.vae_encoder_size
        self.vae_decoder_size = options.vae_decoder_size
        self.reg_coeff = 0.001

        self.display_interval = options.display_epoch
        self.train_performance_interval = options.train_performnace_epoch
        self.test_interval = options.test_epoch
        self.log = logging.getLogger(options.main + '.' + __name__)
        self.log.setLevel(logging.DEBUG)

        self.init_variables()
        self.build_graph()

    def xavier_init(fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    def init_variables(self):
        self.tweet_vec = tf.placeholder(shape=[self.batch_size, self.max_seq_len, self.vocab_size], dtype=tf.float32)
        self.seq_len = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.stance_output = tf.placeholder(shape=[self.batch_size, self.max_seq_len], dtype=tf.float32)
        self.rumor_output = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)
        self.user_feat = tf.placeholder(shape=[self.batch_size, self.max_seq_len, self.user_feat_size], dtype=tf.float32)

        self.vae_weights = {
            "encoder_h1": tf.get_variable('encoder_h1', shape=[self.user_feat_size, self.vae_encoder_size],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            "encoder_mu": tf.get_variable('encoder_mu', shape=[self.vae_encoder_size, self.vae_latent_size],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            "encoder_logvar": tf.get_variable('encoder_logvar', shape=[self.vae_encoder_size, self.vae_latent_size],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            "decoder_h1": tf.get_variable('decoder_h1', shape=[self.vae_latent_size, self.vae_decoder_size],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            "decoder_reconstruction": tf.get_variable('decoder_recog', shape=[self.vae_decoder_size, self.user_feat_size],
                                          initializer=tf.contrib.layers.xavier_initializer())
        }

        # define bias shape
        self.vae_biases = {
            "encoder_h1": tf.Variable(tf.zeros([self.vae_encoder_size])),
            "encoder_mu": tf.Variable(tf.zeros([self.vae_latent_size])),
            "encoder_logvar": tf.Variable(tf.zeros([self.vae_latent_size])),
            "decoder_h1": tf.Variable(tf.zeros([self.vae_decoder_size])),
            "decoder_reconstruction": tf.Variable(tf.zeros([self.user_feat_size])),
        }

    def construct_vae(self):
        with tf.variable_scope('user-vae'):
            l2_loss = tf.constant(0.0)  # l2 norm
            l2_loss += tf.nn.l2_loss(self.vae_weights["encoder_h1"])
            hidden_encoder = tf.nn.relu(tf.add(tf.tensordot(self.user_feat,
                                self.vae_weights["encoder_h1"], axes=[2, 0]), self.vae_biases["encoder_h1"]))

            ## encoder_mu
            l2_loss += tf.nn.l2_loss(self.vae_weights["encoder_mu"])
            mu_encoder = tf.add(tf.tensordot(hidden_encoder, self.vae_weights["encoder_mu"], axes=[2, 0]), self.vae_biases["encoder_mu"])

            ## encoder_logvar
            l2_loss += tf.nn.l2_loss((self.vae_weights["encoder_logvar"]))
            self.logvar_encoder = tf.add(tf.tensordot(hidden_encoder, self.vae_weights["encoder_logvar"], axes=[2, 0]),
                                              self.vae_biases["encoder_logvar"])

            self.epsilon = tf.random_normal(tf.shape(self.logvar_encoder), dtype=tf.float32, name='epsilon')

            ## sample latent variable
            self.std_encoder = tf.exp(tf.multiply(0.5, self.logvar_encoder))
            self.z = tf.add(mu_encoder, tf.multiply(self.std_encoder, self.epsilon))

            l2_loss += tf.nn.l2_loss(self.vae_weights["decoder_h1"])
            hidden_decoder = tf.nn.relu(tf.add(tf.tensordot(self.z, self.vae_weights["decoder_h1"], axes=[2, 0]), self.vae_biases["decoder_h1"]))

            ## decoder_reconstruction
            l2_loss += tf.nn.l2_loss(self.vae_weights["decoder_reconstruction"])
            self.x_hat = tf.add(tf.tensordot(hidden_decoder, self.vae_weights["decoder_reconstruction"], axes=[2, 0]),
                           self.vae_biases["decoder_reconstruction"])

            # calculate loss
            kl_divergence = -0.5 * tf.reduce_sum(1 + self.logvar_encoder - tf.square(mu_encoder) - tf.exp(self.logvar_encoder),
                                                 axis=2)
            bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_hat, labels=self.user_feat), axis=2)
            self.vae_loss = tf.reduce_mean(tf.add(kl_divergence, bce))
            self.total_vae_cost = tf.add(self.vae_loss, tf.multiply(self.reg_coeff, l2_loss))

    def _weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def build_graph(self):
        self.construct_vae()

        with tf.variable_scope('shared'):
            if self.cell_type == 'lstm':
                self.rnn_cell_shared = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
            else:
                self.rnn_cell_shared = tf.nn.rnn_cell.GRUCell(self.state_size)
            self.init_state_shared = self.rnn_cell_shared.zero_state(self.batch_size, tf.float32)
            self.shared_rnn_output, self.shared_hidden_state = \
                tf.nn.dynamic_rnn(self.rnn_cell_shared, self.tweet_vec, self.seq_len, self.init_state_shared)

        with tf.variable_scope('rumor-clf'):
            if self.cell_type == 'lstm':
                self.rumor_score = tf.layers.dense(inputs=self.shared_hidden_state.h, units=self.r_label_size,
                    activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.constant_initializer(0.25))
            else:
                self.rumor_score = tf.layers.dense(inputs=self.shared_hidden_state, units=self.r_label_size,
                                                   activation=tf.nn.relu,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.constant_initializer(0.25))
            rumor_true = tf.one_hot(tf.cast(self.rumor_output, dtype=tf.int32), self.r_label_size)
            self.rumor_label_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                                   (logits=self.rumor_score, labels=rumor_true))
            self.pred_rumor_label = tf.cast(tf.argmax(self.rumor_score, axis=1), dtype=tf.int32)

        with tf.variable_scope('stance-clf'):
            self.input_feat = tf.concat([self.shared_rnn_output, self.z], axis=2)
            weight, bias = self._weight_and_bias(self.state_size + self.vae_latent_size, self.s_label_size)
            output = tf.reshape(self.input_feat, [-1, self.state_size + self.vae_latent_size])
            self.stance_score = (tf.matmul(output, weight) + bias)
            self.stance_score = tf.reshape(self.stance_score, [-1, self.max_seq_len, self.s_label_size])
            self.stance_true = tf.one_hot(tf.cast(self.stance_output, dtype=tf.int32), self.s_label_size)
            self.temp_stance_cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.stance_true, logits=self.stance_score, dim=2)
            mask = tf.sequence_mask(lengths=self.seq_len, maxlen=self.max_seq_len, dtype=tf.float32)
            self.temp_stance_cost *= mask
            self.stance_label_cost = tf.reduce_mean(self.temp_stance_cost)
            # self.stance_label_cost = self.cost()
            self.pred_stance_label = tf.cast(tf.argmax(self.stance_score, axis=2), dtype=tf.int32)

        if self.arch == 'joint':
            with tf.variable_scope('joint-opt'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.total_cost = self.rumor_label_cost + self.stance_label_cost + self.total_vae_cost
                self.train_op = self.optimizer.minimize(self.total_cost)
        else:
            with tf.variable_scope('rumor-opt'):
                self.rumor_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.rumor_train_op = self.rumor_optimizer.minimize(self.rumor_label_cost)

            with tf.variable_scope('stance-opt'):
                self.stance_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.stance_train_op = self.stance_optimizer.minimize(self.stance_label_cost)

            with tf.variable_scope('vae-opt'):
                self.vae_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.vae_train_op = self.vae_optimizer.minimize(self.vae_optimizer)

    def train_model(self, train_data_loader, test_data_loader):
        # tf.reset_default_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            max_rumor_micro, max_rumor_macro, max_rumor_acc = 0., 0., 0.
            max_stance_micro, max_stance_macro, max_stance_acc = 0., 0., 0.
            best_rumor_cost = 10000.0
            best_stance_cost = 10000.0

            best_train_macro_rumor = 0.
            best_train_macro_stance = 0.
            best_train_micro_rumor = 0.
            best_train_micro_stance = 0.

            for epi in range(1, self.epochs + 1):
                global_cost = 0.
                global_rumor_cost = 0.
                global_stance_cost = 0.
                n_batches = len(train_data_loader)

                for i in range(0, n_batches):
                    batch_data, rumor_batch_label, stance_batch_label, batch_length, batch_user = train_data_loader()
                    if batch_data.shape[0] < self.batch_size:
                        continue
                    if self.arch == 'joint':
                        fd_all = {
                            self.tweet_vec: batch_data,
                            self.rumor_output: rumor_batch_label,
                            self.stance_output: stance_batch_label,
                            self.seq_len: batch_length,
                            self.user_feat: batch_user
                        }
                        _, total_cost, rumor_cost, stance_cost = sess.run([self.train_op, self.total_cost,
                                self.rumor_label_cost, self.stance_label_cost], feed_dict=fd_all)
                       #  print(vae_cost)
                        '''output_state, hidden_state = \
                            sess.run([self.shared_hidden_state, self.shared_rnn_output], feed_dict=fd_all)'''

                        global_cost += total_cost
                        global_rumor_cost += rumor_cost
                        global_stance_cost += stance_cost
                    else:
                        fd_rumor = {
                            self.tweet_vec: batch_data,
                            self.rumor_output: rumor_batch_label,
                            self.seq_len: batch_length
                        }
                        _, rumor_cost = sess.run([self.rumor_train_op, self.rumor_label_cost], feed_dict=fd_rumor)
                        global_rumor_cost += rumor_cost
                        fd_stance = {
                            self.tweet_vec: batch_data,
                            self.stance_output: stance_batch_label,
                            self.seq_len: batch_length
                        }
                        _, stance_cost = sess.run([self.stance_train_op, self.stance_label_cost], feed_dict=fd_stance)
                        global_stance_cost += stance_cost

                if self.arch == 'joint':
                    global_cost /= n_batches
                    global_rumor_cost /= n_batches
                    global_stance_cost /= n_batches
                else:
                    global_rumor_cost /= n_batches
                    global_stance_cost /= n_batches

                if epi % self.display_interval == 0:
                    if self.arch == 'joint':
                        print('end of epoch: {} avg total cost: {:0.3f}'.
                              format(epi, global_cost))
                        self.log.debug('end of epoch: {} avg total cost: {:0.3f}'.
                              format(epi, global_cost))
                    else:
                        print('end of epoch: {} avg rumor cost: {:0.3f} avg stance cost: {:0.3f}'.
                            format(epi, global_rumor_cost, global_stance_cost))
                        self.log.debug('end of epoch: {} avg rumor cost: {:0.3f} avg stance cost: {:0.3f}'.
                            format(epi, global_rumor_cost, global_stance_cost))
                '''if epi % self.train_performance_interval == 0:
                    avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
                    avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = self.test_model(sess, train_data_loader)

                    print('train: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor))
                    self.log.debug('train: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor))
                    print('train: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance))
                    self.log.debug('train: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance))'''

                if epi % self.test_interval == 0:
                    train_micro_f1_rumor, train_macro_f1_rumor, train_acc_rumor, \
                    train_micro_f1_stance, train_macro_f1_stance, train_acc_stance = self.test_model(sess, train_data_loader)

                    avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
                    avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = self.test_model(sess, test_data_loader)

                    '''if global_rumor_cost < best_rumor_cost:
                        best_rumor_cost = global_rumor_cost
                        max_rumor_macro = avg_macro_f1_rumor
                        max_rumor_micro = avg_micro_f1_rumor
                        max_rumor_acc = avg_acc_rumor
                    if global_stance_cost < best_stance_cost:
                        best_stance_cost = global_stance_cost
                        max_stance_macro = avg_macro_f1_stance
                        max_stance_micro = avg_micro_f1_stance
                        max_stance_acc = avg_acc_stance'''

                    if train_macro_f1_rumor > best_train_macro_rumor:
                        best_train_macro_rumor = train_macro_f1_rumor
                        max_rumor_macro = avg_macro_f1_rumor
                   
                    if train_micro_f1_rumor > best_train_micro_rumor:
                        best_train_micro_rumor = train_micro_f1_rumor
                        max_rumor_micro = avg_micro_f1_rumor
                        max_rumor_acc = avg_acc_rumor

                    if train_macro_f1_stance > best_train_macro_stance:
                        best_train_macro_stance = train_macro_f1_stance
                        max_stance_macro = avg_macro_f1_stance

                    if train_micro_f1_stance > best_train_micro_stance:
                        best_train_micro_stance = train_micro_f1_stance
                        max_stance_micro = avg_micro_f1_stance
                        max_stance_acc = avg_acc_stance

                    print('test: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor))
                    self.log.debug('test: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor))
                    print('test: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance))
                    self.log.debug('test: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance))

        print('final test: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
              format(max_rumor_micro, max_rumor_macro, max_rumor_acc))
        self.log.debug('final test: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                       format(max_rumor_micro, max_rumor_macro, max_rumor_acc))
        print('final test: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
              format(max_stance_micro, max_stance_macro, max_stance_acc))
        self.log.debug('final test: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                       format(max_stance_micro, max_stance_macro, max_stance_acc))

    def test_model(self, sess: tf.Session, data_loader):
        n_batches = len(data_loader)
        avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor = 0., 0., 0.
        avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = 0., 0., 0.
        for i in range(0, n_batches):
            batch_data, rumor_batch_label, stance_batch_label, batch_length, batch_user = data_loader()
            if batch_data.shape[0] < self.batch_size:
                continue
            fd_rumor = {
                self.tweet_vec: batch_data,
                self.rumor_output: rumor_batch_label,
                self.seq_len: batch_length,
                self.user_feat: batch_user
            }
            pred_rumor = sess.run(self.pred_rumor_label, feed_dict=fd_rumor)
            micro_f1_rumor = f1_score(rumor_batch_label, pred_rumor, average='micro')
            macro_f1_rumor = f1_score(rumor_batch_label, pred_rumor, average='macro')
            acc_rumor = accuracy_score(rumor_batch_label, pred_rumor)

            avg_macro_f1_rumor += macro_f1_rumor
            avg_micro_f1_rumor += micro_f1_rumor
            avg_acc_rumor += acc_rumor

            fd_stance = {
                self.tweet_vec: batch_data,
                self.stance_output: stance_batch_label,
                self.seq_len: batch_length,
                self.user_feat: batch_user
            }
            pred_stance = sess.run(self.pred_stance_label, feed_dict=fd_stance)
            temp_macro_f1_stance, temp_micro_f1_stance, temp_acc_stance = 0., 0., 0.
            for b in range(0, self.batch_size):
                micro_f1_stance = f1_score(stance_batch_label[b, 0: batch_length[b]],
                                           pred_stance[b, 0: batch_length[b]], average='micro')
                macro_f1_stance = f1_score(stance_batch_label[b, 0: batch_length[b]],
                                           pred_stance[b, 0: batch_length[b]], average='macro')
                acc_stance = accuracy_score(stance_batch_label[b, 0: batch_length[b]],
                                           pred_stance[b, 0: batch_length[b]])

                temp_macro_f1_stance += macro_f1_stance
                temp_micro_f1_stance += micro_f1_stance
                temp_acc_stance += acc_stance

            avg_macro_f1_stance += (temp_macro_f1_stance / self.batch_size)
            avg_micro_f1_stance += (temp_micro_f1_stance / self.batch_size)
            avg_acc_stance += (temp_acc_stance / self.batch_size)

        avg_macro_f1_rumor /= n_batches
        avg_micro_f1_rumor /= n_batches
        avg_acc_rumor /= n_batches

        avg_macro_f1_stance /= n_batches
        avg_micro_f1_stance /= n_batches
        avg_acc_stance /= n_batches

        return avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
               avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance