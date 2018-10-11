import json
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report


class MLT_ES:
    def __init__(self, options):
        #hyper params and other input parameters
        self.batch_size = options.batch_size
        self.max_seq_len = options.seq_len
        self.state_size = options.state_size
        self.learning_rate = options.learning_rate
        self.vocab_size = options.vocab_size
        self.emb_size = options.emb_size
        self.epochs = options.epochs
        self.add_regularization = options.add_regularization
        self.cell_type = options.cell_type

        # type of architecture/learning mech to use
        # it can be 1) joint (both stance and rumor are optimized together)
        # or 2) just rumor or stance
        self.arch = options.archi

        # ground truth labels
        self.r_label_size = options.rumor_label
        self.s_label_size = options.stance_label

        # options to control logging / performance checks
        self.display_interval = options.display_epoch
        self.train_performance_interval = options.train_performnace_epoch
        self.test_interval = options.test_epoch
        self.log = logging.getLogger(options.main + '.' + __name__)
        self.log.setLevel(logging.DEBUG)

        self.init_variables()
        self.build_graph()

    def _weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def cost(self):
        # Compute cross entropy for each frame.
        mask = tf.sequence_mask(lengths=self.seq_len, maxlen=self.max_seq_len, dtype=tf.float32)
        cross_entropy = self.stance_true * tf.log(self.stance_score)
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)
        # mask = tf.sign(tf.reduce_max(tf.abs(self.stance_true), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        cross_entropy /= tf.cast(self.seq_len, tf.float32)
        return tf.reduce_mean(cross_entropy)

    def init_variables(self):
        self.tweet_vec = tf.placeholder(shape=[self.batch_size, self.max_seq_len, self.vocab_size], dtype=tf.float32)
        self.seq_len = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.stance_output = tf.placeholder(shape=[self.batch_size, self.max_seq_len], dtype=tf.float32)
        self.rumor_output = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

    def make_rnn_cell(self):
        if self.cell_type == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        else:
            cell = tf.nn.rnn_cell.GRUCell(self.state_size)

        return cell

    def get_rnn(self, cell, rnn_input, scope):
        init_state = cell.zero_state(self.batch_size, tf.float32)
        allstates, last_hidden_state = tf.nn.dynamic_rnn(cell, rnn_input,
                                                                   self.seq_len, init_state, scope=scope)
        if self.cell_type == "lstm":
            last_hidden_state = last_hidden_state.h

        return allstates, last_hidden_state

    def build_graph(self):
        self.reglosses = {}
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        with tf.variable_scope('stance_pre_shared', regularizer=self.regularizer, reuse=tf.AUTO_REUSE):
            self.stanceEmbedding = tf.layers.dense(self.tweet_vec, self.emb_size, use_bias=False)

        if self.add_regularization:
            self.reglosses['stance_pre_shared'] = tf.reduce_sum((tf.losses.get_regularization_losses(scope='stance_pre_shared')))

        with tf.variable_scope('rumor_pre_shared', regularizer=self.regularizer, reuse=tf.AUTO_REUSE):
            self.rumorEmbedding = tf.layers.dense(self.tweet_vec, self.emb_size, use_bias=False)

        if self.add_regularization:
            self.reglosses['rumor_pre_shared'] = tf.reduce_sum(tf.losses.get_regularization_losses(scope='rumor_pre_shared'))

        with tf.variable_scope('shared', regularizer=self.regularizer, reuse=tf.AUTO_REUSE):
            shared_rnn_cell = self.make_rnn_cell()

        if self.add_regularization:
            self.reglosses['shared'] = tf.reduce_sum(tf.losses.get_regularization_losses(scope='shared'))

        self.sharedRumorOut = self.get_rnn(shared_rnn_cell, self.rumorEmbedding, scope='shared')
        self.sharedStanceOut = self.get_rnn(shared_rnn_cell, self.stanceEmbedding, scope='shared')

        with tf.variable_scope('stance_rnn_taskSpecific'):
            self.stanceCell = self.make_rnn_cell()

        if self.add_regularization:
            self.reglosses['stance_rnn_taskSpecific'] = tf.reduce_sum((tf.losses.get_regularization_losses(scope='stance_rnn_taskSpecific')))

        input = tf.concat([self.stanceEmbedding, self.sharedRumorOut[0]], axis=-1)
        self.stanceRNN = self.get_rnn(self.stanceCell, input, scope="stance_rnn_taskSpecific")

        with tf.variable_scope('rumor_rnn_taskspecific'):
            self.rumorCell = self.make_rnn_cell()

        if self.add_regularization:
            self.reglosses['rumor_rnn_taskspecific'] = tf.reduce_sum((tf.losses.get_regularization_losses(scope='rumor_rnn_taskspecific')))

        input = tf.concat([self.rumorEmbedding, self.sharedStanceOut[0]], axis=-1)
        self.rumorRNN = self.get_rnn(self.rumorCell, input, scope='rumor_rnn_taskspecific')



        with tf.variable_scope('rumor-clf', regularizer=self.regularizer, reuse=tf.AUTO_REUSE):
            self.rumor_score = tf.layers.dense(inputs=self.rumorRNN[-1], units=self.r_label_size,
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               bias_initializer=tf.constant_initializer(0.25))
            rumor_true = tf.one_hot(tf.cast(self.rumor_output, dtype=tf.int32), self.r_label_size)
            self.rumor_label_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                                   (logits=self.rumor_score, labels=rumor_true))
            self.pred_rumor_label = tf.cast(tf.argmax(self.rumor_score, axis=1), dtype=tf.int32)

        if self.add_regularization:
            self.reglosses['rumor-clf'] = tf.reduce_sum(tf.losses.get_regularization_losses(scope='rumor-clf'))

        with tf.variable_scope('stance-clf', regularizer=self.regularizer, reuse=tf.AUTO_REUSE):
            weight, bias = self._weight_and_bias(self.state_size, self.s_label_size)

            stancernn_first = tf.tile(self.stanceRNN[0][:, :1, :], [1, self.stanceRNN.shape[1], 1])
            stancernn_concat = tf.concat([stancernn_first, self.stanceRNN])
            output = tf.reshape(self.stanceRNN[0], [-1, self.state_size])
            self.stance_score = (tf.matmul(output, weight) + bias)
            self.stance_score = tf.reshape(self.stance_score, [-1, self.max_seq_len, self.s_label_size])
            self.stance_true = tf.one_hot(tf.cast(self.stance_output, dtype=tf.int32), self.s_label_size)
            self.temp_stance_cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.stance_true, logits=self.stance_score, dim=2)
            mask = tf.sequence_mask(lengths=self.seq_len, maxlen=self.max_seq_len, dtype=tf.float32)
            self.temp_stance_cost *= mask
            self.stance_label_cost = tf.reduce_mean(self.temp_stance_cost)
            # self.stance_label_cost = self.cost()
            self.pred_stance_label = tf.cast(tf.argmax(self.stance_score, axis=2), dtype=tf.int32)

        if self.add_regularization:
            self.reglosses['stance-clf'] = tf.reduce_sum(tf.losses.get_regularization_losses(scope='stance-clf'))

        regloss = 0
        if self.arch == 'joint':
            with tf.variable_scope('joint-opt'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                if self.add_regularization:
                    for v in self.reglosses.values():
                        regloss += v

                self.total_cost = self.rumor_label_cost + self.stance_label_cost  + regloss
                self.train_op = self.optimizer.minimize(self.total_cost)
        else:
            with tf.variable_scope('rumor-opt'):
                if self.add_regularization:
                    regloss = (self.reglosses['rumor_pre_shared'] + self.reglosses['shared'] +
                               self.reglosses['rumor-clf'] + self.reglosses['rumor_rnn_taskspecific'])

                self.rumor_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                self.rumor_train_op = self.rumor_optimizer.minimize(self.rumor_label_cost +regloss)

            with tf.variable_scope('stance-opt'):
                if self.add_regularization:
                    regloss = (self.reglosses['stance_pre_shared'] + self.reglosses['shared'] +
                               self.reglosses['stance-clf'] + self.reglosses['stance_rnn_taskSpecific'])

                self.stance_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                self.stance_train_op = self.stance_optimizer.minimize(self.stance_label_cost + regloss)

    def train_model(self, train_data_loader, test_data_loader, val_data_loader):
        # tf.reset_default_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            max_rumor_micro, max_rumor_macro, max_rumor_acc = 0., 0., 0.
            max_stance_micro, max_stance_macro, max_stance_acc = 0., 0., 0.
            max_cr_micro_rumor, max_cr_macro_rumor = None, None
            max_cr_micro_stance, max_cr_macro_stance = None, None
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
                    batch_data, rumor_batch_label, stance_batch_label, batch_length, _ = train_data_loader()
                    if batch_data.shape[0] < self.batch_size:
                        continue
                    if self.arch == 'joint':
                        fd_all = {
                            self.tweet_vec: batch_data,
                            self.rumor_output: rumor_batch_label,
                            self.stance_output: stance_batch_label,
                            self.seq_len: batch_length
                        }
                        _, total_cost, rumor_cost, stance_cost = sess.run([self.train_op, self.total_cost,
                                                    self.rumor_label_cost, self.stance_label_cost], feed_dict=fd_all)
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
                    '''train_micro_f1_rumor, train_macro_f1_rumor, train_acc_rumor, \
                    train_micro_f1_stance, train_macro_f1_stance, train_acc_stance = self.test_model(sess, train_data_loader)'''

                    '''avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
                    avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = self.test_model(sess, test_data_loader)'''

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
                    cr_rumor_val, cr_stance_val = self.test_model(sess, val_data_loader)
                    cr_rumor_test, cr_stance_test = self.test_model(sess, test_data_loader)

                    avg_micro_f1_rumor = cr_rumor_test['micro avg']['f1-score']
                    avg_macro_f1_rumor = cr_rumor_test['macro avg']['f1-score']
                    avg_acc_rumor = cr_rumor_test['micro avg']['f1-score']

                    avg_micro_f1_stance = cr_stance_test['micro avg']['f1-score']
                    avg_macro_f1_stance = cr_stance_test['macro avg']['f1-score']
                    avg_acc_stance = cr_stance_test['micro avg']['f1-score']

                    val_macro_f1_rumor = cr_rumor_val['macro avg']['f1-score']
                    val_micro_f1_rumor = cr_rumor_val['micro avg']['f1-score']
                    val_macro_f1_stance = cr_stance_val['macro avg']['f1-score']
                    val_micro_f1_stance = cr_stance_val['micro avg']['f1-score']

                    if val_macro_f1_rumor > best_train_macro_rumor:
                        best_train_macro_rumor = val_macro_f1_rumor
                        max_rumor_macro = cr_rumor_test['macro avg']['f1-score']
                        max_cr_macro_rumor = cr_rumor_test

                    if val_micro_f1_rumor > best_train_micro_rumor:
                        best_train_micro_rumor = val_micro_f1_rumor
                        max_rumor_micro = cr_rumor_test['micro avg']['f1-score']
                        max_rumor_acc = cr_rumor_test['micro avg']['f1-score']
                        max_cr_micro_rumor = cr_rumor_test

                    if val_macro_f1_stance > best_train_macro_stance:
                        best_train_macro_stance = val_macro_f1_stance
                        max_stance_macro = cr_stance_test['macro avg']['f1-score']
                        max_cr_macro_stance = cr_stance_test

                    if val_micro_f1_stance > best_train_micro_stance:
                        best_train_micro_stance = val_micro_f1_stance
                        max_stance_micro = cr_stance_test['micro avg']['f1-score']
                        max_stance_acc = cr_stance_test['micro avg']['f1-score']
                        max_cr_micro_stance = cr_stance_test

                    print('val: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(val_micro_f1_rumor, val_macro_f1_rumor, val_micro_f1_rumor))
                    self.log.debug('val: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                                   format(val_micro_f1_rumor, val_macro_f1_rumor, val_micro_f1_rumor))
                    print('val: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(val_micro_f1_stance, val_macro_f1_stance, val_micro_f1_stance))
                    self.log.debug('val: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                                   format(val_micro_f1_stance, val_macro_f1_stance, val_micro_f1_stance))

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

        self.log.debug('classification report best micro rumor')
        self.log.debug(json.dumps(max_cr_micro_rumor, indent=2))
        self.log.debug('classification report best macro rumor')
        self.log.debug(json.dumps(max_cr_macro_rumor, indent=2))

        self.log.debug('classification report best micro stance')
        self.log.debug(json.dumps(max_cr_micro_stance, indent=2))
        print("0:{:0.3f}, 1: {:0.3f}, 2: {:0.3f}, 3: {:0.3f}".format(max_cr_micro_stance["0"]["f1-score"],
                                                                     max_cr_micro_stance["1"]["f1-score"],
                                                                     max_cr_micro_stance["2"]["f1-score"],
                                                                     max_cr_micro_stance["3"]["f1-score"]))
        self.log.debug('classification report best macro stance')
        print("0:{:0.3f}, 1: {:0.3f}, 2: {:0.3f}, 3: {:0.3f}".format(max_cr_macro_stance["0"]["f1-score"],
                                                                     max_cr_macro_stance["1"]["f1-score"],
                                                                     max_cr_macro_stance["2"]["f1-score"],
                                                                     max_cr_macro_stance["3"]["f1-score"]))
        self.log.debug(json.dumps(max_cr_macro_stance, indent=2))

    def test_model(self, sess: tf.Session, data_loader):
        n_batches = len(data_loader)
        avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor = 0., 0., 0.
        avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = 0., 0., 0.
        all_pred_rumor = list()
        all_true_rumor = list()
        all_pred_stance = list()
        all_true_stance = list()

        for i in range(0, n_batches):
            batch_data, rumor_batch_label, stance_batch_label, batch_length, _ = data_loader()
            if batch_data.shape[0] < self.batch_size:
                continue
            fd_rumor = {
                self.tweet_vec: batch_data,
                self.rumor_output: rumor_batch_label,
                self.seq_len: batch_length
            }
            pred_rumor = sess.run(self.pred_rumor_label, feed_dict=fd_rumor)
            all_pred_rumor.extend(pred_rumor.tolist())
            all_true_rumor.extend(rumor_batch_label.tolist())

            '''micro_f1_rumor = f1_score(rumor_batch_label, pred_rumor, average='micro')
            macro_f1_rumor = f1_score(rumor_batch_label, pred_rumor, average='macro')
            acc_rumor = accuracy_score(rumor_batch_label, pred_rumor)

            avg_macro_f1_rumor += macro_f1_rumor
            avg_micro_f1_rumor += micro_f1_rumor
            avg_acc_rumor += acc_rumor'''

            fd_stance = {
                self.tweet_vec: batch_data,
                self.stance_output: stance_batch_label,
                self.seq_len: batch_length
            }
            pred_stance = sess.run(self.pred_stance_label, feed_dict=fd_stance)
            # temp_macro_f1_stance, temp_micro_f1_stance, temp_acc_stance = 0., 0., 0.
            for b in range(0, self.batch_size):
                all_true_stance.extend(stance_batch_label[b, 0: batch_length[b]].tolist())
                all_pred_stance.extend(pred_stance[b, 0: batch_length[b]].tolist())
            '''for b in range(0, self.batch_size):
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
            avg_acc_stance += (temp_acc_stance / self.batch_size)'''

        '''avg_macro_f1_rumor /= n_batches
        avg_micro_f1_rumor /= n_batches
        avg_acc_rumor /= n_batches

        avg_macro_f1_stance /= n_batches
        avg_micro_f1_stance /= n_batches
        avg_acc_stance /= n_batches'''

        '''avg_micro_f1_rumor = f1_score(y_true=all_true_rumor, y_pred=all_pred_rumor, average='micro')
        avg_macro_f1_rumor = f1_score(y_true=all_true_rumor, y_pred=all_pred_rumor, average='macro')
        avg_acc_rumor = accuracy_score(y_true=all_true_rumor, y_pred=all_pred_rumor)

        avg_micro_f1_stance = f1_score(y_true=all_true_stance, y_pred=all_pred_stance, average='micro')
        avg_macro_f1_stance = f1_score(y_true=all_true_stance, y_pred=all_pred_stance, average='macro')
        avg_acc_stance = accuracy_score(y_true=all_true_stance, y_pred=all_pred_stance)'''

        cr_rumor = classification_report(y_true=all_true_rumor, y_pred=all_pred_rumor, output_dict=True)
        cr_stance = classification_report(y_true=all_true_stance, y_pred=all_pred_stance, output_dict=True)
        return cr_rumor, cr_stance
