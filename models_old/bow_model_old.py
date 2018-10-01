import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

from args import Args


class BOWModel:
    def __init__(self, options: Args):
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

        self.display_interval = options.display_epoch
        self.train_performance_interval = options.train_performnace_epoch
        self.test_interval = options.test_epoch
        self.log = logging.getLogger(options.main + __name__)
        self.log.setLevel(logging.DEBUG)

        self.init_variabels()
        self.build_graph()

    def init_variabels(self):
        self.tweet_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_seq_len])
        self.seq_len = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.rumor_output = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.stance_output = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.emb = tf.get_variable('emb', initializer=tf.truncated_normal(shape=[self.vocab_size, self.emb_size]))
        self.inputs_seqs = tf.nn.embedding_lookup(self.emb, tf.cast(self.tweet_input, dtype=tf.int32))

    def build_graph(self):
        self.inputs_feat = []
        for i in range(0, self.batch_size):
            self.inputs_feat.append(tf.reduce_mean(self.inputs_seqs[i, 0: self.seq_len[i], :], axis=0))
        self.inputs_feat = tf.convert_to_tensor(self.inputs_feat)

        # tf.get_variable('input_feat', initializer=self.inputs_feat)
        self.rumor_score = tf.layers.dense(inputs=self.inputs_feat, units=self.r_label_size,
            activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           bias_initializer=tf.constant_initializer(0.25))
        rumor_true = tf.one_hot(tf.cast(self.rumor_output, dtype=tf.int32), self.r_label_size)
        self.rumor_label_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                                               (logits=self.rumor_score, labels=rumor_true))
        self.pred_rumor_label = tf.cast(tf.argmax(self.rumor_score, axis=1), dtype=tf.int32)

        self.stance_score = tf.layers.dense(inputs=self.inputs_feat, units=self.s_label_size,
            activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.constant_initializer(0.25))
        stance_true = tf.one_hot(tf.cast(self.stance_output, dtype=tf.int32), self.s_label_size)
        self.stance_label_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                                                (logits=self.stance_score, labels=stance_true))
        self.pred_stance_label = tf.cast(tf.argmax(self.stance_score, axis=1), dtype=tf.int32)

        if self.arch == 'joint':
            with tf.variable_scope('joint-opt'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.total_cost = self.rumor_label_cost + self.stance_label_cost
                self.train_op = self.optimizer.minimize(self.total_cost)
        else:
            with tf.variable_scope('rumor-opt'):
                self.rumor_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.rumor_train_op = self.rumor_optimizer.minimize(self.rumor_label_cost)

            with tf.variable_scope('stance-opt'):
                self.stance_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.stance_train_op = self.stance_optimizer.minimize(self.stance_label_cost)

    def train_model(self, train_data_loader, test_data_loader):
        # tf.reset_default_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            max_rumor_micro, max_rumor_macro, max_rumor_acc = 0., 0., 0.
            max_stance_micro, max_stance_macro, max_stance_acc = 0., 0., 0.
            for epi in range(1, self.epochs + 1):
                if self.arch == 'joint':
                    global_cost = 0.
                else:
                    global_rumor_cost = 0.
                    global_stance_cost = 0.
                n_batches = len(train_data_loader)
                for i in range(0, n_batches):
                    batch_data, batch_label, batch_length = train_data_loader()
                    if batch_data.shape[0] < self.batch_size:
                        continue
                    if self.arch == 'joint':
                        fd_all = {
                            self.tweet_input: batch_data,
                            self.rumor_output: batch_label[:, 0],
                            self.stance_output: batch_label[:, 1],
                            self.seq_len: batch_length
                        }
                        _, total_cost = sess.run([self.train_op, self.total_cost], feed_dict=fd_all)
                        '''output_state, hidden_state = \
                            sess.run([self.shared_hidden_state, self.shared_rnn_output], feed_dict=fd_all)'''
                        global_cost += total_cost
                    else:
                        fd_rumor = {
                            self.tweet_input: batch_data,
                            self.rumor_output: batch_label[:, 0],
                            self.seq_len: batch_length
                        }
                        _, rumor_cost = sess.run([self.rumor_train_op, self.rumor_label_cost], feed_dict=fd_rumor)
                        global_rumor_cost += rumor_cost
                        fd_stance = {
                            self.tweet_input: batch_data,
                            self.stance_output: batch_label[:, 1],
                            self.seq_len: batch_length
                        }
                        _, stance_cost = sess.run([self.stance_train_op, self.stance_label_cost], feed_dict=fd_stance)
                        global_stance_cost += stance_cost

                if self.arch == 'joint':
                    global_cost /= n_batches
                else:
                    global_rumor_cost /= n_batches
                    global_stance_cost /= n_batches

                if epi % self.display_interval == 0:
                    if self.arch == 'joint':
                        print('end of epoch: {} avg total cost: {:0.3f}'.
                              format(epi, global_cost))
                    else:
                        print('end of epoch: {} avg rumor cost: {:0.3f} avg stance cost: {:0.3f}'.
                            format(epi, global_rumor_cost, global_stance_cost))
                if epi % self.train_performance_interval == 0:
                    avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
                    avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = self.test_model(sess, train_data_loader)

                    print('train: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor))
                    print('train: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance))

                if epi % self.test_interval == 0:
                    avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
                    avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = self.test_model(sess, test_data_loader)

                    print('test: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
                          format(avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor))
                    print('test: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
                          format(avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance))

    def test_model(self, sess, data_loader):
        n_batches = len(data_loader)
        avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor = 0., 0., 0.
        avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance = 0., 0., 0.
        for i in range(0, n_batches):
            batch_data, batch_label, batch_length = data_loader()
            if batch_data.shape[0] < self.batch_size:
                continue
            fd_rumor = {
                self.tweet_input: batch_data,
                self.rumor_output: batch_label[:, 0],
                self.seq_len: batch_length
            }
            pred_rumor = sess.run(self.pred_rumor_label, feed_dict=fd_rumor)
            micro_f1_rumor = f1_score(batch_label[:, 0], pred_rumor, average='micro')
            macro_f1_rumor = f1_score(batch_label[:, 0], pred_rumor, average='macro')
            acc_rumor = accuracy_score(batch_label[:, 0], pred_rumor)

            avg_macro_f1_rumor += macro_f1_rumor
            avg_micro_f1_rumor += micro_f1_rumor
            avg_acc_rumor += acc_rumor

            fd_stance = {
                self.tweet_input: batch_data,
                self.stance_output: batch_label[:, 1],
                self.seq_len: batch_length
            }
            pred_stance = sess.run(self.pred_stance_label, feed_dict=fd_stance)
            micro_f1_stance = f1_score(batch_label[:, 1], pred_stance, average='micro')
            macro_f1_stance = f1_score(batch_label[:, 1], pred_stance, average='macro')
            acc_stance = accuracy_score(batch_label[:, 1], pred_stance)

            avg_macro_f1_stance += macro_f1_stance
            avg_micro_f1_stance += micro_f1_stance
            avg_acc_stance += acc_stance

        avg_macro_f1_rumor /= n_batches
        avg_micro_f1_rumor /= n_batches
        avg_acc_rumor /= n_batches

        avg_macro_f1_stance /= n_batches
        avg_micro_f1_stance /= n_batches
        avg_acc_stance /= n_batches

        return avg_micro_f1_rumor, avg_macro_f1_rumor, avg_acc_rumor, \
               avg_micro_f1_stance, avg_macro_f1_stance, avg_acc_stance