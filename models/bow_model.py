import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score


class BOWModel:
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

        self.display_interval = options.display_epoch
        self.train_performance_interval = options.train_performnace_epoch
        self.test_interval = options.test_epoch
        self.log = logging.getLogger(options.main + '.' + __name__)
        self.log.setLevel(logging.DEBUG)

    def init_variables(self):
        self.tweet_vec = tf.placeholder(shape=[self.batch_size, self.max_seq_len, self.vocab_size], dtype=tf.float32)
        self.seq_len = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.stance_output = tf.placeholder(shape=[self.batch_size, self.max_seq_len], dtype=tf.float32)
        self.rumor_output = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

    def build_graph(self):
        self.inputs_feat = []
        for i in range(0, self.batch_size):
            self.inputs_feat.append(tf.reduce_mean(self.tweet_vec[i, 0: self.seq_len[i], :], axis=0))
        self.inputs_feat = tf.convert_to_tensor(self.inputs_feat)
        self.rumor_score = tf.layers.dense(inputs=self.inputs_feat, units=self.r_label_size,
            activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.25))
        rumor_true = tf.one_hot(tf.cast(self.rumor_output, dtype=tf.int32), self.r_label_size)
        self.rumor_label_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                                               (logits=self.rumor_score, labels=rumor_true))
        self.pred_rumor_label = tf.cast(tf.argmax(self.rumor_score, axis=1), dtype=tf.int32)