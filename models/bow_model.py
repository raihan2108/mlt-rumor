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