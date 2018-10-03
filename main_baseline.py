import numpy as np
import logging
from sklearn.model_selection import train_test_split
from args import Args
from utils import load_all_cascades, Corpus, Loader

# from data_utils import load_cascades, Loader

from models_old.mlt_single import MLTSingle
from models_old.bow_model_old import BOWModel
from models_old.mlt_shared_rnn import MLTSharedRNN
from models_old.mlt_shared_mlp import MLTSharedMLP

if __name__ == '__main__':
    options = Args()

    handler = logging.FileHandler('{}-{}.log'.format(options.main, options.model_type), 'w')
    log = logging.getLogger(options.main)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    corpus = Corpus()
    all_conv = load_all_cascades('cascades')
    data_np, lengths_np, labels_np = corpus.tokenize(all_conv)
    options.vocab_size = corpus.dictionary.__len__()
    options.seq_len = np.max(lengths_np)

    train_data_np, test_data_np, train_label_np, test_label_np, train_length_np, test_length_np = \
        train_test_split(data_np, labels_np, lengths_np, test_size=0.20, random_state=42)
    train_data_loader = Loader(train_data_np, train_length_np, train_label_np, options=options)
    test_data_loader = Loader(test_data_np, test_length_np, test_label_np, options=options)

    if options.model_type == 'shared-rnn':
        mlt_model = MLTSharedRNN(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader)
    elif options.model_type == 'shared-mlp':
        mlt_model = MLTSharedMLP(options)
        mlt_model.train_model(train_data_loader, test_data_loader)
    elif options.model_type == 'bow':
        bow_model = BOWModel(options)
        bow_model.train_model(train_data_loader, test_data_loader)
    elif options.model_type == 'single':
        single_model = MLTSingle(options=options)
        single_model.train_model(train_data_loader, test_data_loader)
