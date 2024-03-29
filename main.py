import logging
from sklearn.model_selection import train_test_split
from datetime import datetime

from args import Args
# from utils import load_all_cascades, Corpus, Loader

from data_utils import load_cascades, Loader

# from models.mlt_shared_rnn import MLTSharedRNN
# from models.mlt_shared_mlp import MLTSharedMLP
from models.mlt_us import MLT_US
from models.bow_model import BOWModel
from models.mlt_user import MLTUser
from models.mlt_user_rnn import MLTUserRNN
from models.mlt_single import MLT_Single

dataset_name = 'pheme' # 'pheme'            # twitter

if __name__ == '__main__':
    options = Args()
    start_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    handler = logging.FileHandler('{}-{}-{}-{}.log'.format(options.main, options.model_type, dataset_name, start_time), 'w')
    log = logging.getLogger(options.main)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    tweet_vec, rumor_label, stance_label, seq_len, user_feat = load_cascades('{}_cascades'.format(dataset_name))   #pheme_cascades
    print(tweet_vec.shape, rumor_label.shape, stance_label.shape)

    train_data_np, tmp_data_np, train_rumor_np, tmp_rumor_np, train_stance_np, tmp_stance_np, train_seq_len_np, \
        tmp_seq_len_np, train_user_feat_np, tmp_user_feat_np = train_test_split(tweet_vec, rumor_label,
            stance_label, seq_len, user_feat, test_size=0.20, random_state=42)

    test_data_np, val_data_np, test_rumor_np, val_rumor_np, test_stance_np, val_stance_np, test_seq_len_np, \
        val_seq_len_np, test_user_feat_np, val_user_feat_np = train_test_split(tmp_data_np, tmp_rumor_np,
            tmp_stance_np, tmp_seq_len_np, tmp_user_feat_np, test_size=0.50, random_state=42)

    options.seq_len = tweet_vec.shape[1]
    options.vocab_size = tweet_vec.shape[2]
    options.user_feat_size = user_feat.shape[2]

    log.debug(str(options))

    train_data_loader = Loader(train_data_np, train_rumor_np, train_stance_np, train_seq_len_np, train_user_feat_np, options=options)
    test_data_loader = Loader(test_data_np, test_rumor_np, test_stance_np, test_seq_len_np, test_user_feat_np, options=options)
    val_data_loader = Loader(val_data_np, val_rumor_np, val_stance_np, val_seq_len_np, val_user_feat_np,
                              options=options)

    if options.model_type == 'mlt-us':
        mlt_model = MLT_US(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader, val_data_loader)
    elif options.model_type == 'mlt-bow':
        mlt_model = BOWModel(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader, val_data_loader)
    elif options.model_type == 'mlt-user':
        mlt_model = MLTUser(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader, val_data_loader)
    elif options.model_type == 'mlt-user-rnn':
        mlt_model = MLTUserRNN(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader, val_data_loader)
    elif options.model_type == 'mlt-single':
        mlt_model = MLT_Single(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader, val_data_loader)


    # all_conv = load_all_cascades('cascades')
    # cv = CoprpusVectorizer(all_data=all_conv)
    '''corpus = Corpus()
    data_np, lengths_np, labels_np = corpus.tokenize(all_conv)
    options.vocab_size = corpus.dictionary.__len__()
    options.seq_len = np.max(lengths_np)'''

    '''train_data_np, test_data_np, train_label_np, test_label_np, train_length_np, test_length_np = \
                                    train_test_split(data_np, labels_np, lengths_np, test_size=0.33, random_state=42)
    train_data_loader = Loader(train_data_np, train_length_np, train_label_np, options=options)
    test_data_loader = Loader(test_data_np, test_length_np, test_label_np, options=options)

    if options.mode_type == 'shared-rnn':
        mlt_model = MLTSharedRNN(options=options)
        mlt_model.train_model(train_data_loader, test_data_loader)
    elif options.mode_type == 'shared-mlp':
        mlt_model = MLTSharedMLP(options)
        mlt_model.train_model(train_data_loader, test_data_loader)
    elif options.mode_type == 'bow':
        bow_model = BOWModel(options)
        bow_model.train_model(train_data_loader,test_data_loader)
    elif options.mode_type == 'single':
        single_model = MLTSingle(options=options)
        single_model.train_model(train_data_loader, test_data_loader)'''
