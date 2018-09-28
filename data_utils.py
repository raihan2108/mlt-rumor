import re
import os
import json
import numpy as np
from urlextract import URLExtract
from sklearn.feature_extraction.text import TfidfVectorizer

rumor_label = {'false': 0, 'true': 1, 'unverified': 2}
stance_label = {'supporting': 0, 'denying': 1, 'comment': 3, 'appeal-for-more-information': 3, 'no_stance': 4}


def get_cascade_info(data_path):
    all_tweets = []
    max_cas_length = -1
    events = os.listdir(data_path)
    th_count = 0
    for ev in events:
        thread_names = os.listdir(os.path.join(data_path, ev))
        for th in thread_names:
            json_data = json.load(open(os.path.join(data_path, ev, th)))
            if len(json_data['cascade']) > max_cas_length:
                max_cas_length = len(json_data['cascade'])
            for cas in json_data['cascade']:
                all_tweets.append(cas['text'])
            th_count += 1
    return all_tweets, max_cas_length, th_count


def load_cascades(data_path):
    global rumor_label
    global stance_label
    all_tweets, max_cascade_len, thread_count = get_cascade_info(data_path)
    print(max_cascade_len, thread_count)

    tf_idf_vectorizer = TfidfVectorizer(max_features=5000)
    tf_idf_vectorizer = tf_idf_vectorizer.fit(all_tweets)
    vocab_count = len(tf_idf_vectorizer.vocabulary_)

    all_tweet_numpy = np.zeros((thread_count, max_cascade_len, vocab_count), dtype=np.float32)
    stance_label_numpy = np.zeros((thread_count, max_cascade_len), dtype=np.int32)
    rumor_label_numpy = np.zeros(thread_count, dtype=np.int32)
    seq_len_numpy = np.zeros(thread_count, dtype=np.int32)
    events = os.listdir(data_path)
    tc = 0
    for ev in events:
        thread_names = os.listdir(os.path.join(data_path, ev))
        for th in thread_names:
            json_data = json.load(open(os.path.join(data_path, ev, th)))
            a_rumor_label = rumor_label[json_data['label']]
            rumor_label_numpy[tc] = a_rumor_label
            seq_len_numpy[tc] = len(json_data['cascade'])
            for i, cas in enumerate(json_data['cascade']):
                # entry = {}
                tweet = cas['text']
                text_feat = tf_idf_vectorizer.transform([tweet.lower()])
                a_stance_label = stance_label[cas['support']]
                all_tweet_numpy[tc, i, :] = np.asarray(text_feat.todense()).reshape([-1])
                stance_label_numpy[tc, i] = a_stance_label
                '''tid = cas['tweet_id']
                entry['text'] = tweet.lower()
                entry['stance_label'] = a_stance_label
                entry['rumor_label'] = a_rumor_label
                entry['tid'] = tid'''
            tc += 1
    return all_tweet_numpy, rumor_label_numpy, stance_label_numpy, seq_len_numpy

class Loader:
    def __init__(self, data, rumor_label, stance_label, lengths, options=None):
        self.batch_size = options.batch_size
        self.idx = 0
        self.data = data
        self.lenghts = lengths
        self.rumor_labels = rumor_label
        self.stance_labels = stance_label
        self.shuffle = True
        self.n = len(data)
        self.indices = np.arange(self.n, dtype="int32")
        self.options = options

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_data = [self.data[i] for i in batch_indices]
        batch_rumor_label = [self.rumor_labels[i] for i in batch_indices]
        batch_stance_label = [self.stance_labels[i] for i in batch_indices]
        batch_lengths = [self.lenghts[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return np.asarray(batch_data), np.asarray(batch_rumor_label), \
               np.asarray(batch_stance_label), np.asarray(batch_lengths)


def load_all_cascades(data_path):
    global rumor_label
    global stance_label
    all_data = []
    events = os.listdir(data_path)
    for ev in events:
        thread_names = os.listdir(os.path.join(data_path, ev))
        for th in thread_names:
            json_data = json.load(open(os.path.join(data_path, ev, th)))
            a_rumor_label = rumor_label[json_data['label']]
            for cas in json_data['cascade']:
                entry = {}
                tweet = cas['text']
                a_stance_label = stance_label[cas['support']]
                tid = cas['tweet_id']
                entry['text'] = tweet.lower()
                entry['stance_label'] = a_stance_label
                entry['rumor_label'] = a_rumor_label
                entry['tid'] = tid
                all_data.append(entry)
    return all_data