import re
import os
import json
import numpy as np
from urlextract import URLExtract
from sklearn.feature_extraction.text import TfidfVectorizer

rumor_label = {'false': 0, 'true': 1, 'unverified': 2}
stance_label = {'supporting': 0, 'denying': 1, 'comment': 2, 'appeal-for-more-information': 3}

url_pattern = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.dictionary.add_word('<LINK>')

    def tokenize(self, all_data):
        max_len = 0
        count = 0
        for data in all_data:
            count += 1
            tweet = data['text']
            urls = re.findall(url_pattern, tweet)
            for u in urls:
                tweet.replace(u, '<LINK>')
            words = tweet.split()
            if len(words) > max_len:
                max_len = len(words)
            for word in words:
                self.dictionary.add_word(word)

        data_numpy = np.zeros((count, max_len), dtype=np.int32)
        lengths_numpy = np.zeros(count, dtype=np.int32)
        labels_numpy = np.zeros((count, 2), dtype=np.int32)
        for i, data in enumerate(all_data):
            labels_numpy[i][0] = all_data[i]['rumor_label']
            labels_numpy[i][1] = all_data[i]['stance_label']

            text = all_data[i]['text']
            urls = re.findall(url_pattern, text)
            for u in urls:
                text.replace(u, '<LINK>')
            tokens = text.split()
            for j, word in enumerate(tokens):
                data_numpy[i, j] = self.dictionary.word2idx[word]
            lengths_numpy[i] = len(tokens)
        return data_numpy, lengths_numpy, labels_numpy


class CoprpusVectorizer:
    def __init__(self, all_data, options=None):
        tweets = [c['text'] for c in all_data]
        extractor = URLExtract()
        count = len(tweets)
        for i, tw in enumerate(tweets):
            # urls = re.findall(url_pattern, tw)
            urls = extractor.find_urls(tw)
            for u in urls:
                tweets[i] = tw.replace(u, '<LINK>')
        stance_labels = [c['stance_label'] for c in all_data]
        rumor_labels = [c['rumor_label'] for c in all_data]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(tweets)
        data_numpy = X.todense()
        lengths_numpy = np.zeros(count, dtype=np.int32)
        labels_numpy = np.zeros((count, 2), dtype=np.int32)
        for i, data in enumerate(all_data):
            labels_numpy[i][0] = all_data[i]['rumor_label']
            labels_numpy[i][1] = all_data[i]['stance_label']
        self.data_numpy = np.copy(data_numpy)
        self.length_numy = np.copy(lengths_numpy)
        self.label_numpy


class Loader:
    def __init__(self, data, lengths, labels, options=None):
        self.batch_size = options.batch_size
        self.idx = 0
        self.data = data
        self.lenghts = lengths
        self.labels = labels
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
        batch_label = [self.labels[i] for i in batch_indices]
        batch_lengths = [self.lenghts[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return np.asarray(batch_data), np.asarray(batch_label), np.asarray(batch_lengths)


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


if __name__ == '__main__':
    all_data = load_all_cascades('cascades')
    print(len(all_data))