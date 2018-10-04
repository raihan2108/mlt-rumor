import os
import json
import numpy as np
from  datetime import datetime
from urlextract import URLExtract
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import tree
from pprint import pprint
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

rumor_label = {'false': 0, 'true': 1, 'unverified': 2}
stance_label = {'supporting': 0, 'denying': 1, 'comment': 2, 'appeal-for-more-information': 3}
dataset_name = 'twitter_cascades'

sid = SentimentIntensityAnalyzer()


def get_pos_neg_ratio(text: str):
    total_words = text.split()
    pc, nc = 0., 0.
    for word in total_words:
        pol_score = sid.polarity_scores(word)
        if pol_score['compound'] > 0:
            pc += 1
        if pol_score['compound'] < 0:
            nc += 1
    return [pc / len(total_words), nc / len(total_words)]


def calc_topic_feat(lang_feat_th, user_feat_th):
    topic_feat = list()
    topic_feat.append(len(lang_feat_th))
    if len(lang_feat_th) != 0:
        topic_feat.append(sum([a[0] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[1] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[2] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[3] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[4] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[5] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[6] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[7] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[8] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[9] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[10] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[11] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[12] for a in lang_feat_th]) / len(lang_feat_th))
        topic_feat.append(sum([a[13] for a in lang_feat_th]) / len(lang_feat_th))
    else:
        topic_feat.append(sum([a[0] for a in lang_feat_th]))
        topic_feat.append(sum([a[1] for a in lang_feat_th]))
        topic_feat.append(sum([a[2] for a in lang_feat_th]))
        topic_feat.append(sum([a[3] for a in lang_feat_th]))
        topic_feat.append(sum([a[4] for a in lang_feat_th]))
        topic_feat.append(sum([a[5] for a in lang_feat_th]))
        topic_feat.append(sum([a[6] for a in lang_feat_th]))
        topic_feat.append(sum([a[7] for a in lang_feat_th]))
        topic_feat.append(sum([a[8] for a in lang_feat_th]))
        topic_feat.append(sum([a[9] for a in lang_feat_th]))
        topic_feat.append(sum([a[10] for a in lang_feat_th]))
        topic_feat.append(sum([a[11] for a in lang_feat_th]))
        topic_feat.append(sum([a[12] for a in lang_feat_th]))
        topic_feat.append(sum([a[13] for a in lang_feat_th]))

    if len(user_feat_th) != 0:
        topic_feat.append(sum([a[0] for a in user_feat_th]) / len(user_feat_th))
        topic_feat.append(sum([a[1] for a in user_feat_th]) / len(user_feat_th))
        topic_feat.append(sum([a[2] for a in user_feat_th]) / len(user_feat_th))
        topic_feat.append(sum([a[3] for a in user_feat_th]) / len(user_feat_th))
        topic_feat.append(sum([a[4] for a in user_feat_th]) / len(user_feat_th))
        topic_feat.append(sum([a[5] for a in user_feat_th]) / len(user_feat_th))
        topic_feat.append(sum([a[6] for a in user_feat_th]) / len(user_feat_th))
    else:
        topic_feat.append(sum([a[0] for a in user_feat_th]))
        topic_feat.append(sum([a[1] for a in user_feat_th]))
        topic_feat.append(sum([a[2] for a in user_feat_th]))
        topic_feat.append(sum([a[3] for a in user_feat_th]))
        topic_feat.append(sum([a[4] for a in user_feat_th]))
        topic_feat.append(sum([a[5] for a in user_feat_th]))
        topic_feat.append(sum([a[6] for a in user_feat_th]))
    return topic_feat


def extract_lang_features(entry):
    ret = []
    ret.append(len(entry))
    ret.append(len(entry.split()))
    if '?' in entry:   #2
        ret.append(1)
    else:
        ret.append(0)
    if '!' in entry:   # 3
        ret.append(1)
    else:
        ret.append(0)
    if ':-)' in entry or ';-)' in entry:    # 4
        ret.append(1)
    else:
        ret.append(0)
    if ':-(' in entry or ';-(' in entry:    # 5
        ret.append(1)
    else:
        ret.append(0)
    if 'i' in entry.lower():    # 6
        ret.append(1)
    else:
        ret.append(0)
    if 'you' in entry.lower():    # 7
        ret.append(1)
    else:
        ret.append(0)
    if 'they' in entry.lower() or 'he' in entry.lower() or 'she' in entry.lower():    # 8
        ret.append(1)
    else:
        ret.append(0)

    c = sum(1 for c in entry if c.isupper())
    ret.append(c / len(entry))     # 9
    l = sum(1 for c in entry.split() if c == '<LINK>')
    ret.append(l)            # 10
    if 'RT' in entry:       # 11
        ret.append(1)
    else:
        ret.append(0)
    h = sum(1 for c in entry if c == '#')
    ret.append(h / len(entry))          # 12
    m = sum(1 for c in entry if c == '@')
    ret.append(m / len(entry))      # 13
    ret.extend(get_pos_neg_ratio(entry))   # 14 15

    return ret


def extract_user_features(entry):
    ret = list()
    now_time = datetime.now()
    reg_time = datetime.strptime(entry['reg_time'], '%Y-%m-%d %H:%M:%S')
    time_delta = now_time-reg_time
    ret.append(time_delta.days)    # 0
    ret.append(entry['followers_count'])   # 1
    ret.append(entry['statuses_count'])    # 2
    ret.append(entry['friends_count'])    # 3
    if entry['verified'] is not None:     # 4
        ret.append(1)
    else:
        ret.append(0)
    if entry['description'] is not None:     # 5
        ret.append(1)
    else:
        ret.append(0)
    if entry['url'] is not None:     # 6
        ret.append(1)
    else:
        ret.append(0)

    return ret


def load_json(data_path):
    all_data = {}
    events = os.listdir(data_path)
    for ev in events:
        if ev not in all_data:
            all_data[ev] = {}
        thread_names = os.listdir(os.path.join(data_path, ev))
        for th in thread_names:
            if th not in all_data[ev]:
                all_data[ev][th] = []
            json_data = json.load(open(os.path.join(data_path, ev, th)))
            # all_data.append(json_data)
            a_rumor_label = rumor_label[json_data['label']]
            for cas in json_data['cascade']:
                entry = cas.copy()
                a_stance_label = stance_label[cas['support']]
                entry['stance_label'] = a_stance_label
                entry['rumor_label'] = a_rumor_label
                entry['event'] = ev
                entry['thread'] = th
                '''tweet = cas['text']
                a_stance_label = stance_label[cas['support']]
                tid = cas['tweet_id']
                entry['text'] = tweet.lower()
                entry['stance_label'] = a_stance_label
                entry['rumor_label'] = a_rumor_label
                entry['tid'] = tid'''
                all_data[ev][th].append(entry)
    return all_data


if __name__ == '__main__':
    # all_data = utils.load_all_cascades('../' + dataset_name)
    all_data = load_json('../' + dataset_name)
    # tweets = [c['text'] for c in all_data]
    extractor = URLExtract()
    count = len(all_data)
    for ev in all_data:
        for th in all_data[ev]:
            for i, tw in enumerate(all_data[ev][th]):
                urls = extractor.find_urls(tw['text'])
                for u in urls:
                    tw['text'] = tw['text'].replace(u, '<LINK>')

    '''for i, tw in enumerate(all_data):
        # urls = re.findall(url_pattern, tw)
        urls = extractor.find_urls(tw['text'])
        for u in urls:
            tw['text'] = tw['text'].replace(u, '<LINK>')'''

    # stance_labels = [c['stance_label'] for c in all_data]
    # rumor_labels = [c['rumor_label'] for c in all_data]
    features = []
    '''for i, tw in enumerate(all_data):
        lang_feat = extract_lang_features(tw['text'])
        user_feat = extract_user_features(tw['user_info'])'''

    X = list()
    y_rumor = list()
    y_stance = list()
    for ev in all_data:
        for th in all_data[ev]:
            lang_feat_th = []
            user_feat_th = []
            count = 0
            temp_x = []
            temp_y_rumor = []
            temp_y_stance = []
            for i, tw in enumerate(all_data[ev][th]):
                lang_feat = extract_lang_features(tw['text'])
                user_feat = extract_user_features(tw['user_info'])
                lang_feat_th.append(lang_feat)
                user_feat_th.append(user_feat)
                temp_x.insert(i, lang_feat + user_feat)
                temp_y_rumor.insert(i, tw['rumor_label'])
                temp_y_stance.insert(i, tw['stance_label'])

            topic_feat = calc_topic_feat(lang_feat_th, user_feat_th)
            for i in range(0, len(all_data[ev][th])):
                temp_x[i].extend(topic_feat)
            X.extend(temp_x)
            y_rumor.extend(temp_y_rumor)
            y_stance.extend(temp_y_stance)

    X = np.asarray(X)
    y_rumor = [str(a) for a in y_rumor]
    y_stance = [str(a) for a in y_stance]
    y_rumor = np.asarray(y_rumor, dtype=np.int32)
    y_stance = np.asarray(y_stance, dtype=np.int32)
    column_names = list(map(str, range(X.shape[1]))) + ['last']

    import arff
    import pandas as pd
    df = pd.DataFrame(np.concatenate((X, y_rumor.reshape([y_rumor.shape[0], 1])), axis=1), columns=column_names)
    df['Class'] = ''
    for index, row in df.iterrows():
        if row['last'] == 0:
            df.set_value(index, 'Class', 'false')
        if row['last'] == 1:
            df.set_value(index, 'Class', 'true')
        if row['last'] == 2:
            df.set_value(index, 'Class', 'unverified')
    # df['Class'] = df['last'].astype('category')
    df = df.drop('last', 1)

    arff.dump('rumor.arff', df.values, relation='relation name', names=df.columns)
    df.to_csv('rumor.csv', index=False)

    X_train, X_test, y_train_rumor, y_test_rumor, y_train_stance, y_test_stance = \
        train_test_split(X, y_rumor, y_stance, test_size=0.2)
    clf_rumor = SVC() # tree.DecisionTreeClassifier(max_depth=3)
    clf_rumor.fit(X_train, y_train_rumor)
    #clf_stance = SVC() # tree.DecisionTreeClassifier(max_depth=1)
    # clf_stance.fit(X_train, y_train_stance)

    y_pred_rumor = clf_rumor.predict(X_test)
    # y_pred_stance = clf_stance.predict(X_test)

    cr_rumor = classification_report(y_test_rumor, y_pred_rumor, output_dict=True)
    # cr_stance = classification_report(y_test_stance, y_pred_stance, output_dict=True)

    pprint(cr_rumor)
    # pprint(cr_stance)