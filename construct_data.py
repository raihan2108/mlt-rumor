import os
import time
import json
from os import listdir
from os import makedirs
from operator import itemgetter
from os.path import join, exists

data_dir = 'data/threads/en'


def convert_annotations(annotation, string=True):
    if 'misinformation' in annotation.keys() and 'true' in annotation.keys():
        if int(annotation['misinformation']) == 0 and int(annotation['true']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 0 and int(annotation['true']) == 1:
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 0:
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 1:
            print("OMG! They both are 1!")
            print(annotation['misinformation'])
            print(annotation['true'])
            label = None

    elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation['misinformation']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 1:
            if string:
                label = "false"
            else:
                label = 0

    elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
        print('Has true not misinformation')
        label = None
    else:
        print('No annotations')
        label = None

    return label


def load_stance():
    stances = {}
    with open('data/annotations/en-scheme-annotations.json') as read_json:
        for i, line in enumerate(read_json):
            if line.startswith('#'):
                continue
            entry = json.loads(line)
            ev = entry['event'].strip()
            th = entry['threadid'].strip()
            tid = entry['tweetid'].strip()
            if ev not in stances:
                stances[ev] = {}
            if th not in stances[ev]:
                stances[ev][th] = {}
            if tid not in stances[ev][th]:
                stances[ev][th][tid] = {}
                if 'support' in entry:
                    stances[ev][th][tid]['support'] = entry['support']
                elif 'responsetype-vs-source' in entry:
                        stances[ev][th][tid]['responsetype-vs-source'] = entry['responsetype-vs-source']
                #elif 'responsetype-vs-previous' in entry:
                    #stances[ev][th][tid]['responsetype-vs-previous'] = entry['responsetype-vs-previous']
                else:
                    print('problem %d' % i)
            else:
                print('duplicate entry')

    return stances


'''def get_parent(val, dictionary):
    for k, v in dictionary.items():
        if v == val:
            yield k
        elif isinstance(v, dict):
            for result in get_parent(val, v):
                yield result
            #result = get_parent(val, v)
            # return result
        elif isinstance(v, list):
            for d in v:
                for result in get_parent(val, d):
                    yield result
                #result = get_parent(val, d)
                #return result
    # return -1'''


def get_parent(condition, obj, path=None):
    if path is None:
        path = []

    # In case this is a list
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            new_path = list(path)
            new_path.append(index)
            for result in get_parent(value, condition, path=new_path):
                yield result

    # In case this is a dictionary
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = list(path)
            new_path.append(key)
            for result in get_parent(value, condition, path=new_path):
                yield result

            if condition == key:
                new_path = list(path)
                new_path.append(key)
                yield new_path


def sort_cascade(seq):
    sorted_seq = sorted(seq, key=itemgetter('time'))
    return sorted_seq


def collect_user_feat(entry):
    u_feat = dict()
    u_feat['user_id'] = entry['id_str']
    u_feat['followers_count'] = entry['followers_count']
    u_feat['statuses_count'] = entry['statuses_count']
    u_feat['friends_count'] = entry['friends_count']
    u_feat['verified'] = entry['verified']
    u_feat['listed_count'] = entry['listed_count']

    return u_feat


def process_thread(thread_dir):
    cascade = []
    event = thread_dir.split('/')[-2]
    thread = thread_dir.split('/')[-1]
    reply_tweets = listdir(join(thread_dir, 'reactions'))
    source_tweet_file = listdir(join(thread_dir, 'source-tweets'))[0]
    source_tweet = json.load(open(join(thread_dir, 'source-tweets', source_tweet_file)))
    '''if len(source_tweet) > 1:
        print('more than one source tweets')'''

    retweet_count = source_tweet['retweet_count']
    source_user_id = source_tweet['user']['id_str']
    source_time = source_tweet['created_at']
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(source_time, '%a %b %d %H:%M:%S +0000 %Y'))
    source_tweet_id = source_tweet['id_str']
    source_text = source_tweet['text'].strip().lower()

    st = stances[event][thread][str(source_tweet_id)]
    s_user_info = collect_user_feat(source_tweet['user'])

    with open(join(thread_dir, 'annotation.json')) as raw_file:
        anno = json.load(raw_file)

    label = convert_annotations(annotation=anno, string=True)
    if st['support'] == 'underspecified':
        st['support'] = 'supporting'

    entry = {
        'user_id': source_user_id, 'text': source_text, 'time': ts, 'support': st['support'],
        'tweet_id': source_tweet_id, 'retweet_count': retweet_count,    # , 'parent': '-1'
        'user_info': s_user_info
    }
    cascade.append(entry)
    struct = json.loads(open(join(thread_dir, 'structure.json')).readline())

    reply_tweets_info = {}
    for re in reply_tweets:
        reply_tweets_info[re.split('.')[0]] = {}
        a_reply_tweet = json.load(open(join(thread_dir, 'reactions', re)))
        r_user_info = collect_user_feat(a_reply_tweet['user'])
        parent = a_reply_tweet['in_reply_to_status_id_str']
        r_user_id = a_reply_tweet['user']['id_str']
        r_time = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(a_reply_tweet['created_at'],
                                                                  '%a %b %d %H:%M:%S +0000 %Y'))
        r_tweet_id = a_reply_tweet['id_str']
        r_text = a_reply_tweet['text']
        re_retweet_count = a_reply_tweet['retweet_count']
        try:
            r_stance = stances[event][thread][str(r_tweet_id)]
            # print(r_stance)
        except KeyError:
            # print(event, thread, r_tweet_id)
            r_stance = {'responsetype-vs-source': 'comment'}
        r_stance['parent'] = parent
        reply_tweets_info[re.split('.')[0]]['stance'] = r_stance
        reply_tweets_info[re.split('.')[0]]['time'] = r_time
        reply_tweets_info[re.split('.')[0]]['user_id'] = r_user_id
        reply_tweets_info[re.split('.')[0]]['text'] = r_text
        reply_tweets_info[re.split('.')[0]]['tweet_id'] = r_tweet_id
        reply_tweets_info[re.split('.')[0]]['retweet_count'] = re_retweet_count
        reply_tweets_info[re.split('.')[0]]['user_info'] = r_user_info

    # reply_tweet_ids = [rtf.split()[0] for rtf in reply_tweets]
    for re in reply_tweets_info:
        r_stance = reply_tweets_info[re]['stance']
        parent = r_stance['parent']
        # parent = get_parent(re, struct)
        entry = {'user_id': reply_tweets_info[re]['user_id'],
                 'time': reply_tweets_info[re]['time'],
                 'text': reply_tweets_info[re]['text'],
                 'tweet_id': reply_tweets_info[re]['tweet_id'],
                 'retweet_count': reply_tweets_info[re]['retweet_count'],
                 'user_info': reply_tweets_info[re]['user_info']
                 }
        if parent in list(reply_tweets_info.keys()) + [str(source_tweet_id)]:
            if r_stance['responsetype-vs-source'] == 'comment':
                entry['support'] = 'comment'
            elif r_stance['responsetype-vs-source'] == 'agreed':
                if st['support'] == 'supporting':
                    entry['support'] = 'supporting'
                if st['support'] == 'denying':
                    entry['support'] = 'denying'
                if st['support'] == 'underspecified':
                    entry['support'] = 'supporting'
            elif r_stance['responsetype-vs-source'] == 'disagreed':
                if st['support'] == 'supporting':
                    entry['support'] = 'denying'
                if st['support'] == 'denying':
                    entry['support'] = 'supporting'
                if st['support'] == 'underspecified':
                    entry['support'] = 'denying'
            elif r_stance['responsetype-vs-source'] == 'appeal-for-more-information':
                entry['support'] = 'appeal-for-more-information'
            else:
                print(r_stance['responsetype-vs-source'])
        else:
            entry['support'] = 'comment'
            print('using default: {}'.format(reply_tweets_info[re]['tweet_id']))
        if 'support' not in entry.keys():
            print('%s %s' % (event, thread))
        cascade.append(entry)
    sorted_cascade = sort_cascade(cascade)
    w_entry = {}
    w_entry['label'] = label
    w_entry['thread'] = thread
    w_entry['cascade'] = []
    for i in range(0, len(sorted_cascade)):
        w_entry['cascade'].append(sorted_cascade[i])
    if not exists(join('cascades', event)):
        makedirs(join('cascades', event))
    with open(join('cascades', event, thread + '.json'), 'w+') as write_file:
        json.dump(w_entry, write_file, indent=2)


def process_events(event_dir):
    threads = listdir(event_dir)

    for th in threads:
        thread_dir = join(event_dir, th)
        process_thread(thread_dir)
        #break


if __name__ == '__main__':
    events = listdir(data_dir)
    stances = load_stance()
    for ev in events:
        #if ev == 'ferguson':
        event_dir = join(data_dir, ev)
        process_events(event_dir)
        #break