import json
import time
import pickle
import datetime
from os import listdir
from os.path import join


def process_cascade(cascade):
    data_cascade = {}

    data_cascade['event'] = cascade['event']
    data_cascade['label'] = cascade['label']
    data_cascade['thread'] = cascade['thread']
    cas_list = []
    start_time = time.mktime(datetime.datetime.strptime(
        cascade['cascade'][0]['time'], "%Y-%m-%d %H:%M:%S").timetuple())

    for i, item in enumerate(cascade['cascade']):
        t_item = item.copy()
        '''t_item['user_id'] = item['user_id']
        t_item['text'] = item['text']
        t_item['time'] = item['time']'''
        tsmp = time.mktime(datetime.datetime.strptime(item['time'], "%Y-%m-%d %H:%M:%S").timetuple())
        t_item['time_stamp'] = tsmp
        t_item['time_since_start'] = tsmp - start_time
        if i == 0:
            t_item['time_since_last_event'] = tsmp - start_time
        else:
            t_item['time_since_last_event'] = t_item['time_since_start'] - cas_list[i-1]['time_since_start']
        '''if i == 0:
            t_item['time_since_last_event'] = 0.0
        elif i == 1:
            t_item['time_since_last_event'] = tsmp - start_time
        else:
            t_item['time_since_last_event'] = t_item['time_since_start'] - cas_list[i-1]['time_since_start']'''
        '''if i == 0:
            t_item['time_since_start'] = 0.0'''

        cas_list.append(t_item)
    data_cascade['cascade'] = cas_list
    return data_cascade


if __name__ == '__main__':
    all_events = listdir('cascades')
    train_cascades = []
    test_cascades = []
    for event in all_events:
        event_dir = join('cascades', event)
        for entry in listdir(join('cascades', event, 'train')):
            entry_data = json.load(open(join('cascades', event, 'train', entry)))
            # print(entry_data)
            entry_data['event'] = event
            data_cascade = process_cascade(cascade=entry_data)
            train_cascades.append(data_cascade)
        for entry in listdir(join('cascades', event, 'test')):
            entry_data = json.load(open(join('cascades', event, 'test', entry)))
            # print(entry_data)
            entry_data['event'] = event
            data_cascade = process_cascade(cascade=entry_data)
            test_cascades.append(data_cascade)

    with open('pheme_train.pkl', 'wb+') as write_file:
        pickle.dump(train_cascades, write_file)
    with open('pheme_test.pkl', 'wb+') as write_file:
        pickle.dump(test_cascades, write_file)