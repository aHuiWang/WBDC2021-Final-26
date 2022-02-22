# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 15:12
# @Author  : Hui Wang

import tqdm
import sys
import numpy as np

sys.path.append('.')

from src.utils import *

# 比赛数据集路径
DATASET_PATH = "./data/wedata/wechat_algo_data"
FEATURE_DATA_PATH = './data/wedata/process'
PRE_DATASET_PATH = "./data/wedata/wechat_algo_data1"

check_path(FEATURE_DATA_PATH)
check_path(DATASET_PATH)
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")


def get_w2v_data():
    seqs = []
    feeds = pd.read_csv(FEED_INFO)
    for col in ['description_char', 'ocr_char', 'asr_char']:
        for keywords in tqdm.tqdm(list(feeds[col])):
            if keywords is np.nan:
                continue
            keywords = [str(int(word) + 1) for word in keywords.split(' ')]
            seqs.append(keywords)

    print(len(seqs))
    f1 = open(os.path.join(FEATURE_DATA_PATH, 'char_seq.txt'), 'w')
    for seq in tqdm.tqdm(seqs):
        f1.write(' '.join(seq) + '\n')
    f1.close()

    seqs = []
    for col in ['description', 'ocr', 'asr']:
        for keywords in tqdm.tqdm(list(feeds[col])):
            if keywords is np.nan:
                continue
            keywords = [str(int(word) + 1) for word in keywords.split(' ')]
            seqs.append(keywords)

    print(len(seqs))
    f1 = open(os.path.join(FEATURE_DATA_PATH, 'word_seq.txt'), 'w')
    for seq in tqdm.tqdm(seqs):
        f1.write(' '.join(seq) + '\n')
    f1.close()


def get_w2v_data_tag():
    def process_token_seq(x, delim, with_prob=False):
        if with_prob:
            token_seq = []
            if ' ' in x:
                tokens = x.split(delim)
                for token in tokens:
                    keyword, prob = token.split(' ')
                    if float(prob) > 0.5:
                        token_seq.append(str(int(keyword) + 1))
            else:
                token_seq.append('0')
            return token_seq

        else:
            token_seq = x.split(delim)
            token_seq = [str(int(i) + 1) for i in token_seq]
            return token_seq

    feeds = pd.read_csv(FEED_INFO)
    feeds[['manual_keyword_list', 'manual_tag_list', 'machine_tag_list', 'machine_keyword_list']] = \
        feeds[['manual_keyword_list', 'manual_tag_list', 'machine_tag_list', 'machine_keyword_list']].fillna('-1')
    # feat == 'manual_tag_list' or feat == 'manual_keyword_list' ';' split
    feeds['manual_keyword_list'] = feeds['manual_keyword_list'].apply(process_token_seq, args=(';'))
    feeds['machine_keyword_list'] = feeds['machine_keyword_list'].apply(process_token_seq, args=(';'))
    feeds['manual_tag_list'] = feeds['manual_tag_list'].apply(process_token_seq, args=(';'))
    feeds['machine_tag_list'] = feeds['machine_tag_list'].apply(process_token_seq, args=(';', True))

    feeds[['bgm_song_id', 'bgm_singer_id']] = feeds[['bgm_song_id', 'bgm_singer_id']].fillna(-1)
    feeds[['bgm_song_id', 'bgm_singer_id']] += 1
    feeds[['bgm_song_id', 'bgm_singer_id']] = feeds[['bgm_song_id', 'bgm_singer_id']].astype(int)
    feeds.set_index('feedid', inplace=True)

    train_actions = pd.read_csv(USER_ACTION)[['userid', 'feedid', 'date_']]
    test_actions = pd.read_csv(os.path.join(DATASET_PATH, 'test_a.csv'))[['userid', 'feedid']]
    test_actions['date_'] = 15

    actions = pd.concat([train_actions, test_actions])
    pre_test_actions_a = pd.read_csv(os.path.join(PRE_DATASET_PATH, 'test_a.csv'))[['userid', 'feedid']]
    test_actions['date_'] = 15
    actions = pd.concat([actions, pre_test_actions_a])
    pre_test_actions_b = pd.read_csv(os.path.join(PRE_DATASET_PATH, 'test_b.csv'))[['userid', 'feedid']]
    test_actions['date_'] = 15
    actions = pd.concat([actions, pre_test_actions_b])

    actions.sort_values(by=['userid', 'date_'], ascending=['True', 'True'], inplace=True)
    df_by_user = actions.groupby('userid')

    # TODO check tag keyword+1
    keyword_seq = []
    tag_seq = []

    feed_seq = []
    author_seq = []
    song_seq = []
    singer_seq = []

    def merge_list(lists):
        return_list = []
        for i in lists:
            return_list.extend(i)
        return return_list

    for name, group in tqdm.tqdm(df_by_user):
        feedids = group['feedid'].tolist()
        feats = feeds.loc[feedids]
        # print(feats)
        feed_seq.append([str(i) for i in feedids])
        for feat in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
            seq = [str(int(i)) for i in feats[feat].tolist()]
            if 'author' in feat:
                author_seq.append(seq)
            elif 'song' in feat:
                song_seq.append(seq)
            else:
                singer_seq.append(seq)
        for feat in ['manual_keyword_list', 'manual_tag_list', 'machine_tag_list', 'machine_keyword_list']:
            # print(feats[feat])
            seq = merge_list(feats[feat].tolist())

            if 'keyword' in feat:
                keyword_seq.append(seq)
            else:
                tag_seq.append(seq)

    f1 = open(os.path.join(FEATURE_DATA_PATH, 'keyword_seq.txt'), 'w')
    for seq in tqdm.tqdm(keyword_seq):
        f1.write(' '.join(seq) + '\n')
    f1.close()

    f1 = open(os.path.join(FEATURE_DATA_PATH, 'tag_seq.txt'), 'w')
    for seq in tqdm.tqdm(tag_seq):
        f1.write(' '.join(seq) + '\n')
    f1.close()

    f1 = open(os.path.join(FEATURE_DATA_PATH, 'feed_seq.txt'), 'w')
    for seq in tqdm.tqdm(feed_seq):
        f1.write(' '.join(seq) + '\n')
    f1.close()

    f1 = open(os.path.join(FEATURE_DATA_PATH, 'author_seq.txt'), 'w')
    for seq in tqdm.tqdm(author_seq):
        f1.write(' '.join(seq) + '\n')
    f1.close()

    f1 = open(os.path.join(FEATURE_DATA_PATH, 'song_seq.txt'), 'w')
    for seq in tqdm.tqdm(song_seq):
        f1.write(' '.join(seq) + '\n')
    f1.close()

    f1 = open(os.path.join(FEATURE_DATA_PATH, 'singer_seq.txt'), 'w')
    for seq in tqdm.tqdm(singer_seq):
        f1.write(' '.join(seq) + '\n')
    f1.close()


def get_deep_walk():
    random.seed(42)
    train_actions = pd.read_csv(USER_ACTION)[['userid', 'feedid', 'date_']]
    test_actions = pd.read_csv(os.path.join(DATASET_PATH, 'test_a.csv'))[['userid', 'feedid']]
    test_actions['date_'] = 15
    actions = pd.concat([train_actions, test_actions])

    pre_test_actions_a = pd.read_csv(os.path.join(PRE_DATASET_PATH, 'test_a.csv'))[['userid', 'feedid']]
    test_actions['date_'] = 15
    actions = pd.concat([actions, pre_test_actions_a])

    pre_test_actions_b = pd.read_csv(os.path.join(PRE_DATASET_PATH, 'test_b.csv'))[['userid', 'feedid']]
    test_actions['date_'] = 15
    actions = pd.concat([actions, pre_test_actions_b])

    user_num = actions.userid.max()
    df_by_item = actions.groupby('feedid')
    df_by_user = actions.groupby('userid')

    one2many = {}

    for name, group in tqdm.tqdm(df_by_user):
        feedids = group['feedid'].tolist()
        one2many[name] = [(i + user_num) for i in feedids]

    for name, group in tqdm.tqdm(df_by_item):
        userids = group['userid'].tolist()
        one2many[name + user_num] = [i for i in userids]

    path_length = 20
    sentences = []
    length = []
    for _ in range(10):
        for key in one2many.keys():
            sentence = [key]
            while len(sentence) != path_length:
                key = random.sample(one2many[sentence[-1]], 1)[0]
                if len(sentence) >= 2 and key == sentence[-2]:
                    break
                else:
                    sentence.append(key)

            sentences.append(sentence)
            length.append(len(sentence))
            if len(sentences) % 100000 == 0:
                print(len(sentences))

    print(np.mean(length))
    print(len(sentences))
    random.shuffle(sentences)
    f1 = open(os.path.join(FEATURE_DATA_PATH, 'deepwalk_seq.txt'), 'w')
    for seq in tqdm.tqdm(sentences):
        # if seq[0] > user_num:
        #     seq = [str(i) for i in seq[1::2]]
        # else:
        #     seq = [str(i) for i in seq[::2]]
        seq = [str(i) for i in seq]
        f1.write(' '.join(seq) + '\n')
    f1.close()

get_w2v_data_tag()
get_w2v_data()
get_deep_walk()