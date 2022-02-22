# -*- coding: utf-8 -*-
# @Time    : 2021/6/11 15:40
# @Author  : Hui Wang

import os
import numpy as np
import pandas as pd
import math
import random
import os
import torch
import json


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f'{path} removed!')

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)

def collate_fn(batch):
    ret_batch = dict()
    for k in batch[0].keys():
        ret_batch[k] = torch.cat([b[k].unsqueeze(0) for b in batch])
        # ret_batch[k] = [b[k] for b in batch]
    return ret_batch

# description, ocr, asr, description_char, ocr_char, asr_char,
# manual_keyword_list, machine_keyword_list, manual_tag_list, machine_tag_list
# feeds = ['feedid', 'authorid', 'videoplayseconds', bgm_song_id,bgm_singer_id,

# 'userid', 'feedid', 'date_', 'device', 'read_comment', 'comment', 'like', 'play', 'stay', 'click_avatar', 'forward', 'follow', 'favorite'
# authorid max: 18788, unique: 18789,0 in id:True
# bgm_song_id max: 25158.0, unique: 25160,0 in id:True
# bgm_singer_id max: 17499.0, unique: 17501,0 in id:True
def get_field_info(field_names):
    # ['userid' *]
    field_sources = ['user'] + ['item'] * (len(field_names)-1)
    field2type = {
        'userid': 'token',
        'feedid': 'token',
        'authorid': 'token',
        'bgm_song_id': 'token',
        'bgm_singer_id': 'token',
        'videoplayseconds': 'float',
        'manual_tag_list': 'token_seq',
        'manual_keyword_list': 'token_seq',
        'device': 'token',

    }

    field2num = {
        'userid': 250248,
        'feedid': 112872,
        'authorid': 18790,
        'bgm_song_id': 25160,
        'bgm_singer_id': 17501,
        'videoplayseconds': 1,
        'manual_tag_list': 354,
        'manual_keyword_list': 27272,
        'device': 3,
    }

    actions = ['read_comment', 'comment', 'like', 'play', 'stay', 'click_avatar', 'forward', 'follow', 'favorite']
    for action in actions:
        for agg in ['sum', 'mean']:
            field2type[f'{action}{agg}'] = 'float'
            field2type[f'{action}{agg}_user'] = 'float'
            field2num[f'{action}{agg}'] = 1
            field2num[f'{action}{agg}_user'] = 1
        for agg in ['bin']:
            field2type[f'{action}{agg}'] = 'token'
            field2type[f'{action}{agg}_user'] = 'token'
            field2num[f'{action}{agg}'] = 4
            field2num[f'{action}{agg}_user'] = 4

    return_field2type = {}
    return_field2num = {}
    for field in field_names:
        return_field2type[field] = field2type[field]
        return_field2num[field] = field2num[field]

    return field_sources, return_field2type, return_field2num