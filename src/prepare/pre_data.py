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
PROCESS_DATA_PATH = './data/wedata/process'

check_path(PROCESS_DATA_PATH)
check_path(DATASET_PATH)
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")

FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]

ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.04, "forward": 0.05, "comment": 0.05, "follow": 0.05, "favorite": 0.05}
# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online": 14, "offline": 13}


def merge_data():
    data1 = pd.read_csv(os.path.join('./data/wedata/wechat_algo_data1', 'user_action.csv'))
    data2 = pd.read_csv(os.path.join('./data/wedata/wechat_algo_data2', 'user_action.csv'))
    data = pd.concat([data1, data2])
    print(f'Max userid {data.userid.max()}')
    data.to_csv(os.path.join(DATASET_PATH, 'user_action.csv'))

def dynamic_history_length(last=15, pos_len=100, neg_len=300):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    df = pd.read_csv(os.path.join(DATASET_PATH, 'user_action.csv'))
    user_ids_pos = []
    user_seq_pos = []
    dates_pos = []
    lens_pos = []
    
    user_ids_neg = []
    user_seq_neg = []
    dates_neg = []
    lens_neg = []
    
    for right in range(2, 16):
        left = max(0, right-last)
        print(f'Now process {left}-{right}')
        action_df = df[(df["date_"] >= left) & (df["date_"] < right)]
        
        temp = action_df[(action_df['read_comment'] == 1) | (action_df['like'] == 1) | (action_df['click_avatar'] == 1) | (action_df['forward'] == 1) | (action_df['comment'] == 1) | (action_df['follow'] == 1) | (action_df['favorite'] == 1)]
        temp.drop(columns=['date_'])
        df_by_user = temp.groupby('userid')
        for name, group in tqdm.tqdm(df_by_user):
            feedids = group['feedid'].tolist()
            lens_pos.append(len(feedids))
            feedids = [str(feed) for feed in feedids]
            user_ids_pos.append(name)
            user_seq_pos.append(' '.join(feedids[-pos_len:]))
            dates_pos.append(right)
            
#         temp = action_df[(action_df['read_comment'] == 0) & (action_df['like'] == 0) & (action_df['click_avatar'] == 0) & (action_df['forward'] == 0) & (action_df['comment'] == 0) & (action_df['follow'] == 0) & (action_df['favorite'] == 0)]
#         temp.drop(columns=['date_'])
#         df_by_user = temp.groupby('userid')
#         for name, group in tqdm.tqdm(df_by_user):
#             feedids = group['feedid'].tolist()
#             lens_neg.append(len(feedids))
#             feedids = [str(feed) for feed in feedids]
#             user_ids_neg.append(name)
#             user_seq_neg.append(' '.join(feedids[-neg_len:]))
#             dates_neg.append(right)
            
    user_feature_pos = pd.DataFrame()
    user_feature_pos['userid'] = user_ids_pos
    user_feature_pos['user_his_seq'] = user_seq_pos
    user_feature_pos['date_'] = dates_pos
                                                                                        
#     user_feature_neg = pd.DataFrame()
#     user_feature_neg['userid'] = user_ids_neg
#     user_feature_neg['user_his_seq_neg'] = user_seq_neg
#     user_feature_neg['date_'] = dates_neg
#     user_feature = pd.merge(user_feature_pos, user_feature_neg, on=['userid', 'date_'])
    
    user_feature_pos.to_csv(os.path.join(PROCESS_DATA_PATH, 'user_feature.csv'), index=False)
    print(f'Pos: min_len:{np.min(lens_pos)}, max_len:{np.max(lens_pos)}, avg_len:{np.mean(lens_pos)}.')
#     print(f'Neg: min_len:{np.min(lens_neg)}, max_len:{np.max(lens_neg)}, avg_len:{np.mean(lens_neg)}.')   

def generate_sample_multi(stage="online", seed=15, ratio=0.15):
    end_day = STAGE_END_DAY[stage]
    evaluate_day = STAGE_END_DAY[stage] + 1

    df = pd.read_csv(USER_ACTION)

    # 线下/线上训练
    # 同行为取按时间最近的样本
    for action in ACTION_LIST:
        print(f'Drop duplicates {action}')
        df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
    # 负样本下采样
    action_df = df[(df["date_"] <= end_day)]
    print('Begin negative sample!')
    df_neg = action_df[(action_df['read_comment'] == 0) & (action_df['like'] == 0) & (action_df['click_avatar'] == 0) & (action_df['forward'] == 0) & (action_df['comment'] == 0) & (action_df['follow'] == 0) & (action_df['favorite'] == 0)]
    df_pos = action_df[(action_df['read_comment'] == 1) | (action_df['like'] == 1) | (action_df['click_avatar'] == 1) | (action_df['forward'] == 1) | (action_df['comment'] == 1) | (action_df['follow'] == 1) | (action_df['favorite'] == 1)]
    df_neg = df_neg.sample(frac=ratio, random_state=seed, replace=False)
    df_all = pd.concat([df_neg, df_pos])

    df_all = df_all.sample(frac=1, random_state=42, replace=False)
    col = ["userid", "feedid", "date_", "device"] + ACTION_LIST
    # _{ACTION_SAMPLE_RATE[action]}
    file_name = os.path.join(PROCESS_DATA_PATH, f"{stage}_generate_sample_{seed}.csv")
    print('Save to: %s'%file_name)
    df_all[col].to_csv(file_name, index=False)   

merge_data()
print(f'Begin Sample Generate History Sequence ...')
dynamic_history_length()
print(f'Begin Sample Multitask Data ...')
generate_sample_multi('online', seed=15, ratio=0.15)
generate_sample_multi('online', seed=42, ratio=0.2)
generate_sample_multi('online', seed=1996, ratio=0.2)
generate_sample_multi('online', seed=2021, ratio=0.2)
generate_sample_multi('online', seed=2048, ratio=0.2)