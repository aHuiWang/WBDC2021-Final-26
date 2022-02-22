# -*- coding: utf-8 -*-
# @Time    : 2021/6/13 21:26
# @Author  : Hui Wang

from torch.utils.data import Dataset
from src.utils import *

DATASET_PATH = "./data/wedata/wechat_algo_data"
PROCESS_DATA_PATH = './data/wedata/process'

class WeChatDataset(Dataset):
    def __init__(self, data_file, args, all_field,
                 user_static_col, item_static_col, phase='train'):

        action_data_df = pd.read_csv(data_file)
        args.logger.info(f'Load {data_file}: {action_data_df.shape}')

        if phase == 'train' and args.fold != -1:
            data_len = action_data_df.shape[0]
            len_data = int(data_len*0.1)
            args.logger.info(len_data*args.fold, len_data*(args.fold+1))
            if args.fold == 0:
                action_data_df = action_data_df.iloc[len_data*(args.fold+1):]
                # + action_data_df[len_data*args.group:len_data*(args.group+1)]
            else:
                df_1 = action_data_df.iloc[:len_data * args.fold]
                df_1.reset_index(inplace=True)
                df_2 = action_data_df.iloc[len_data*(args.fold+1):]
                df_2.reset_index(inplace=True)
                action_data_df = pd.concat([df_1, df_2])

        # action_data_df.reset_index(inplace=True)
        args.logger.info(f'train: {action_data_df.shape}')
        if phase == 'test':
            action_data_df['date_'] = 15

        item_feat = pd.read_csv(os.path.join(DATASET_PATH, 'feed_info.csv'))

        item_feat[['bgm_song_id', 'bgm_singer_id']] = item_feat[['bgm_song_id', 'bgm_singer_id']].fillna(-1)
        item_feat[['bgm_song_id', 'bgm_singer_id']] += 1
        item_feat[['bgm_song_id', 'bgm_singer_id']] = item_feat[['bgm_song_id', 'bgm_singer_id']].astype(int)

        item_feat.rename(columns={'description': 'description_word',
                                  'asr': 'asr_word',
                                  'ocr': 'ocr_word'}, inplace=True)

        self.data = pd.merge(action_data_df, item_feat, on='feedid', how='left')

        # item static feature
        # item_static_feat = pd.read_csv(os.path.join('./data/feature', 'feedid_feature.csv'))
        # self.data = pd.merge(self.data, item_static_feat, on=['feedid'],  # 'date_'
        #                      how='left')

        # user static feature
        # user_static_feat = pd.read_csv(os.path.join('./data/feature', 'userid_feature.csv'))
        # self.data = pd.merge(self.data, user_static_feat, on=['userid'], # 'date_'
        #                      how='left', suffixes=("", "_user"))

        if 'videoplayseconds' in all_field:
            self.data['videoplayseconds'] = np.log(self.data['videoplayseconds']+1.0)

        all_static_col = item_static_col+user_static_col

        if len(all_static_col) > 0:
            self.data[all_static_col] = self.data[all_static_col].fillna(0.0)
            # for col in all_static_col:
            #     if 'sum' in col:
            #         self.data[col] = np.log(self.data[col] + 1.0)

        self.all_field = all_field + user_static_col + item_static_col

        self.label = args.label
        self.description_char_len = 80
        self.description_len = 80

        self.asr_char_len = 300
        self.asr_len = 150
        self.tags_len = 8
        self.phase = phase

        # users_df = pd.read_csv(os.path.join(PROCESS_DATA_PATH, 'user_id.csv'), header=0)

        # a = dict(users_df['userid'])
        # user_id 从1开始
        # id2user = {key: value for key, value in a.items()}
        # self.user2id = {value: key for key, value in id2user.items()}

    def process_token_seq(self, token_seq, max_len):
        token_seq = [int(i)+1 for i in token_seq]
        padding = max_len - len(token_seq)
        pad_seq = token_seq + [0] * padding
        pad_seq = pad_seq[:max_len]
        return pad_seq

    def __len__(self):
        return self.data.shape[0]
    
    def get_feature_tensor(self, inter):
        feedid = inter['feedid']
        userid = inter['userid']
        data = {
            'userid': torch.tensor(userid, dtype=torch.long),
            'feedid': torch.tensor(feedid, dtype=torch.long),
        }

        # omit userid feedid
        for feat in self.all_field:
            if feat == 'userid' or feat == 'feedid':
                continue
            if feat == 'description_char':
                if inter[feat] is np.nan:
                    token_seq = [0]
                else:
                    token_seq = inter[feat].split(' ')
                data[feat] = torch.tensor(self.process_token_seq(token_seq, self.description_char_len), dtype=torch.long)

            elif feat == 'description_word':
                if inter[feat] is np.nan:
                    token_seq = [0]
                else:
                    token_seq = inter[feat].split(' ')
                data[feat] = torch.tensor(self.process_token_seq(token_seq, self.description_len), dtype=torch.long)

            elif feat == 'ocr_char' or feat == 'asr_char':
                if inter[feat] is np.nan:
                    token_seq = [0]
                else:
                    token_seq = inter[feat].split(' ')
                data[feat] = torch.tensor(self.process_token_seq(token_seq, self.asr_char_len), dtype=torch.long)

            elif feat == 'ocr_word' or feat == 'asr_word':
                if inter[feat] is np.nan:
                    token_seq = [0]
                else:
                    token_seq = inter[feat].split(' ')
                data[feat] = torch.tensor(self.process_token_seq(token_seq, self.asr_len), dtype=torch.long)

            elif feat == 'manual_tag_list' or feat == 'manual_keyword_list':
                if inter[feat] is np.nan:
                    token_seq = [0]
                else:
                    token_seq = inter[feat].split(';')
                data[feat] = torch.tensor(self.process_token_seq(token_seq, self.tags_len), dtype=torch.long)

            elif feat == 'authorid' or feat == 'device' or feat == 'bgm_song_id' or feat == 'bgm_singer_id':
                data[feat] = torch.tensor(inter[feat], dtype=torch.long)

            elif feat == 'feed_emb':
                continue

            elif 'bin' in feat:
                data[feat] = torch.tensor(inter[feat], dtype=torch.long)

            else:
                data[feat] = torch.tensor(inter[feat], dtype=torch.float)

        return data
    
    def __getitem__(self, index):
        inter = self.data.iloc[index]
        
        data = self.get_feature_tensor(inter)
        if self.phase == 'test':
            label = torch.tensor(0.0, dtype=torch.float)
        else:
            label = torch.tensor(inter[self.label], dtype=torch.float)
        data[self.label] = label
        return data


class MultiTaskDataset(WeChatDataset):
    def __init__(self, data_file, args, all_field,
                 user_static_col, item_static_col, phase='train'):
        super(MultiTaskDataset, self).__init__(
            data_file, args, all_field,
            user_static_col, item_static_col, phase
        )
    
    def __getitem__(self, index):
        inter = self.data.iloc[index]
        
        data = self.get_feature_tensor(inter)
        actions = ['read_comment', 'like', 'click_avatar', 'forward', "comment", "follow", "favorite"]
        if self.phase == 'test':
            for action in actions:
                data[action] = torch.tensor(0.0, dtype=torch.float)
        else:
            for action in actions:
                data[action] = torch.tensor(inter[action], dtype=torch.float)
        return data