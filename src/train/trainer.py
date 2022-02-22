# -*- coding: utf-8 -*-
# @Time    : 2021/6/13 22:16
# @Author  : Hui Wang


import numpy as np
import pandas as pd
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from src.train.optimization import BertAdam, FocalLoss

from src.evaluation import uAUC
from src.utils import check_path
import os

PROCESS_DATA_PATH = './data/wedata/process'
SUBMIT_PATH = './data/submission/'

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = args.cuda_condition
        self.device = args.device

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        # self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim = BertAdam(self.model.parameters(), lr=self.args.lr, warmup=self.args.warmup, weight_decay=self.args.weight_decay)
        self.args.logger.info("Total Parameters: {0}".format(sum([p.nelement() for p in self.model.parameters()])))
        # self.criterion = FocalLoss()
        self.criterion = nn.BCELoss()
        self.total_auc = 0
        self.total_loss = 100
        if args.lr_decay:
            self.scheduler = ReduceLROnPlateau(self.optim, 
                                               mode='max', 
                                               factor=args.lr_decay, 
                                               verbose=True,
                                               threshold=1e-6,
                                               patience=0)
        
        check_path(SUBMIT_PATH)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self):
        self.model.eval()

        pred_list = None
        rec_data_iter = tqdm.tqdm(enumerate(self.test_dataloader),
                                  desc="Submit",
                                  total=len(self.test_dataloader),
                                  bar_format="{l_bar}{r_bar}")

        for i, batch in rec_data_iter:
            interaction = {}
            for field, data in batch.items():
                interaction[field] = data.to(self.device)
            batch_pred = self.model.predict(interaction)

            if i == 0:
                pred_list = batch_pred.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred.cpu().data.numpy(), axis=0)

        if self.args.fold == -1:
            np_path = os.path.join(SUBMIT_PATH, f'{self.args.model_name}_{self.args.label}.npy')
            np.save(np_path, pred_list)
        else:
            np_path = os.path.join(SUBMIT_PATH, f'{self.args.model_name}_{self.args.label}_{self.args.fold}.npy')
            np.save(np_path, pred_list)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def iteration(self, epoch, dataloader, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            if self.args.lr_decay:
                self.scheduler.step(self.total_auc)
                decay_lr = self.optim.param_groups[0]['lr']
                lr_str = f'AUC:{self.total_auc:.6f}, decayed_lr:{decay_lr}'
                self.args.logger.info(lr_str)
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            pred_list = None
            label_list = None
            user_list = None

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)

                interaction = {}
                for field, data in batch.items():
                    interaction[field] = data.to(self.device)
                batch_pred, l2_loss = self.model.calculate_loss(interaction)
                batch_label = interaction[self.args.label]
                batch_pred = torch.sigmoid(batch_pred)
                loss = self.criterion(batch_pred, batch_label)
                loss = loss + l2_loss
                batch_user = interaction['userid']

                # loss = self.criterion(batch_pred, batch_label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

                if i == 0:
                    pred_list = batch_pred.cpu().data.numpy()
                    label_list = batch_label.cpu().data.numpy()
                    user_list = batch_user.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred.cpu().data.numpy(), axis=0)
                    label_list = np.append(label_list, batch_label.cpu().data.numpy(), axis=0)
                    user_list = np.append(user_list, batch_user.cpu().data.numpy(), axis=0)

            # 全猜0 会得到 0.5
            score = uAUC(labels=label_list, preds=pred_list, user_id_list=user_list)

            post_fix = {
                "train_epoch": epoch,
                'uAUC': '{:.6f}'.format(score),
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss)
            }
            self.args.logger.info(post_fix)

        else:
            self.model.eval()
            
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            pred_list = None
            label_list = None
            user_list = None

            for i, batch in rec_data_iter:
                interaction = {}
                for field, data in batch.items():
                    interaction[field] = data.to(self.device)
                    
                batch_pred, l2_loss = self.model.calculate_loss(interaction)
                batch_label = interaction[self.args.label]
                batch_pred = torch.sigmoid(batch_pred)
                loss = self.criterion(batch_pred, batch_label)
                loss = loss + l2_loss
                batch_user = interaction['userid']
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
                if i == 0:
                    pred_list = batch_pred.cpu().data.numpy()
                    label_list = batch_label.cpu().data.numpy()
                    user_list = batch_user.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred.cpu().data.numpy(), axis=0)
                    label_list = np.append(label_list, batch_label.cpu().data.numpy(), axis=0)
                    user_list = np.append(user_list, batch_user.cpu().data.numpy(), axis=0)

            score = uAUC(labels=label_list, preds=pred_list, user_id_list=user_list)
            self.total_auc = score
            post_fix = {
                "valid_epoch": epoch,
                "uAUC": '{:.6}'.format(score),
                "valid_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "valid_cur_loss": '{:.4f}'.format(rec_cur_loss)
            }
            self.args.logger.info(str(post_fix))
            return [score], post_fix
    
    def init_embedding(self, args, model):
        weight_path = os.path.join(PROCESS_DATA_PATH, f"char_{args.w2v_emb}_20_w2v.npy")
        pretrained_weight = np.load(weight_path).astype('float32')
        model.char_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))
        post_fix = f'Init char Embedding from {weight_path}'
        self.args.logger.info(post_fix)
        self.args.logger.info('-'*50)
        # and args.model_name == 'DCN' and args.label == 'like'
        if 'userid' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('userid')
            left = model.token_field_offsets[index]
            right = model.token_field_offsets[index + 1]
            print(model.token_field_offsets, left, right)

            weight_path = os.path.join(PROCESS_DATA_PATH, f"deepwalk_{args.embedding_size}_{args.w2v_window_user}_w2v.npy")
            pretrained_weight = np.load(weight_path).astype(
                'float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init userid Embedding from {weight_path}'
            self.args.logger.info(post_fix)

        if 'authorid' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('authorid')
            left = model.token_field_offsets[index]
            try:
                right = model.token_field_offsets[index + 1]
            except:
                right = model.token_embedding_table.embedding.weight.shape[0]
            
            print(model.token_field_offsets, left, right)
            
            weight_path = os.path.join(PROCESS_DATA_PATH,
                                       f"author_{args.embedding_size}_{args.w2v_window_author}_w2v.npy")

            pretrained_weight = np.load(weight_path).astype(
                'float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init authorid Embedding from {weight_path}'
            self.args.logger.info(post_fix)

        if 'bgm_song_id' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('bgm_song_id')
            left = model.token_field_offsets[index]
            right = model.token_field_offsets[index + 1]
            print(model.token_field_offsets, left, right)
            weight_path = os.path.join(PROCESS_DATA_PATH,
                                       f'song_{args.embedding_size}_{args.w2v_window_song}_w2v.npy')
            pretrained_weight = np.load(weight_path).astype('float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init bgm_song_id Embedding from {weight_path}'
            self.args.logger.info(post_fix)
                
        if 'bgm_singer_id' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('bgm_singer_id')
            left = model.token_field_offsets[index]
            right = model.token_field_offsets[index + 1]
            print(model.token_field_offsets, left, right)
            weight_path = os.path.join(PROCESS_DATA_PATH,
                                       f"singer_{args.embedding_size}_{args.w2v_window_singer}_w2v.npy")
            pretrained_weight = np.load(weight_path).astype('float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init bgm_singer_id Embedding from {weight_path}'
            self.args.logger.info(post_fix)

        self.args.logger.info('-'*50)
        if 'manual_tag_list' in model.token_seq_field_names and self.args.init_tag:
            index = model.token_seq_field_names.index('manual_tag_list')
            weight_path = f"src/prepare/tag_emb.npy"
            # weight_path = os.path.join(PROCESS_DATA_PATH, f"tag_{args.embedding_size}_{args.w2v_window_tag}_w2v.npy")
            pretrained_weight = np.load(weight_path).astype('float32')
            model.token_seq_embedding_table[index].weight.data.copy_(torch.from_numpy(pretrained_weight))
            post_fix = f'Init tag Embedding from {weight_path}'
            self.args.logger.info(post_fix)
        
        if 'manual_keyword_list' in model.token_seq_field_names and self.args.init_tag:
            index = model.token_seq_field_names.index('manual_keyword_list')
            weight_path = os.path.join(PROCESS_DATA_PATH, f"keyword_{args.embedding_size}_{args.w2v_window_keyword}_w2v.npy")
            pretrained_weight = np.load(weight_path).astype('float32')
            model.token_seq_embedding_table[index].weight.data.copy_(torch.from_numpy(pretrained_weight))
            post_fix = f'Init keyword Embedding! from keyword_{args.embedding_size}_{args.w2v_window_keyword}_w2v.npy'
            self.args.logger.info(post_fix)


class FMTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        super(FMTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        try:
            self.init_embedding(args, self.model)
            self.init_feed_embedding(args, self.model)
        except:
            pass

    def init_feed_embedding(self, args, model):
        if 'feedid' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('feedid')
            left = model.token_field_offsets[index]
            right = model.token_field_offsets[index + 1]
            print(model.token_field_offsets, left, right)
            weight_path = os.path.join(PROCESS_DATA_PATH,
                                       f"feed_{args.embedding_size}_{args.w2v_window_feed}_w2v.npy")

            pretrained_weight = np.load(weight_path).astype(
                'float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init feedid Embedding from {weight_path}'
            self.args.logger.info(post_fix)
            
class SeqTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        super(SeqTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

        self.init_embedding(args, self.model)
        self.init_feed_embedding(args, self.model)

    def init_feed_embedding(self, args, model):
        if 'feedid' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('feedid')
            left = model.token_field_offsets[index] + 1
            right = model.token_field_offsets[index + 1]
            print(model.token_field_offsets, left, right)
            weight_path = os.path.join(PROCESS_DATA_PATH,
                                       f"feed_{args.embedding_size}_{args.w2v_window_feed}_w2v.npy")

            pretrained_weight = np.load(weight_path).astype(
                'float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init feedid Embedding from {weight_path}'
            self.args.logger.info(post_fix)
            
            
class MultiTrainer(FMTrainer):
    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        super(MultiTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.criterion = nn.BCELoss()
#         param_optimizer = list(self.model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and p.requires_grad)],
#              'weight_decay': 0.01,
#              'lr': args.learning_rate,
#              'warmup': args.warmup_proportion},

#             {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and p.requires_grad)],
#              'weight_decay': 0.0,
#              'lr': args.learning_rate,
#              'warmup': args.warmup_proportion},

#         ]
#         self.optim = BertAdam(optimizer_grouped_parameters,
#                               lr=args.learning_rate,
#                               warmup=args.warmup_proportion)
    def iteration(self, epoch, dataloader, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            avg_loss = [0.0] * len(self.args.tasks)
            cur_loss = [0.0] * len(self.args.tasks)
            pred_list = [None] * len(self.args.tasks)
            label_list = [None] * len(self.args.tasks)
            user_list = None

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)

                interaction = {}
                for field, data in batch.items():
                    interaction[field] = data.to(self.device)
                losses = []
                batch_preds = self.model.calculate_loss(interaction)
                batch_preds = torch.sigmoid(batch_preds)
                for task_id, task in enumerate(self.args.tasks):
                    batch_pred = batch_preds[:, task_id]
                    batch_label = interaction[task]
                    now_loss = self.criterion(batch_pred, batch_label)
                    avg_loss[task_id] += now_loss.item()
                    cur_loss[task_id] = now_loss.item()
                    losses.append(now_loss)
                    if i == 0:
                        pred_list[task_id] = batch_pred.cpu().data.numpy()
                        label_list[task_id] = batch_label.cpu().data.numpy()
                    else:
                        pred_list[task_id] = np.append(pred_list[task_id], batch_pred.cpu().data.numpy(), axis=0)
                        label_list[task_id] = np.append(label_list[task_id], batch_label.cpu().data.numpy(), axis=0)

                batch_user = interaction['userid']
                if i == 0:
                    user_list = batch_user.cpu().data.numpy()
                else:
                    user_list = np.append(user_list, batch_user.cpu().data.numpy(), axis=0)
                    
                self.optim.zero_grad()
                loss = None
                loss_weight = self.args.loss_weight
                for task_id, task in enumerate(losses):
                    if task_id == 0:
                        loss = loss_weight[task_id] * losses[task_id]
                    else:
                        # if task_id == 1 or (task_id > 1 and random.random() < 0.25):
                        loss += loss_weight[task_id] * losses[task_id]
                loss.backward()
                self.optim.step()

            scores = {}
            score_weights = {
                'read_comment': 4,
                'like': 3,
                'click_avatar': 2,
                'forward': 1,
                'comment': 1,
                'follow': 1,
                'favorite': 1
            }
            # 全猜0 会得到 0.5
            for task_id, task in enumerate(self.args.tasks):
                scores[task] = (uAUC(labels=label_list[task_id], preds=pred_list[task_id], user_id_list=user_list))
            
            score_sum = 0
            score_div = 0
            for task, weights in score_weights.items():
                score_sum += scores[task] * weights
                score_div += weights
            score_sum /= score_div
            post_fix = {
                "train_epoch": epoch,
                'wAUC': '{:.6f}'.format(score_sum)
            }
            self.args.logger.info(post_fix)
            for task_id, task in enumerate(self.args.tasks):
                post_fix = {}
                post_fix[f'{task}_avg_loss'] = '{:.4f}'.format(avg_loss[task_id] / len(rec_data_iter))
                post_fix[f'{task}_cur_loss'] = '{:.4f}'.format(cur_loss[task_id])
                post_fix[f'{task}_uAUC'] = '{:.6f}'.format(scores[task])

                self.args.logger.info(post_fix)

        else:
            self.model.eval()

            pred_list = [None] * len(self.args.tasks)
            label_list = [None] * len(self.args.tasks)
            user_list = None

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)

                interaction = {}
                for field, data in batch.items():
                    interaction[field] = data.to(self.device)
                batch_preds = self.model.calculate_loss(interaction)
                batch_preds = torch.sigmoid(batch_preds)
                for task_id, task in enumerate(self.args.tasks):
                    batch_pred = batch_preds[:, task_id]
                    batch_label = interaction[task]
                    if i == 0:
                        pred_list[task_id] = batch_pred.cpu().data.numpy()
                        label_list[task_id] = batch_label.cpu().data.numpy()
                    else:
                        pred_list[task_id] = np.append(pred_list[task_id], batch_pred.cpu().data.numpy(), axis=0)
                        label_list[task_id] = np.append(label_list[task_id], batch_label.cpu().data.numpy(), axis=0)

                batch_user = interaction['userid']
                if i == 0:
                    user_list = batch_user.cpu().data.numpy()
                else:
                    user_list = np.append(user_list, batch_user.cpu().data.numpy(), axis=0)

            scores = {}
            score_weights = {
                'read_comment': 4,
                'like': 3,
                'click_avatar': 2,
                'forward': 1,
                'comment': 1,
                'follow': 1,
                'favorite': 1
            }
            # 全猜0 会得到 0.5
            for task_id, task in enumerate(self.args.tasks):
                scores[task] = (uAUC(labels=label_list[task_id], preds=pred_list[task_id], user_id_list=user_list))

            score_sum = 0
            score_div = 0
            for task, weights in score_weights.items():
                score_sum += scores[task] * weights
                score_div += weights
            score_sum /= score_div
            post_fix = {
                "valid_epoch": epoch,
                'wAUC': '{:.6f}'.format(score_sum)
            }
            self.args.logger.info(str(post_fix))
            for task_id, task in enumerate(self.args.tasks):
                cur_fix = {}
                cur_fix[f'{task}_uAUC'] = '{:.6f}'.format(scores[task])
                self.args.logger.info(str(cur_fix))
            return [score_sum], post_fix
        
    def test(self):
        self.args.logger.info('Genarate Submit File in MultiTrainer!')
        self.model.eval()
        rec_data_iter = tqdm.tqdm(enumerate(self.test_dataloader),
                                  desc="Submit",
                                  total=len(self.test_dataloader),
                                  bar_format="{l_bar}{r_bar}")

        pred_list = [None] * len(self.args.tasks)
        user_list = None
        feed_list = None

        for i, batch in rec_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            interaction = {}
            for field, data in batch.items():
                interaction[field] = data.to(self.device)
            batch_preds = self.model(interaction)
            for task_id, task in enumerate(self.args.tasks):
                batch_pred = batch_preds[:, task_id]
                if i == 0:
                    pred_list[task_id] = batch_pred.cpu().data.numpy()
                else:
                    pred_list[task_id] = np.append(pred_list[task_id], batch_pred.cpu().data.numpy(), axis=0)

            batch_user = interaction['userid']
            batch_feed = interaction['feedid']
            if i == 0:
                user_list = batch_user.cpu().data.numpy()
                feed_list = batch_feed.cpu().data.numpy()
            else:
                user_list = np.append(user_list, batch_user.cpu().data.numpy(), axis=0)
                feed_list = np.append(feed_list, batch_feed.cpu().data.numpy(), axis=0)
                
        submit = pd.DataFrame()
        submit['userid'] = user_list
        submit['feedid'] = feed_list - 1
        submit['feedid'] = submit['feedid'].astype(int)
        for task_id, task in enumerate(self.args.tasks):
            submit[task] = pred_list[task_id]
        submit.to_csv(os.path.join(SUBMIT_PATH, f'result.csv'), index=False)

class MultiSeqTrainer(MultiTrainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        super(MultiSeqTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def init_feed_embedding(self, args, model):
        if 'feedid' in args.pre_trained_feat:
            index = args.pre_trained_feat.index('feedid')
            # s3 need modify left and weight
            left = model.token_field_offsets[index] + 1
            right = model.token_field_offsets[index + 1]
            print(model.token_field_offsets, left, right)
            # weight_path = f"src/prepare/feed_emb_30.npy"
            weight_path = os.path.join(PROCESS_DATA_PATH, f"feed_{args.embedding_size}_{args.w2v_window_feed}_w2v.npy")
            pretrained_weight = np.load(weight_path).astype(
                'float32')
            model.token_embedding_table.embedding.weight[left:right].data.copy_(
                torch.from_numpy(pretrained_weight))
            post_fix = f'Init feedid Embedding from {weight_path}'
            self.args.logger.info(post_fix)
