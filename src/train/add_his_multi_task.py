# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 20:52
# @Author  : Hui Wang


import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.train.dataset_seq import MultiTaskDataset as WeChatDataset
from src.train.trainer import MultiSeqTrainer as FMTrainer
from src.utils import check_path, set_seed, EarlyStopping, collate_fn, get_field_info
import logging
# from src.deepfm import DeepFM
from src.model.pnn_multi import PNN
from src.model.stack_model import StackModel
# from src.lr import LR
# from src.widedeep import WideDeep
from src.model.dcn_multi import DCN

DATASET_PATH = "./data/wedata/wechat_algo_data"
PROCESS_DATA_PATH = './data/wedata/process'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default='./data/his_seq_multi/', type=str)
    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument("--label", default='read_comment', type=str)
    parser.add_argument("--log_name", type=str, required=True)
    parser.add_argument("--tasks", nargs='+', type=str,
                        default=['read_comment', 'like', 'click_avatar', 'forward', "comment", "follow", "favorite"])
    parser.add_argument("--loss_weight", nargs='+', type=float,
                        default=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    parser.add_argument("--pre_trained_feat", nargs='+', type=str, # 'bgm_song_id', 'bgm_singer_id' 'bgm_song_id', 'bgm_singer_id'
                        default=['feedid', 'userid', 'authorid'])

    parser.add_argument("--keyword_tag", nargs='+', type=str,
                        default=['manual_tag_list'])
    
    parser.add_argument("--init_tag", action='store_true')

    # parser.add_argument("--rate", default=0.05, type=float)

    # model args
    parser.add_argument("--model_name", default='DCN', type=str)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--w2v_dir", type=str, default='2')
    parser.add_argument("--w2v_emb", type=int, default=64)
    parser.add_argument("--w2v_window_user", type=int, default=5)
    parser.add_argument("--w2v_window_tag", type=int, default=20)
    parser.add_argument("--w2v_window_keyword", type=int, default=20)
    parser.add_argument("--w2v_window_feed", type=int, default=20)
    parser.add_argument("--w2v_window_author", type=int, default=20)
    parser.add_argument("--w2v_window_song", type=int, default=20)
    parser.add_argument("--w2v_window_singer", type=int, default=20)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--last_day", type=int, default=7)
    parser.add_argument("--feed_seq_len", type=int, default=50)
    parser.add_argument("--feed_seq_len_neg", type=int, default=150)
    
    parser.add_argument("--double_tower", action='store_true')
    # PNN
    # mlp_hidden_size: [128, 256, 128]
    # dropout_prob: 0.0
    # reg_weight: 0
    # use_inner: True
    # use_outer: False
    parser.add_argument("--mlp_hidden_size", nargs='+', type=int, default=[256])
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="hidden dropout p")
    parser.add_argument("--moe_dropout_prob", type=float, default=0.1, help="moe dropout p")

    # WideDeep
    # mlp_hidden_size: [32, 16, 8]
    # dropout_prob: 0.1

    # DCN
    # mlp_hidden_size: [256, 256, 256]
    # cross_layer_num: 6
    # reg_weight: 2
    # dropout_prob: 0.2
    parser.add_argument("--cross_layer_num", type=int, default=2, help="cross layer num")

    # autoint
    # attention_size: 16
    # n_layers: 3
    # num_heads: 2
    # dropout_probs: [0.2, 0.2, 0.2]
    # mlp_hidden_size: [128, 128]
    parser.add_argument("--dropout_probs", nargs='+', type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--n_layers", type=int, default=3, help="autoint self-attention")
    parser.add_argument("--num_heads", type=int, default=2, help="autoint attention heads")
    parser.add_argument("--attention_size", type=int, default=32, help="autoint attention heads")
    parser.add_argument('--has_residual', action='store_false')

    # xDeepFM
    # mlp_hidden_size: [128, 128, 128]
    # reg_weight: 5e-4
    # dropout_prob: 0.2
    # direct: False
    # cin_layer_size: [100, 100, 100]
    parser.add_argument("--cin_layer_size", nargs='+', type=int, default=[100, 100, 100])
    parser.add_argument("--reg_weight", type=float, default=5e-4, help="l2 reg weight")
    parser.add_argument('--direct', action='store_true')

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--lr_decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    if args.do_eval:
        train_file = os.path.join(PROCESS_DATA_PATH, f'online_generate_sample_{args.seed}.csv')
    else:

        train_file = os.path.join(PROCESS_DATA_PATH, f'offline_generate_sample_{args.seed}.csv')
        test_file = os.path.join(PROCESS_DATA_PATH, f'offline_evaluate_all.csv')

    # "read_comment", 'stay', 'play', "like", "click_avatar",  , "comment", "follow", "favorite"
    static_feature = []

    # model used char need specific process
    # b + 'bin' for b in static_feature

    item_static_cols = [b + 'mean' for b in static_feature]
    user_static_cols = [b + 'mean_user' for b in static_feature]

    # 'bgm_singer_id' 'device' 'manual_tag_list',,'bgm_song_id' + ['device', 'videoplayseconds'] 
    need_feat_info = args.pre_trained_feat + ['device', 'videoplayseconds'] + args.keyword_tag + \
                     item_static_cols + user_static_cols

    no_need_info = ['description_char', 'user_his_seq']
    
    field_sources, field2type, field2num = get_field_info(need_feat_info)
    field2num['feedid'] = field2num['feedid'] + 1

    all_field = need_feat_info + no_need_info
    args_str = args.log_name
    
    args.log_file = os.path.join(args.output_dir, args_str + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_file)
    #然后 Handler 对象单独指定了 Formatter 对象单独配置输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(f'*'*60)
    logger.info(f'feature num: {len(all_field)}, {all_field}')
    logger.info(str(args))
    args.logger = logger
    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    if args.do_eval:
        train_dataset = WeChatDataset(train_file, args, all_field,
                                      item_static_col=item_static_cols,
                                      user_static_col=user_static_cols)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

        test_dataset = WeChatDataset(os.path.join(DATASET_PATH, 'test_a.csv'), args, all_field,
                                     item_static_col=item_static_cols,
                                     user_static_col=user_static_cols,
                                     phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

        eval_dataloader = None
    else:

        train_dataset = WeChatDataset(train_file, args, all_field,
                                      item_static_col=item_static_cols,
                                      user_static_col=user_static_cols)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

        eval_dataset = WeChatDataset(test_file, args, all_field,
                                     item_static_col=item_static_cols,
                                     user_static_col=user_static_cols,
                                     phase='eval')
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
        test_dataloader = None
    if args.model_name == 'DeepFM':
        Model = DeepFM
    elif 'PNN' in args.model_name:
        Model = PNN
    elif 'Stack' in args.model_name:
        Model = StackModel
    elif args.model_name == 'WideDeep':
        Model = WideDeep
    elif args.model_name == 'DCN':
        Model = DCN
    elif args.model_name == 'AutoInt':
        Model = AutoInt
    elif args.model_name == 'xDeepFM':
        Model = xDeepFM
    else:
        raise NotImplementedError
    
    model = Model(args, all_field, field_sources, field2type, field2num)

    trainer = FMTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)


    if args.do_eval:
        for epoch in range(args.epochs):
            trainer.train(epoch)
            trainer.save(args.checkpoint_path)
        # checkpoint = 'test_all_feature.pt'
        # args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
        trainer.load(args.checkpoint_path)
        logger.info(f'Load model from {args.checkpoint_path} for test!')
        trainer.test()

    else:
        # trainer.load(args.checkpoint_path)
        # print(f'Load model from {args.checkpoint_path} for continue training!')
        
        early_stopping = EarlyStopping(args.checkpoint_path, patience=1, verbose=True)
       
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.valid(0)
        logger.info(args_str)
        logger.info(result_info)
main()