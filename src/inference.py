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

from src.model.stack_model import StackModel

DATASET_PATH = "./data/wedata/wechat_algo_data"
PROCESS_DATA_PATH = './data/wedata/process'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default='./data/model/', type=str)
    parser.add_argument('--testfile', default="./data/wedata/wechat_algo_data/test_a.csv", type=str)
    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument("--label", default='read_comment', type=str)
    parser.add_argument("--log_name", type=str, default="submit")
    parser.add_argument("--tasks", nargs='+', type=str,
                        default=['read_comment', 'like', 'click_avatar', 'forward', "comment", "follow", "favorite"])
    parser.add_argument("--loss_weight", nargs='+', type=float,
                        default=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    
    parser.add_argument("--init_tag", action='store_true')

    # parser.add_argument("--rate", default=0.05, type=float)

    # model args
    parser.add_argument("--model_name", default='PNN2', type=str)
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
    
    parser.add_argument("--mlp_hidden_size", nargs='+', type=int, default=[128, 128, 128])
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="hidden dropout p")
    parser.add_argument("--moe_dropout_prob", type=float, default=0.1, help="moe dropout p")
    parser.add_argument("--cross_layer_num", type=int, default=2, help="cross layer num")
    parser.add_argument("--reg_weight", type=float, default=5e-4, help="l2 reg weight")

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

    

    # 'bgm_singer_id' 'device' 'manual_tag_list',,'bgm_song_id' + ['device', 'videoplayseconds'] 
    need_feat_info = ['feedid', 'userid', 'authorid', 'bgm_song_id', 'bgm_singer_id'] + ['device', 'videoplayseconds'] +['manual_tag_list']

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
    logger.info(str(args))
    args.logger = logger
    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    
    train_dataloader = None

    test_dataset = WeChatDataset(args.testfile, args, all_field,
                                item_static_col=[],
                                user_static_col=[],
                                phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    eval_dataloader = None
    
    field_names1 = ['feedid', 'userid', 'authorid', 'device', 'videoplayseconds', 'manual_tag_list', 'description_char']
    field_names2 = ['feedid', 'userid', 'authorid', 'device', 'videoplayseconds', 'manual_tag_list', 'description_char', 'user_his_seq']
    field_names3 = ['feedid', 'userid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device', 'videoplayseconds', 'manual_tag_list', 'description_char']
    field_names4 = ['feedid', 'userid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device', 'videoplayseconds', 'manual_tag_list', 'description_char', 'user_his_seq']
    logger.info(f'feature_num: {len(field_names1)} {field_names1}')
    logger.info(f'feature_num: {len(field_names2)} {field_names2}')
    logger.info(f'feature_num: {len(field_names3)} {field_names3}')
    logger.info(f'feature_num: {len(field_names4)} {field_names4}')
    
    model = StackModel(args, field_names1, field_names2, field_names3, field_names4, field_sources, field2type, field2num)
    
    checkpoint = 'seed_15.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn1.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_42.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn2.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_1996.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn3.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_2021.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.pnn.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_15_ckp.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn1_.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_42_ckp.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn2_.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_1996_ckp.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn3_.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_2021_ckp.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.dcn4_.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    checkpoint = 'seed_2048_ckp.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint)
    model.pnn_.load_state_dict(torch.load(checkpoint_path))
    logger.info(f'Load model from {checkpoint_path} for test!')
    
    trainer = FMTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)
    trainer.test()

main()