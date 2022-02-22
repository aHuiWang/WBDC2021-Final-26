# -*- coding: utf-8 -*-
# @Time    : 2021/6/11 15:39
# @Author  : Hui Wang

import fasttext
import argparse
import sys
sys.path.append('.')

from src.utils import *

PROCESS_DATA_PATH = './data/wedata/process'

def train_w2v(w2v_dir, doc, embed_size, window_size, max_words):

    emb_dim = embed_size
    window_size = window_size
    min_count = 1

    text_path = os.path.join(w2v_dir, f'{doc}_seq.txt')
    print(f'load seq from {text_path}')
    embedding_path = os.path.join(w2v_dir, f'{doc}_{emb_dim}_{window_size}_w2v.npy')

    print(f'embedding_size: {emb_dim}, window_size: {window_size}')

    # train
    model = fasttext.train_unsupervised(text_path, model='skipgram', ws=window_size,
                                        minCount=min_count, minn=0, maxn=0,
                                        bucket=10000000, dim=emb_dim)
    words = set(model.words)
    embeddings = []

    for i in range(max_words+1):
        if str(i) in words:
            embeddings.append(model[str(i)])
        else:
            embeddings.append([0.0]*embed_size)

    embeddings = np.array(embeddings)
    np.save(embedding_path, embeddings)
    print(embeddings.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_dir", type=str, default=PROCESS_DATA_PATH)

    args = parser.parse_args()
    w2v_dir = args.w2v_dir
    train_w2v(w2v_dir, doc='char', embed_size=64, window_size=20, max_words=33378)
    # train_w2v(args.w2v_dir, doc='keyword', embed_size=32, window_size=20, max_words=27271)
    # train_w2v(args.w2v_dir, doc='tag', embed_size=32, window_size=20, max_words=353)

    train_w2v(w2v_dir, doc='deepwalk', embed_size=32, window_size=5, max_words=250247)

    # feed=5 author=50/64
    # train_w2v(args.w2v_dir, doc='feed', embed_size=32, window_size=5, max_words=112871)
    train_w2v(w2v_dir, doc='feed', embed_size=32, window_size=20, max_words=112871)

    # train_w2v(w2v_dir, doc='author', embed_size=32, window_size=128, max_words=18789)
    train_w2v(w2v_dir, doc='author', embed_size=32, window_size=20, max_words=18789)

    # 空值0填充

    train_w2v(w2v_dir, doc='song', embed_size=32, window_size=20, max_words=25159)
    train_w2v(w2v_dir, doc='singer', embed_size=32, window_size=20, max_words=17500)

    train_w2v(w2v_dir, doc='word', embed_size=64, window_size=20, max_words=150861)


main()