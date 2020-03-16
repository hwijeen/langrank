import pickle
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
# import pytest
import numpy as np
from langrank import prepare_train_file, train, rank_to_relevance
from preprocessing import build_preprocess
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score

def test_train_mt():
    langs = ["aze", "ben", "fin"]
    datasets = [os.path.join(root, "sample-data", "ted-train.orig.{}".format(l)) for l in langs]
    seg_datasets = [os.path.join(root, "sample-data", "ted-train.orig.spm8000.{}".format(l)) for l in langs]
    rank = [[0, 1, 2], [1, 0, 2], [2, 1, 0]] # random
    tmp_dir = "tmp"
    prepare_train_file(datasets=datasets, segmented_datasets=seg_datasets,
                       langs=langs, rank=rank, tmp_dir=tmp_dir, task="MT")
    output_model = "{}/model.txt".format(tmp_dir)
    train(tmp_dir=tmp_dir, output_model=output_model)
    assert os.path.isfile(output_model)

def train_olid(exclude_lang=None):
    langs= ['ara', 'dan', 'ell', 'eng', 'tur']
    data_dir = 'datasets/olid/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    rank = [[0, 4, 2, 1, 3],
            [2, 0, 4, 1, 3],
            [2, 4, 0, 1, 3],
            [3, 1, 4, 0, 2],
            [3, 1, 4, 2, 0]]

    if exclude_lang is not None: # for cross validation
        exclude_idx = langs.index(exclude_lang)
        langs.pop(exclude_idx)
        rank = rerank(rank, exclude_idx)

    tmp_dir = "tmp"
    preprocess = build_preprocess()
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task="OLID", preprocess=preprocess)
    output_model = "{}/olid_model.txt".format(tmp_dir)
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'transfer_nr', 'transfer_vr', 'distance_n2v',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    train(tmp_dir=tmp_dir, output_model=output_model,
          feature_name=feature_name, task="OLID")
    assert os.path.isfile(output_model)

def rerank(rank, without_idx=None):
    for i, r in enumerate(rank):
        r.pop(without_idx)
        reranked = rankdata(r) - 1
        rank[i] = reranked
    return rank

def train_sa(exclude_lang=None):
    langs = ['ara', 'chi', 'dut', 'eng', 'fre',
             'ger', 'jap', 'kor', 'per', 'rus',
             'spa', 'tam', 'tha', 'tur']
    data_dir = 'datasets/sa/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    rank = pickle.load(os.path.join(data_dir, 'rankings.pkl'))

    if exclude_lang is not None: # for cross validation
        exclude_idx = langs.index(exclude_lang)
        langs.pop(exclude_idx)
        rank = rerank(rank, exclude_idx)

    tmp_dir = 'tmp'
    preprocess = None
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task="SA", preprocess=preprocess)
    output_model = "{}/sa_model.txt".format(tmp_dir)
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'transfer_nr', 'transfer_vr', 'distance_n2v',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    train(tmp_dir=tmp_dir, output_model=output_model,
          feature_name=feature_name, task="SA")
    assert os.path.isfile(output_model)


def evaluate(pred_rank, gold_rank, k=3):
    # NDCG@3 as default
    num_lang = len(pred)
    pred_rel = rank_to_relevance(pred_rank, num_lang)
    gold_rel = rank_to_relevance(gold_rank, num_lang)
    pred_rel = np.expand_dims(pred_rel, axis=0)
    gold_rel = np.expand_dims(gold_rel, axis=0)
    return ndcg_score(y_score=pred_rel, y_true=gold_rel, k=k)


if __name__ == '__main__':
    train_olid(exclude_lang='eng')
    # train_sa(exclude_lang='eng')

    # pred = [0,1,2,4,3]
    # gold = [0,4,3,2,1]
    # print(evaluate(pred_rank=pred, gold_rank=gold))