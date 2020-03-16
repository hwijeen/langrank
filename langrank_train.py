import pickle
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
# import pytest
import numpy as np
from langrank import prepare_train_file, train, rank_to_relevance
from preprocessing import build_preprocess
from scipy.stats import rankdata

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
    else:
        exclude_lang = 'all' # for model file name

    model_save_dir = 'pretrained/OLID'
    tmp_dir = "tmp"
    preprocess = build_preprocess()
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task="OLID", preprocess=preprocess)
    output_model = "{}/olid_model.txt".format(tmp_dir)
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'transfer_nr', 'transfer_vr', 'distance_n2v',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model, feature_name=feature_name, task="OLID")
    assert os.path.isfile(output_model)

def rerank(rank, without_idx=None):
    for i, r in enumerate(rank):
        r.pop(without_idx)
        reranked = rankdata(r) - 1
        rank[i] = reranked
    return rank

def train_sa(exclude_lang=None):
    langs = ['ara', 'zho', 'nld', 'eng', 'fra',
             'deu', 'kor', 'rus', # no jap, no per
             'spa', 'tam', 'tur'] # no tha
    data_dir = 'datasets/sa/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    ranking_f = open(os.path.join(data_dir, 'rankings_wo_jpn_per_tha.pkl'), 'rb')
    rank = pickle.load(ranking_f)

    if exclude_lang is not None: # exclude for cross validation
        exclude_idx = langs.index(exclude_lang)
        langs.pop(exclude_idx)
        rank = rerank(rank, exclude_idx)
    else:
        exclude_lang = 'all' # for model file name

    model_save_dir = 'pretrained/SA'
    tmp_dir = 'tmp'
    preprocess = None
    prepare_train_file(datasets=datasets, langs=langs, rank=rank, tmp_dir=tmp_dir, task="SA", preprocess=preprocess)
    output_model = f"{model_save_dir}/lgbm_model_sa_{exclude_lang}.txt"
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'transfer_nr', 'transfer_vr', 'distance_n2v',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model, feature_name=feature_name, task="SA")
    assert os.path.isfile(output_model)


if __name__ == '__main__':
    # langs = ['ara', 'zho', 'nld', 'eng', 'fra',
    #          'deu', 'kor', 'rus', # no jap, no per
    #          'spa', 'tam', 'tur'] # no tha
    langs= ['ara', 'dan', 'ell', 'eng', 'tur']
    for exclude in langs:
        print(f'Start training with {exclude} excluded')
        # train_sa(exclude_lang=exclude)
        train_olid(exclude_lang=exclude)
