import pickle
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import numpy as np
from langrank import prepare_train_file, train, rank_to_relevance
from preprocessing import build_preprocess
from scipy.stats import rankdata


def rerank(rank, without_idx=None):
    for i, r in enumerate(rank):
        r.pop(without_idx)
        reranked = rankdata(r) - 1
        rank[i] = reranked
    return rank

def train_olid(exclude_lang=None, model='best'):
    langs= ['ara', 'dan', 'ell', 'eng', 'tur']
    data_dir = 'datasets/olid/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    rank = [[0, 4, 2, 1, 3], # manually
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
    preprocess = build_preprocess() # preprocessing for tweeter data
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task="OLID", preprocess=preprocess, model=model)
    output_model = f"{model_save_dir}/lgbm_model_{model}_olid_{exclude_lang}.txt"
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    if model == 'pos':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif model == 'all':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model,
          feature_name=feature_name, task="OLID")
    train(tmp_dir=tmp_dir, output_model=output_model, feature_name=feature_name, task="OLID")
    assert os.path.isfile(output_model)

def train_sa(exclude_lang=None, model='best'):
    langs = ['ara', 'zho', 'nld', 'eng', 'fra',
             'deu', 'kor', 'rus', # no jap, no per
             'spa', 'tam', 'tur'] # no tha
    data_dir = 'datasets/sa/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    ranking_f = open(os.path.join(data_dir, 'rankings_wo_jpn_per_tha.pkl'), 'rb') # FIXME: temporary
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
    prepare_train_file(datasets=datasets, langs=langs, rank=rank, tmp_dir=tmp_dir, task="SA", preprocess=preprocess,
                       model=model)

    output_model = f"{model_save_dir}/lgbm_model_sa_{exclude_lang}.txt"

    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']

    if model == 'pos':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif model == 'all':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']

    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model,
          feature_name=feature_name, task="SA")
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
