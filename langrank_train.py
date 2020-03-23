import pickle
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import numpy as np
from langrank import prepare_train_file, train, rank_to_relevance
from preprocessing import build_preprocess
from scipy.stats import rankdata


def rerank(rank, without_idx=None):
    reranked = []
    for r in rank:
        r.pop(without_idx)
        rr = rankdata(r) - 1
        reranked.append(rr)
    return reranked

def train_olid(exclude_lang=None, feature='base'):
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

    model_save_dir = f'pretrained/OLID/{feature}/'
    tmp_dir = "tmp"
    preprocess = build_preprocess() # preprocessing for tweeter data
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task="OLID", preprocess=preprocess, feature=feature)
    output_model = f"{model_save_dir}/lgbm_model_olid_{exclude_lang}.txt"
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    if feature == 'pos':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif feature == 'all':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    print(f'Features used are: {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model, feature_name=feature_name, task="OLID")
    assert os.path.isfile(output_model)


def train_sa(exclude_lang=None, feature='base'):
    langs = ['ara', 'deu', 'eng', 'fas', 'fra', 'hin', 'jpn', 'kor', 'nld', 'rus', 'spa', 'tam', 'tur', 'zho'] # no tha
    data_dir = 'datasets/sa/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    # ranking_f = open(os.path.join(data_dir, 'rankings/sa.pkl'), 'rb') # FIXME: temporary
    ranking_f = open('rankings/sa.pkl', 'rb') # FIXME: temporary
    rank = pickle.load(ranking_f)

    if exclude_lang is not None: # exclude for cross validation
        exclude_idx = langs.index(exclude_lang)
        langs.pop(exclude_idx)
        rank.pop(exclude_idx)
        rank = rerank(rank, exclude_idx)
    else:
        exclude_lang = 'all' # for model file name

    model_save_dir = f'pretrained/SA/{feature}/'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    tmp_dir = 'tmp'
    preprocess = None
    prepare_train_file(datasets=datasets, langs=langs, rank=rank, tmp_dir=tmp_dir, task="SA", preprocess=preprocess,
                       feature=feature)

    output_model = f"{model_save_dir}/lgbm_model_sa_{exclude_lang}.txt"

    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']

    if feature == 'pos':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif feature == 'emot':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'emotion_dist',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    # TODO: include MWE feature
    # NOTE: order of features must be consistent with the list in `distance_vec`
    elif feature == 'all':
        feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                        'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'noun_to_verb', 'pron_to_noun', 'distance_noun', 'distance_pron', 'distance_verb',
                        'emotion_dist',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']

    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model, feature_name=feature_name, task="SA")
    assert os.path.isfile(output_model)

# TODO: into shell file
if __name__ == '__main__':
    # langs= ['ara', 'dan', 'ell', 'eng', 'tur']
    langs = ['ara', 'deu', 'eng', 'fas', 'fra', 'hin', 'jpn', 'kor', 'nld', 'rus', 'spa', 'tam', 'tur', 'zho'] # no tha
    feature = 'base' # base, pos
    for exclude in langs:
        print(f'Start training with {exclude} excluded')
        print(f'Features: {feature}')
        # train_olid(exclude_lang=exclude)
        train_sa(exclude_lang=exclude, feature=feature)
