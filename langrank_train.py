import pickle
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import numpy as np
from langrank import prepare_train_file, train, rank_to_relevance
from scipy.stats import rankdata


code_convert = { 'ara': 'ar',
                 'ces': 'cs',
                 'deu': 'de',
                 'eng': 'en',
                 'spa': 'es',
                 'fas': 'fa',
                 'fra': 'fr',
                 'hin': 'hi',
                 'jpn': 'ja',
                 'kor': 'ko',
                 'nld': 'nl',
                 'pol': 'pl',
                 'rus': 'ru',
                 'tam': 'ta',
                 'tur': 'tr',
                 'zho': 'zh'}

def rerank(rank, without_idx=None):
    reranked = []
    for r in rank:
        r.pop(without_idx)
        rr = rankdata(r, method='min') - 1
        reranked.append(rr)
    return reranked

def train_langrank(task='sa', exclude_lang=None, feature='base'):
    langs = ['ara', 'ces', 'deu', 'eng', 'fas',
             'fra', 'hin', 'jpn', 'kor', 'nld',
             'pol', 'rus', 'spa', 'tam', 'tur', 'zho'] # no tha
    data_dir = f'datasets/{task}/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]

    ranking_f = open(f'rankings/{task}.pkl', 'rb')
    rank = pickle.load(ranking_f)

    if exclude_lang is not None: # exclude for cross validation
        exclude_idx = langs.index(exclude_lang)
        langs.pop(exclude_idx)
        rank.pop(exclude_idx)
        rank = rerank(rank, exclude_idx)
        datasets.pop(exclude_idx)
    else:
        exclude_lang = 'all' # for model file name

    model_save_dir = f'pretrained/{task.upper()}/{feature}/'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    tmp_dir = 'tmp'
    preprocess = None
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task=task.upper(), preprocess=preprocess,
                       feature=feature)

    output_model = f"{model_save_dir}/lgbm_model_{task}_{exclude_lang}.txt"

    # NOTE: order of features must be consistent with the list in `distance_vec`
    if feature == 'base':
        feature_name = ['word_overlap',
                        'transfer_data_size', 'task_data_size', 'ratio_data_size',
                        'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    if feature == 'dataset':
        feature_name = ['word_overlap',
                        'transfer_data_size', 'task_data_size', 'ratio_data_size',
                        'transfer_ttr', 'task_ttr', 'distance_ttr']
    if feature == 'uriel':
        feature_name = ['genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']

    elif feature == 'pos':
        feature_name = [#'word_overlap',
                        'transfer_data_size', 'task_data_size', 'ratio_data_size',
                        # 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        # 'pron_to_noun', 'distance_pron', 'distance_verb', # 3
                        'distance_pron', 'distance_verb'] # 2
                        # 'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif feature == 'emot':
        feature_name = [#'word_overlap',
                        'transfer_data_size', 'task_data_size', 'ratio_data_size',
                        # 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'emotion_dist']
                        # 'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif feature == 'ltq':
        feature_name = [#'word_overlap',
                        'transfer_data_size', 'task_data_size', 'ratio_data_size',
                        # 'transfer_ttr', 'task_ttr', 'distance_ttr',
                        'ltq_score']
                        # 'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    elif feature == 'all':
        feature_name = ['word_overlap',
                        'transfer_data_size', 'task_data_size', 'ratio_data_size',
                        'transfer_ttr', 'task_ttr', 'distance_ttr',
                        # 'pron_to_noun', 'distance_pron', 'distance_verb', # 3
                        'distance_pron', 'distance_verb', # 2
                        'emotion_dist',
                        'ltq_score',
                        'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']

    elif feature == 'typo_group':
        feature_name = ['genetic', 'syntactic', 'featural', 'phonological', 'inventory']
    elif feature == 'geo_group':
        feature_name = ['geographical']
    elif feature == 'cult_group':
        feature_name = ['transfer_ttr', 'task_ttr', 'distance_ttr',
                        'distance_pron', 'distance_verb', 'ltq_score', 'emotion_dist']
    elif feature == 'ortho_group':
        feature_name = ['word_overlap']
    elif feature == 'data_group':
        feature_name = ['transfer_data_size', 'task_data_size', 'ratio_data_size']

    print(f'Features used are {feature_name}')
    train(tmp_dir=tmp_dir, output_model=output_model, feature_name=feature_name, task=f"{task.upper()}")
    assert os.path.isfile(output_model)

if __name__ == '__main__':
    langs = ['ara', 'ces', 'deu', 'eng', 'fas',
             'fra', 'hin', 'jpn', 'kor', 'nld',
             'pol', 'rus', 'spa', 'tam', 'tur', 'zho'] # no tha
    task = 'sa' # 'sa'
    features = ['base', 'pos', 'emot', 'ltq', 'all', 'dataset', 'uriel',]
    features += ['typo_group', 'geo_group', 'cult_group', 'ortho_group', 'data_group']
    for f in features:
        for exclude in langs:
            print(f'\nStart training with {exclude} excluded for task {task}')
            print(f'Features: {f}')
            train_langrank(task=task, exclude_lang=exclude, feature=f)
