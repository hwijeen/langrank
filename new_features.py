import os
import re
import sys
import ast
from copy import copy
from collections import Counter, defaultdict
from tqdm import tqdm

from test_pos_tagger import *

lang2code = {
    'ara': 'ar', 'ces': 'cs',
    'deu': 'de', 'eng': 'en',
    'fas': 'fa', 'fra': 'fr',
    'hin': 'hi', 'jpn': 'ja',
    'kor': 'ko', 'nld': 'nl',
    'rus': 'ru', 'pol': 'pl',
    'spa': 'es', 'tam': 'ta',
    'tur': 'tr', 'zho': 'zh'
}

NOUN_TAGS = {
    'kor': ['NNG'],
    'jpn': ['名詞']
}
PRONOUN_TAGS = {
    'kor': ['NP'],
    'jpn': ['代名詞']
}
VERB_TAGS = {
    'kor': ['VV'],
    'jpn': ['動詞']
}

POS_FEATURES = ['noun', 'pron', 'verb', 'noun2verb', 'pron2noun']

def fetch_files(cond, data_dir):
    return sorted([os.path.join(data_dir, f) for f
                   in os.listdir(data_dir) if cond in f])


def read_file(fname):
    with open(fname, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    return lines


def parse_pos(line, lang):
    lst = ast.literal_eval(line)
    if lang == 'kor':
        pos = [tag[1].split('+')[0] for tag in lst]
    elif lang == 'nld':
        pos = [tag[1].split('.')[0].upper() for tag in lst]
    else:
        pos = [tag[1] for tag in lst]
    return pos

def count_pos(lines, lang):
    counts = Counter()
    for l in tqdm(lines, desc=lang):
        counts.update(parse_pos(l, lang))
    return counts


def ratio_x2y(x, y):
    n2v = x / (x + y)
    return n2v


def build_counts(data_dir):
    pos_counts = {}
    for lang, code in lang2code.items():
        fname = f'{data_dir}/{code}_pos.txt'
        print(f'Reading {fname} ...')
        lines = read_file(fname)
        counts = count_pos(lines, lang)
        pos_counts[lang] = counts
    return pos_counts


def get_pos_ratio(lang, counter, pos):
    assert pos in ['noun', 'verb', 'pron']
    num_tokens = sum(counter.values())
    if pos == 'noun':
        tag = NOUN_TAGS.get(lang, ['NOUN'])
    elif pos == 'verb':
        tag = VERB_TAGS.get(lang, ['VERB'])
    else:
        tag = PRONOUN_TAGS.get(lang, ['PRON'])
    cnt = sum([counter.get(t, 0) for t in tag]) / num_tokens
    return cnt


def get_feature(lang, counter, name):
    if name in ['noun', 'pron', 'verb']:
        return get_pos_ratio(lang, counter, name)
    elif name in ['noun2verb', 'pron2noun']:
        x, y = name.split('2')
        return ratio_x2y(get_pos_ratio(lang, counter, x),
                         get_pos_ratio(lang, counter, y))
    else:
        raise ValueError('Feature name should be noun, pron, verb, noun2verb or pron2noun.')


def build_features(data_dir, feature_dir, feature_name):
    pos_fpath = os.path.join(feature_dir, 'pos-ratio.csv')
    feature_fpath = os.path.join(feature_dir, f'{feature_name}.csv')
    if os.path.isfile(pos_fpath):
        import pandas as pd
        df = pd.read_csv(pos_fpath, index_col=0)
        feature_df = df[feature_name]
        feature_df.to_csv(feature_fpath)
        feature_dict = read_features(feature_fpath)
    elif os.path.isfile(feature_fpath):
        feature_dict = read_features(feature_fpath)
    else:
        if isinstance(feature_name, str):
            feature_name = [feature_name]
        pos_count_dict = build_counts(data_dir)
        feature_dict = {}
        print(f"Building {feature_name} features ...")
        for lang, pos_counts in pos_count_dict.items():
            features = [get_feature(lang, pos_counts, n) for n in feature_name]
            feature_dict[lang] = tuple(features)
    return feature_dict


def read_features(f):
    feature_dict = {}
    with open(f, 'r') as f:
        for line in f.readlines()[1:]:
            lang = line.split(',')[0]
            features = map(float, line.split(',')[1:])
            feature_dict[lang] = tuple(features)[0]
    return feature_dict


def write_output(feature_dict, col_name, out_file):
    if isinstance(col_name, str):
        col_name = [col_name]
    col_name = ['lang'] + col_name
    header = ','.join(col_name) + '\n'
    with open(out_file, 'w') as f:
        f.write(header)
        for lang, features in feature_dict.items():
            if isinstance(features, list):
                row = [lang] + list(map(str, features))
            else:
                row = [lang, str(features)]
            row = ','.join(row)
            print(row, file=f)
    print(f'Results saved as {out_file}')


def pos_features(lang, feature, feature_dir='./feature', data_dir='./mono'):
    assert feature in POS_FEATURES
    out_file = os.path.join(feature_dir, f'{feature}.csv')

    if not os.path.isfile(out_file):
        if 'news' in feature_dir:
            data_dir = './mono-news-processed'
        feature_dict = build_features(data_dir, feature_dir, feature)
        write_output(feature_dict, feature, out_file)
    else:
        feature_dict = read_features(out_file)
    return feature_dict[lang]

# FIXME: read only once
def emo_features(lang1, lang2, fpath='./features/', pairwise=True):
    if pairwise:
        # fpath = os.path.join(fpath, 'emo-diffs-cosine-5.txt') # old
        # fpath = os.path.join(fpath, 'emo-diffs-en-cc-cosine-5-norm.txt') # new
        # fpath = os.path.join(fpath, 'emo-diffs-en-cosine-5.txt') # en
        # fpath = os.path.join(fpath, 'emo-diffs-cc-cos-5iter-norm.txt') # en
        fpath = os.path.join(fpath, 'emo-diffs-cc-cos-5iter-zero-one-norm.txt') # en
    else:
        pass

    lang_to_code = copy(lang2code)
    lang_to_code['zho'] = 'zh'

    # code_to_lang = {v:k for k,v in lang_to_code.items()}
    # if asymmetric score is correct
    feature_dict = defaultdict(dict)
    with open(fpath) as f:
        for line in f:
            lang1_code, lang2_code, emo_score = line.split('\t')
            feature_dict[lang1_code][lang2_code] = emo_score
    return feature_dict[lang_to_code[lang1]][lang_to_code[lang2]].strip()

    # # if symmetric score is correct
    # feature_dict = dict()
    # with open(fpath) as f:
    #     for line in f:
    #         lang1_code, lang2_code, emo_score = line.split('\t')
    #         feature_dict[(lang1_code, lang2_code)] = emo_score
    # return feature_dict[(lang_to_code[lang1], lang_to_code[lang2])]

def mwe_features(lang1, lang2, fpath='./features/', norm=True):
    if norm:
        fpath = os.path.join(fpath, 'ltq_either_norm.txt')
        # fpath = os.path.join(fpath, 'ltq_either.txt')
    else:
        fpath = os.path.join(fpath, 'ltq_either_norm.txt')

    lang_to_code = copy(lang2code)
    lang_to_code['zho'] = 'zh'

    # if asymmetric score is correct
    feature_dict = defaultdict(dict)
    with open(fpath) as f:
        for line in f:
            lang1_code, lang2_code, mwe_score = line.split('\t')
            feature_dict[lang1_code][lang2_code] = mwe_score
    return feature_dict[lang_to_code[lang1]][lang_to_code[lang2]].strip()

if __name__ == "__main__":
    features = ['noun', 'pron', 'verb', 'noun2verb', 'pron2noun']
    # feature_dict = build_features('./mono-news-processed-pos', './features-news', features)
    # for f in features:
    #     feature_dict = build_features('./mono-news-processed-pos', './features-news', f)
    #     out_file = f'./features-news/{f}.csv'
    #     write_output(feature_dict, f, out_file)

    print(pos_features('zho', 'noun', './features-news'))
    print(pos_features('zho', 'verb', './features-news'))