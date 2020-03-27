import os
import re
import sys
import ast
from copy import copy
from collections import Counter, defaultdict

from test_pos_tagger import *

lang2code = {
    'ara': 'ar', 'ces': 'cs',
    'deu': 'de', 'eng': 'en',
    'fas': 'fa', 'fra': 'fr',
    'hin': 'hi', 'jpn': 'ja',
    'kor': 'ko', 'nld': 'nl',
    'rus': 'ru', 'pol': 'pl',
    'spa': 'es', 'tam': 'ta',
    'tur': 'tr', 'zho': 'zh-Hans'
}

NOUN_TAGS = {
    'kor': ['NNG'],
    'jpn': ['名詞'],
    # 'nld': ['noun.N(soort,ev,basis,zijd,stan)',
    #         'noun.N(soort,ev,basis,onz,stan)',
    #         'noun.N(soort,mv,basis)'],
    # 'zho': ['n', 'an', 'f', 's']
}
PRONOUN_TAGS = {
    'kor': ['NP'],
    'jpn': ['代名詞'],
    # 'nld': ['pron.VNW(pers,pron,nomin,red,1,mv)',
    #         'pron.VNW(pers,pron,nomin,vol,1,ev)'],
    # 'zho': ['rz', 'rr', 'r']
}
VERB_TAGS = {
    'kor': ['VV'],
    'jpn': ['動詞'],
    # 'nld': ['verb.WW(inf,vrij,zonder)',
    #         'verb.WW(pv,tgw,mv)'],
    # 'zho': ['v', 'vn']
}


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
    for l in lines:
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


def build_features(data_dir, feature_name):
    if os.path.isfile('./features/pos-ratio.csv'):
        import pandas as pd
        df = pd.read_csv('./features/pos-ratio.csv', index_col=0)
        feature_df = df[[feature_name]]
        feature_df.to_csv(f'./features/{feature_name}.csv')
        feature_dict = read_features(f'./features/{feature_name}.csv')
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
            row = [lang] + list(map(str, features))
            row = ','.join(row)
            print(row, file=f)
    print(f'Results saved as {out_file}')


def pos_features(lang, feature, data_dir='./mono'):
    col_name = ['noun', 'pron', 'verb', 'noun2verb', 'pron2noun']
    assert feature in col_name

    out_file = f'./features/{feature}.csv'

    if not os.path.isfile(out_file):
        feature_dict = build_features(data_dir, feature)
        write_output(feature_dict, col_name, out_file)
    else:
        feature_dict = read_features(out_file)
    return feature_dict[lang]

# FIXME: read only once
def emo_features(lang1, lang2, fpath='features/', pairwise=True):
    if pairwise:
        fpath = os.path.join(fpath, 'emo-diffs-cosine-5.txt')
    else:
        fpath = os.path.join(fpath, 'emo-diffs-en-cosine-5.txt')

    lang_to_code = copy(lang2code)
    lang_to_code['zho'] = 'zh'

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







if __name__ == "__main__":
    features = ['noun', 'pron', 'verb', 'noun2verb', 'pron2noun']
    # feature_dict = build_features('./mono', features)
    for f in features:
        feature_dict = build_features('./mono', f)
        out_file = f'./features/{f}.csv'
        write_output(feature_dict, f, out_file)