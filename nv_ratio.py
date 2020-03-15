import os
import re
import sys
from collections import Counter

from test_pos_tagger import *

ud_resources={'ara': 'Arabic-PADT',
              'chi': 'Chinese-CFL', # GSD, HK
              'dut': 'Dutch-Alpino',
              'eng': 'English-EWT', # GUM, LinES, ParTUT
              'fre': 'French-FQB', # GSD, ParTUT, Sequoia, Spoken
              'ger': 'German-GSD', #HDT, LIT
              'jap': 'Japanese-GSD', #Modern
              'kor': 'Korean-Kaist', #Kaist
              'per': 'Persian-Seraji',
              'rus': 'Russian-GSD', # SynTagRus, Taiga
              'spa': 'Spanish-AnCora', #GSD
              'tam': 'Tamil-TTB',
              'tur': 'Turkish-GB'} # IMST
etc_resources = {'tha': ['Models/POS/Thai.RDR', 'Models/POS/Thai.DICT']}

pos_tagger_dir = 'RDRPOSTagger'
resources_by_lang = get_resources_path(pos_tagger_dir, ud_resources, etc_resources)
taggers_by_lang = load_pos_taggers(pos_tagger_dir, resources_by_lang)

results_file = './features/nv_ratio.txt'


def read_features(f):
    feature_dict = {}
    with open(f, 'r') as f:
        for line in f.readlines()[1:]:
            lang, noun, verb, n2v = line.split('\t')
            feature_dict[lang] = (float(noun), float(verb), float(n2v))
    return feature_dict


def findTags(taggedSample):
    tags = re.findall('/[A-Z]+', taggedSample)
    return tags


def noun2verb(noun_cnt, verb_cnt):
    n2v = noun_cnt / (noun_cnt + verb_cnt)
    return n2v


def count_pos(pos_counter, pos='noun'):
    if pos == 'noun':
        tags = ['/NOUN', '/NCMN', '/NTTL', '/NLBL']
    elif pos == 'verb':
        tags = ['/VERB', '/VACT', '/VSTA', '/VATT']
    cnt = sum([pos_counter.get(t, 0) for t in tags])
    return cnt


def build_counter(lang, dataset_source):

    if isinstance(dataset_source, str):
        with open(dataset_source) as inp:
            source_lines = inp.readlines()[1:]
    elif isinstance(dataset_source, list):
        source_lines = dataset_source
    else:
        raise Exception("dataset_source should either be a filnename (str) or a list of sentences.")

    pos_counter = Counter()
    tagger = taggers_by_lang[lang]

    for l in source_lines:
        try:
            sample = l.strip().split('\t')[1]
            result = tagger(rawLine=sample)
            pos_counter.update(findTags(result))
        except:
            pass
    return pos_counter


def build_features(data_dir='../data'):
    feature_dict = {}

    languages = list(ud_resources.keys()) + list(etc_resources.keys())
    languages = sorted(languages)

    for lang in languages:
        fname = [f for f in os.listdir(f'{data_dir}/{lang}') if f.endswith('reviews.tsv') or 'amazon' in f][0]
        fname = os.path.join(f'{data_dir}/{lang}', fname)
        pos_counter = build_counter(lang, fname)

        num_tokens = sum(pos_counter.values())
        noun_cnt = count_pos(pos_counter, 'noun')
        verb_cnt = count_pos(pos_counter, 'verb')

        n2v_ratio = noun2verb(noun_cnt, verb_cnt)
        n_ratio = noun_cnt / num_tokens
        v_ratio = verb_cnt / num_tokens
        feature_dict[lang] = (n_ratio, v_ratio, n2v_ratio)

        print(f"*** {lang} {fname} ***")
        print(f"Noun {n_ratio} | Verb {v_ratio} | Noun-to-verb {n2v_ratio}")
        print("-" * 50)
    write_output(feature_dict)
    return feature_dict


def write_output(feature_dict):
    with open(results_file, 'w') as f:
        f.write('lang\tnoun\tverb\tn2v\n')
        for lang, features in feature_dict.items():
            n, v, n2v = tuple(features)
            print(f'{lang}\t{n}\t{v}\t{n2v}', file=f)
    print(f'Results saved as {results_file}')


def nv_features(lang):
    if os.path.isfile(results_file):
        feature_dict = read_features(results_file)
        noun, verb, n2v = tuple(feature_dict[lang])
        return noun, verb, n2v
    feature_dict = build_features('../lang-selection/data')
    return feature_dict[lang]


if __name__ == "__main__":
    build_features(data_dir='../lang-selection/data')