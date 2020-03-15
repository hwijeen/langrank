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

def readFeatures(f):
    feature_dict = {}
    with open(f, 'r') as f:
        for line in f.readlines()[1:]:
            lang, noun, verb = line.split('\t')
            feature_dict[lang] = [float(noun), float(verb)]
    return feature_dict


def findTags(taggedSample):
    tags = re.findall('/[A-Z]+', taggedSample)
    return tags


def NounVerbRatio(lang, dataset_source=None):
    if os.path.isfile(results_file):
        feature_dict = readFeatures(results_file)
        noun, verb = feature_dict[lang][0], feature_dict[lang][1]
        return noun, verb

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

    num_tokens = sum(pos_counter.values())

    if lang == 'tha':
        noun_tags = ['/NCMN', '/NTTL', '/NLBL']
        verb_tags = ['/VACT', '/VSTA', '/VATT']
        noun_ratio = sum([pos_counter.get(t, 0) for t in noun_tags]) / num_tokens
        verb_ratio = sum([pos_counter.get(t, 0) for t in verb_tags]) / num_tokens        
    else:
        noun_ratio = pos_counter.get('/NOUN', 0) / num_tokens
        verb_ratio = pos_counter.get('/VERB', 0) / num_tokens

    return noun_ratio, verb_ratio

if __name__ == "__main__":
    languages = list(ud_resources.keys()) + list(etc_resources.keys())
    languages = sorted(languages)
    noun_ratio = []
    verb_ratio = []

    for lang in languages:
        fname = [f for f in os.listdir(f'../data/{lang}') if f.endswith('reviews.tsv') or 'amazon' in f][0]
        fname = os.path.join(f'../data/{lang}', fname)
        n, v = NounVerbRatio(lang, fname)
        noun_ratio.append(n)
        verb_ratio.append(v)

        print(f"*** {lang} {fname} ***")
        print(f"Noun {n} | Verb {v}")
        # if counts is not None:
        #     print(counts)
        print("-"*50)

    with open(results_file, 'w') as f:
        f.write('lang\tnoun\tverb\n')
        for l, n, v in zip(languages, noun_ratio, verb_ratio):
            print(f'{l}\t{n}\t{v}', file=f)
    
    print(f'Results saved as {results_file}')