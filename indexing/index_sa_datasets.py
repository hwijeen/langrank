import os
import subprocess
import numpy as np
from collections import Counter

import sys; sys.path.append('/home/hwijeen/langrank')
from new_features import pos_features

def read_data(fn):
    with open(fn) as inp:
        lines = inp.readlines()
    c = []
    v = []
    for l in lines:
        l = l.strip().split()
        if len(l) == 2:
            c.append(int(l[1]))
            v.append(l[0])
    return v,c


dataset_dir = "datasets/sa"

# Not needed now
#eng_vocab_f = "datasets/eng/word.vocab"
#en_v, en_c = read_data(eng_vocab_f)
#eng_vocab_f = "datasets/eng/subword.vocab"
#en_sub_v, en_sub_c = read_data(eng_vocab_f)

#w2i = {w:i for i,w in enumerate(en_v)}
#subw2i = {w:i for i,w in enumerate(en_sub_v)}

def get_vocab(filename):
    with open(filename) as inp:
        lines = inp.readlines()
    all_words = [w for l in lines for w in l.strip().split()]
    return all_words



features = {}
# Add if adding target-side features for MT
#features["eng"] = {}
#features["eng"]["word_vocab"] = en_v
#features["eng"]["subword_vocab"] = en_sub_v

exclude_langs = {'tha'}
for filename in os.listdir(dataset_dir):
    if filename.endswith('.pkl'): continue
    #print(filename)
    temp = filename.split(".")
    language = temp[0]
    if language in exclude_langs: 
        print(f'Skipping language {language}')
        continue

    # Get number of lines in training data
    #filename = "ted-train.orig."+temp[0]
    if len(language) == 2:
        language = LETTER_CODES[language]
    filename = os.path.join(dataset_dir,filename)
    bashCommand = "wc -l " + filename
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    lines = int(output.strip().split()[0])
    print(filename + " " + str(lines))

    all_words = get_vocab(filename)
    c = Counter(all_words)
    key = filename.split('-')[0]
    features[key] = {}
    features[key]["lang"] = language
    features[key]["dataset_size"] = lines

    unique = list(c)
    # Get number of types and tokens
    features[key]["token_number"] = len(all_words)
    features[key]["type_number"] = len(unique)
    features[key]["word_vocab"] = unique
    features[key]["type_token_ratio"] = features[key]["type_number"]/float(features[key]["token_number"])

    features[key]["noun_ratio"] = pos_features(language, 'noun')
    features[key]["verb_ratio"] = pos_features(language, 'verb')
    features[key]["pron_ratio"] = pos_features(language, 'pron')
    features[key]["n2v_ratio"] = pos_features(language, 'noun2verb')
    features[key]["p2n_ratio"] = pos_features(language, 'pron2noun')

indexed = "indexed/SA"
outputfile = os.path.join(indexed, "sa.npy")
import ipdb; ipdb.set_trace(context=5)
np.save(outputfile, features)







