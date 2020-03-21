import os
import sys
from functools import partial

def listdir(dir_):
    return [os.path.join(dir_, f) for f in os.listdir(dir_)]

def find_rdr_dict(resource_path, kind='UPOS'):
    files = listdir(resource_path)
    rdr_path = [f for f in files if kind in f and f.endswith('RDR')][0]
    dict_path = [f for f in files if kind in f and f.endswith('DICT')][0]
    return rdr_path, dict_path

def get_resources_path(pos_tagger_dir, ud_resources, etc_resources):
    resources_by_lang = {}
    for lang, resource_id in ud_resources.items():
        resource_dir = f'Models/ud-treebanks-v2.4/UD_{resource_id}'
        resource_path = os.path.join(pos_tagger_dir, resource_dir)
        rdr_path, dict_path = find_rdr_dict(resource_path)
        resources_by_lang[lang] = [rdr_path, dict_path]
    for lang, resource_path in etc_resources.items():
        resource_path = [os.path.join(pos_tagger_dir, p) for p in resource_path]
        resources_by_lang[lang] = resource_path
    return resources_by_lang

def get_sample(fpath, num=5):
    samples = []
    with open(fpath, 'r') as f:
        f.readline() # remove header
        for _ in range(num):
            sample = f.readline().strip('\n').split('\t')[1]
            samples.append(sample)
        return samples

def load_sample_data(base_path, langs, num=5):
    samples_by_lang = {}
    for lang in langs:
        lang_dir = os.path.join(base_path, lang)
        print(lang_dir)
        fpath = [f for f in listdir(lang_dir) if f.endswith('.tsv')][0] # arbitrary data
        samples = get_sample(fpath, num)
        samples_by_lang[lang] = samples
    return samples_by_lang

def load_pos_taggers(pos_tagger_dir, resources_by_lang):
    py_tagger_path = os.path.join(pos_tagger_dir, 'pSCRDRtagger')
    os.chdir(py_tagger_path)
    import sys; sys.path.append('.')
    from RDRPOSTagger import RDRPOSTagger, readDictionary
    os.chdir('../../')
    taggers_by_lang = {}
    for lang, resources in resources_by_lang.items():
        tagger = RDRPOSTagger()
        rdr_path, dict_path = resources
        tagger.constructSCRDRtreeFromRDRfile(rdr_path)
        dict_ = readDictionary(dict_path)
        taggers_by_lang[lang] = partial(tagger.tagRawSentence, DICT=dict_)
    return taggers_by_lang

if __name__ == '__main__':
    ud_resources={'ara': 'Arabic-PADT',
                  'zho': 'Chinese-CFL', # GSD, HK
                  'eng': 'English-EWT', # GUM, LinES, ParTUT
                  'fra': 'French-FQB', # GSD, ParTUT, Sequoia, Spoken
                  'deu': 'German-GSD', #HDT, LIT
                  'jpn': 'Japanese-GSD', #Modern
                  'kor': 'Korean-Kaist', #Kaist
                  'fas': 'Persian-Seraji',
                  'rus': 'Russian-GSD', # SynTagRus, Taiga
                  'spa': 'Spanish-AnCora', #GSD
                  'tam': 'Tamil-TTB',
                  'tur': 'Turkish-GB'} # IMST
    etc_resources = {'tha': ['Models/POS/Thai.RDR', 'Models/POS/Thai.DICT'],
                     'nld': ['Models/MORPH/Dutch.RDR', 'Models/MORPH/Dutch.DICT'] }

    pos_tagger_dir = 'RDRPOSTagger'
    resources_by_lang = get_resources_path(pos_tagger_dir, ud_resources, etc_resources)

    data_dir = 'datasets/sa'
    sample_size = 3
    langs = [l for l in ud_resources.keys()] + [l for l in etc_resources.keys()]
    samples_by_lang = load_sample_data(data_dir, langs, sample_size)

    taggers_by_lang = load_pos_taggers(pos_tagger_dir, resources_by_lang)

    # PRINT SAMPLE RESULT!!!
    # test_lang = 'tha'
    for test_lang in list(ud_resources.keys()) + list(etc_resources.keys()):
        for lang, samples in  samples_by_lang.items():
            tagger = taggers_by_lang[lang]
            if lang == test_lang:
                for sample in samples:
                    print(tagger(rawLine=sample))



