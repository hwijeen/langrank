import lang2vec.lang2vec as l2v
import numpy as np
import pkg_resources
import os
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from new_features import pos_features, emo_features, mwe_features

TASKS = ["MT", "DEP", "EL", "POS", "OLID", "SA"]


MT_DATASETS = {
    "ted" : "ted.npy",
}
POS_DATASETS = {
    "ud" : "ud.npy"
}
EL_DATASETS = {
    "wiki" : "wiki.npy"
}
DEP_DATASETS = {
    "conll" : "conll.npy"
}
OLID_DATASETS = {
    "olid" : "olid.npy"
}
SA_DATASETS = {
    "sa": "sa.npy"
}

MT_MODELS = {
    "all" : "all.lgbm",
    "geo" : "geo.lgbm",
    "aze" : "lgbm_model_mt_aze.txt",
    "fin" : "lgbm_model_mt_fin.txt",
    "ben" : "lgbm_model_mt_ben.txt",
    "best": "lgbm_model_mt_all.txt",
}
POS_MODELS = {
        "best": "lgbm_model_pos_all.txt"
}
EL_MODELS = {}
DEP_MODELS = {}

OLID_MODELS = {
    "best": "lgbm_model_olid_all.txt",
    "ara":"lgbm_model_olid_ara.txt",
    "dan":"lgbm_model_olid_dan.txt",
    "ell":"lgbm_model_olid_ell.txt",
    "eng":"lgbm_model_olid_eng.txt",
    "tur":"lgbm_model_olid_tur.txt",
}
SA_MODELS = {
    "all":"lgbm_model_sa_all.txt",
    "ara":"lgbm_model_sa_ara.txt",
    "ces":"lgbm_model_sa_ces.txt",
    "deu":"lgbm_model_sa_deu.txt",
    "eng":"lgbm_model_sa_eng.txt",
    "fas":"lgbm_model_sa_fas.txt",
    "fra":"lgbm_model_sa_fra.txt",
    "hin":"lgbm_model_sa_hin.txt",
    "jpn":"lgbm_model_sa_jpn.txt",
    "kor":"lgbm_model_sa_kor.txt",
    "nld":"lgbm_model_sa_nld.txt",
    "pol":"lgbm_model_sa_pol.txt",
    "rus":"lgbm_model_sa_rus.txt",
    "spa":"lgbm_model_sa_spa.txt",
    "tam":"lgbm_model_sa_tam.txt",
    "tur":"lgbm_model_sa_tur.txt",
    "zho":"lgbm_model_sa_zho.txt"
}

# checks
def check_task(task):
    if task not in TASKS:
        raise Exception("Unknown task " + task + ". Only 'MT', 'DEP', 'EL', 'POS', 'OLID', 'SA' are supported.")

def check_task_model(task, model):
    check_task(task)
    avail_models = map_task_to_models(task)
    if model not in avail_models:
        ll = ', '.join([key for key in avail_models])
        raise Exception("Unknown model " + model + ". Only "+ll+" are provided.")

def check_task_model_data(task, model, data):
    check_task_model(task, model)
    avail_data = map_task_to_data(task)
    if data not in avail_data:
        ll = ', '.join([key for key in avail_data])
        raise Exception("Unknown dataset " + data + ". Only "+ll+" are provided.")


# utils
def map_task_to_data(task):
    if task == "MT":
        return MT_DATASETS
    elif task == "POS":
        return POS_DATASETS
    elif task == "EL":
        return EL_DATASETS
    elif task == "DEP":
        return DEP_DATASETS
    elif task == "OLID":
        return OLID_DATASETS
    elif task == "SA":
        return SA_DATASETS
    else:
        raise Exception("Unknown task")

def map_task_to_models(task):
    if task == "MT":
        return MT_MODELS
    elif task == "POS":
        return POS_MODELS
    elif task == "EL":
        return EL_MODELS
    elif task == "DEP":
        return DEP_MODELS
    elif task == "OLID":
        return OLID_MODELS
    elif task == "SA":
        return SA_MODELS
    else:
        raise Exception("Unknown task")

def read_vocab_file(fn):
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


# used for ranking
def get_candidates(task, languages=None):
    if languages is not None and not isinstance(languages, list):
        raise Exception("languages should be a list of ISO-3 codes")

    datasets_dict = map_task_to_data(task)
    cands = []
    for dt in datasets_dict:
        fn = pkg_resources.resource_filename(__name__, os.path.join('indexed', task, datasets_dict[dt]))
        d = np.load(fn, encoding='latin1', allow_pickle=True).item()
        # languages with * means to exclude
        cands += [(key,d[key]) for key in d if '*' + key.split('/')[2][:3] not in languages]
    cand_langs = [i[0] for i in cands]
    print(f"Candidate languages are: {cand_langs}")

    # Possibly restrict to a subset of candidate languages
    # FIXME: why limits to 2?
    # cands = cands[:2]

    if languages is not None and task == "MT":
        add_languages = []
        sub_languages = []
        for l in languages:
            if len(l) == 3:
                add_languages.append(l)
            else:
                # starts with -
                sub_languages.append(l[1:])
        # Keep a candidate if it matches the languages
        # -aze indicates all except aze
        if len(add_languages) > 0:
            new_cands = [c for c in cands if c[0][-3:] in add_languages and c[0][-3] not in sub_languages]
        else:
            new_cands = [c for c in cands if c[0][-3:] not in sub_languages]
        return new_cands

    return cands

# extract dataset dependent feautures
def prepare_new_dataset(lang, task="MT", dataset_source=None,
                        dataset_target=None, dataset_subword_source=None,
                        dataset_subword_target=None):
    features = {}
    features["lang"] = lang

    # Get dataset features
    if dataset_source is None and dataset_target is None and dataset_subword_source is None and dataset_subword_target is None:
        # print("NOTE: no dataset provided. You can still use the ranker using language typological features.")
        return features
    elif dataset_source is None: # and dataset_target is None:
        # print("NOTE: no word-level dataset provided, will only extract subword-level features.")
        pass
    elif dataset_subword_source is None: # and dataset_subword_target is None:
        # print("NOTE: no subword-level dataset provided, will only extract word-level features.")
        pass


    source_lines = []
    if isinstance(dataset_source, str):
        with open(dataset_source) as inp:
            source_lines = inp.readlines()
    elif isinstance(dataset_source, list):
        source_lines = dataset_source
    else:
        raise Exception("dataset_source should either be a filnename (str) or a list of sentences.")
    '''
    if isinstance(dataset_target, str):
        with open(dataset_target) as inp:
            target_lines = inp.readlines()
    elif isinstance(dataset_target, list):
        source_lines = dataset_target
    else:
        raise Exception("dataset_target should either be a filnename (str) or a list of sentences.")
    '''
    if source_lines:
        if task != "EL":
            features["dataset_size"] = len(source_lines)
            tokens = [w for s in source_lines for w in s.strip().split()]
            features["token_number"] = len(tokens)
            types = set(tokens)
            features["type_number"] = len(types)
            features["word_vocab"] = types
            features["type_token_ratio"] = features["type_number"]/float(features["token_number"])
        elif task == "EL":
            features["dataset_size"] = len(source_lines)
            tokens = [w for s in source_lines for w in s.strip().split()]
            types = set(tokens)
            features["word_vocab"] = types

    if task in ["SA", "OLID", "DEP", "POS"]:
        # read all pos features even if we might not use everything
        # features["noun_ratio"] = pos_features(lang, 'noun')
        features["verb_ratio"] = pos_features(lang, 'verb')
        features["pron_ratio"] = pos_features(lang, 'pron')
        # features["n2v_ratio"] = pos_features(lang, 'noun2verb')
        features["p2n_ratio"] = pos_features(lang, 'pron2noun')

    if task == "MT":
        # Only use subword overlap features for the MT task
        if isinstance(dataset_subword_source, str):
            with open(dataset_subword_source) as inp:
                source_lines = inp.readlines()
        elif isinstance(dataset_subword_source, list):
            source_lines = dataset_subword_source
        elif dataset_subword_source is None:
            pass
            # Use the word-level info, just in case. TODO(this is only for MT)
            # source_lines = []
        else:
            raise Exception("dataset_subword_source should either be a filnename (str) or a list of sentences.")
        if source_lines:
            features["dataset_size"] = len(source_lines) # This should be be the same as above
            tokens = [w for s in source_lines for w in s.strip().split()]
            features["subword_token_number"] = len(tokens)
            types = set(tokens)
            features["subword_type_number"] = len(types)
            features["subword_vocab"] = types
            features["subword_type_token_ratio"] = features["subword_type_number"]/float(features["subword_token_number"])

    return features

def uriel_distance_vec(languages):
    # print('...geographic')
    geographic = l2v.geographic_distance(languages)
    # print('...genetic')
    genetic = l2v.genetic_distance(languages)
    # print('...inventory')
    inventory = l2v.inventory_distance(languages)
    # print('...syntactic')
    syntactic = l2v.syntactic_distance(languages)
    # print('...phonological')
    phonological = l2v.phonological_distance(languages)
    # print('...featural')
    featural = l2v.featural_distance(languages)
    uriel_features = [genetic, syntactic, featural, phonological, inventory, geographic]
    return uriel_features


def distance_vec(test, transfer, uriel_features, task, feature):
    output = []
    # Dataset specific
    # Dataset Size
    transfer_dataset_size = transfer["dataset_size"]
    task_data_size = test["dataset_size"]
    ratio_dataset_size = float(transfer_dataset_size)/task_data_size
    # TTR
    if task != "EL":
        transfer_ttr = transfer["type_token_ratio"]
        task_ttr = test["type_token_ratio"]
        distance_ttr = (1 - transfer_ttr/task_ttr) ** 2

    # Word overlap
    if task != "EL":
        word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"])))) / (transfer["type_number"] + test["type_number"])
    elif task == "EL":
        word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"]))))
    # Subword overlap
    if task == "MT":
        subword_overlap = float(len(set(transfer["subword_vocab"]).intersection(set(test["subword_vocab"])))) / (transfer["subword_type_number"] + test["subword_type_number"])

    # POS related features
    if task == "SA" or task == "OLID":
        # transfer_nr = transfer["noun_ratio"]
        transfer_vr = transfer["verb_ratio"]
        transfer_pr = transfer["pron_ratio"]
        # transfer_n2v = transfer["n2v_ratio"]
        transfer_p2n = transfer["p2n_ratio"]

        # task_nr = test["noun_ratio"]
        task_vr = test["verb_ratio"]
        task_pr = test["pron_ratio"]
        # task_n2v = test["n2v_ratio"]
        task_p2n = test["p2n_ratio"]


        # distance_noun = (1 - transfer_nr / task_nr) ** 2
        distance_verb = (1 - transfer_vr / task_vr) ** 2
        distance_pron = (1 - transfer_pr / task_pr) ** 2
        # distance_n2v = (1 - transfer_n2v / task_n2v) ** 2
        distance_p2n = (1 - transfer_p2n / task_p2n) ** 2

        emotion_dist = emo_features(test['lang'], transfer['lang'])
        mwe_dist = mwe_features(test['lang'], transfer['lang'])
        # mwe_dist = mwe_features(test['lang'], transfer['lang'], norm=False)

    # # TODO: var name - this is no longer data_specific_features
    genetic, syntactic, featural, phonological, inventory, geographic = tuple(uriel_features)
    typo_features = [genetic, syntactic, featural, phonological, inventory]
    geo_features = [geographic]
    ortho_features = [word_overlap]
    data_features = [transfer_dataset_size, task_data_size, ratio_dataset_size]
    ttr_features = [transfer_ttr, task_ttr, distance_ttr]

    if task == "MT":
        # data_specific_features = [word_overlap, subword_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
        ortho_features = [word_overlap, subword_overlap]
        data_specific_features = ortho_features + data_features + ttr_features
    elif task == "POS" or task == "DEP":
        # data_specific_features = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
        data_specific_features = ortho_features + data_features + ttr_features
    elif task == "EL":
        # data_specific_features = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size]
        data_specific_features = ortho_features + data_features

    elif task == "OLID" or task == "SA":
        data_specific_features = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size,
                                  transfer_ttr, task_ttr, distance_ttr]
        if feature == 'base':
            feats = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size,
                                      transfer_ttr, task_ttr, distance_ttr]
            feats += uriel_features
            return np.array(feats)
        elif feature == 'nogeo':
            feats = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size,
                                      transfer_ttr, task_ttr, distance_ttr]
            feats += uriel_features[:-1]
            return np.array(feats)
        elif feature == 'pos': # TODO: if this is not good enough, using raw ratio as well as distances
            feats = [word_overlap, transfer_dataset_size, task_data_size,
                     ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
            data_specific_features += [distance_p2n, distance_pron, distance_verb] # 3
            # feats += [distance_pron, distance_verb] # 2
            feats += uriel_features
            return np.array(feats)
        elif feature == 'emot':
            feats = [word_overlap, transfer_dataset_size, task_data_size,
                     ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
            feats += [emotion_dist]
            feats += uriel_features
            return np.array(feats)
        elif feature == 'mwe':
            feats = [word_overlap, transfer_dataset_size, task_data_size,
                     ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
            feats += [mwe_dist]
            feats += uriel_features
            return np.array(feats)
        elif feature == 'syn_only':
            feats = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size]
            feats += uriel_features[:-1]
            return np.array(feats)
        elif feature == 'cult_only':
            feats = [transfer_dataset_size, task_data_size, ratio_dataset_size]
            feats += [transfer_ttr, task_ttr, distance_ttr]
            feats += [distance_pron, distance_verb]
            feats += [emotion_dist, mwe_dist]
            feats += [uriel_features[-1]] # geo
            return np.array(feats)
        elif feature == 'all':
            raise Exception('Not implemented')
            # data_specific_features += [distance_n2v, distance_p2n, distance_noun, distance_pron, distance_verb]
            # data_specific_features += [distance_p2n, distance_pron, distance_verb]
            # data_specific_features += [distance_pron, distance_verb]
            # data_specific_features += [emotion_dist]
            # data_specific_features += [mwe_dist]

    return np.array(data_specific_features + uriel_features)

def lgbm_rel_exp(BLEU_level, cutoff):
    if isinstance(BLEU_level, np.ndarray):
        return np.where(BLEU_level >= cutoff, BLEU_level - cutoff + 1, 0)
    else:
        return BLEU_level - cutoff + 1 if BLEU_level >= cutoff else 0

# TODO: cutoff when data is big!
def rank_to_relevance(rank, num_lang): # so that lower ranks are given higher relevance
    if isinstance(rank, list):
        rank = np.array(rank)
    return np.where(rank != 0, -rank + num_lang, 0)

# preparing the file for training
def prepare_train_file(datasets, langs, rank, segmented_datasets=None,
                       task="MT", tmp_dir="tmp", preprocess=None, feature='all'):
    """
    dataset: [ted_aze, ted_tur, ted_ben]
    lang: [aze, tur, ben]
    rank: [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
    """
    num_langs = len(langs)
    # REL_EXP_CUTOFF = num_langs - 1 - 9
    # REL_EXP_CUTOFF = num_langs - 3 # no cutoff

    if not isinstance(rank, np.ndarray):
        rank = np.array(rank)
    # BLEU_level = -rank + len(langs)
    # rel_BLEU_level = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)
    relevance = rank_to_relevance(rank, len(langs))

    features = {}
    for i, (ds, lang) in enumerate(zip(datasets, langs)):
        with open(ds, "r") as ds_f:
            if preprocess is not None:
                lines = [preprocess(l.strip()) for l in ds_f.readlines()]
            else:
                lines = ds_f.readlines()
        seg_lines = None
        if segmented_datasets is not None:
            sds = segmented_datasets[i]
            with open(sds, "r") as sds_f:
                seg_lines = sds_f.readlines()
        features[lang] = prepare_new_dataset(lang=lang, task=task, dataset_source=lines,
                                             dataset_subword_source=seg_lines)
    uriel = uriel_distance_vec(langs)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    train_file = os.path.join(tmp_dir, f"train_{task}.csv")
    train_file_f = open(train_file, "w")
    train_size = os.path.join(tmp_dir, f"train_{task}_size.csv")
    train_size_f = open(train_size, "w")

    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if i != j:
                uriel_features = [u[i, j] for u in uriel]
                distance_vector = distance_vec(features[lang1], features[lang2], uriel_features, task, feature)
                distance_vector = ["{}:{}".format(i, d) for i, d in enumerate(distance_vector)]
                # line = " ".join([str(rel_BLEU_level[i, j])] + distance_vector) # svmlight format
                line = " ".join([str(relevance[i, j])] + distance_vector) # svmlight format
                train_file_f.write(line + "\n")
        train_size_f.write("{}\n".format(num_langs-1))
    train_file_f.close()
    train_size_f.close()
    # print("Dump train file to {} ...".format(train_file_f))
    # print("Dump train size file to {} ...".format(train_size_f))

def train(tmp_dir, output_model, feature_name='auto', task='OLID'):
    train_file = os.path.join(tmp_dir, f"train_{task}.csv")
    train_size = os.path.join(tmp_dir, f"train_{task}_size.csv")
    X_train, y_train = load_svmlight_file(train_file)
    print('Training in prog...')
    model = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=16,
                        max_depth=-1, learning_rate=0.1, n_estimators=100,
                        min_child_samples=5)
    model.fit(X_train, y_train, group=np.loadtxt(train_size),
              feature_name=feature_name)
    model.booster_.save_model(output_model)
    print(f'Model saved at {output_model}')

def rank(test_dataset_features, task="MT", candidates="all", model="best", feature='base', print_topK=3):
    '''
    est_dataset_features : the output of prepare_new_dataset(). Basically a dictionary with the necessary dataset features.
    '''
    # Checks
    check_task_model(task, model)

    # Get candidates to be compared against
    # print("Preparing candidate list...")
    if candidates == 'all':
        candidate_list = get_candidates(task)
    else:
        # Restricts to a specific set of languages
        candidate_list = get_candidates(task, candidates)

    # print("Collecting URIEL distance vectors...")

    languages = [test_dataset_features["lang"]] + [c[1]["lang"] for c in candidate_list]
    # TODO: This takes forever...
    uriel = uriel_distance_vec(languages)


    # print("Collecting dataset distance vectors...")
    test_inputs = []
    for i,c in enumerate(candidate_list):
        key = c[0]
        cand_dict = c[1]
        candidate_language = key[-3:]
        uriel_j = [u[0,i+1] for u in uriel]
        distance_vector = distance_vec(test_dataset_features, cand_dict, uriel_j, task,  feature)
        test_inputs.append(distance_vector)

    # load model
    model_dict = map_task_to_models(task) # this loads the dict that will give us the name of the pretrained model
    model_fname = model_dict[model] # this gives us the filename (needs to be joined, see below)
    # modelfilename = pkg_resources.resource_filename(__name__, os.path.join('pretrained', task, model_fname))
    modelfilename = pkg_resources.resource_filename(__name__, os.path.join('pretrained', task, feature, model_fname))
    print(f"Loading model... {modelfilename}")

    # rank
    bst = lgb.Booster(model_file=modelfilename)

    # print("predicting...")
    predict_contribs = bst.predict(test_inputs, pred_contrib=True)
    predict_scores = predict_contribs.sum(-1)


    # # 0 means we ignore this feature (don't compute single-feature result of it)
    # if task == "MT":
    #     sort_sign_list = [-1, -1, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    #     feature_name = ["Overlap word-level", "Overlap subword-level", "Transfer lang dataset size",
    #                 "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR",
    #                 "Target lang TTR", "Transfer target TTR distance", "GENETIC",
    #                 "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    # elif task == "POS" or task == "DEP":
    #     sort_sign_list = [-1, 0, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 1]
    #     feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                     "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                     "Transfer target TTR distance", "GENETIC", "SYNTACTIC", "FEATURAL",
    #                     "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    # elif task == "EL":
    #     sort_sign_list = [-1, -1, 0, -1, 1, 1, 1, 1, 1, 1]
    #     feature_name = ["Entity overlap", "Transfer lang dataset size", "Target lang dataset size",
    #                     "Transfer over target size ratio", "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL",
    #                     "INVENTORY", "GEOGRAPHIC"]
    # # TODO: emotion distance?
    # elif task == "OLID" or task == "SA":
    #     if feature == 'base':
    #         sort_sign_list = [-1, -1, 0, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                         "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                         "Transfer target TTR distance",
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    #     elif feature =='nogeo':
    #         sort_sign_list = [-1, -1, 0, -1, -1, 0, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                         "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                         "Transfer target TTR distance",
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY"]
    #     elif feature == 'pos':
    #         sort_sign_list = [-1, -1, 0, -1, -1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                         "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                         "Transfer target TTR distance",
    #                         # "p2n dist", "pron dist", "verb dist", # 2
    #                         "pron dist", "verb dist", # 2
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    #     elif feature == 'emot':
    #         sort_sign_list = [-1, -1, 0, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                         "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                         "Transfer target TTR distance",
    #                         "Emotion distance",
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    #     elif feature == 'mwe':
    #         sort_sign_list = [-1, -1, 0, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                         "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                         "Transfer target TTR distance",
    #                         "MWE distance",
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    #     elif feature == 'all':
    #         sort_sign_list = [-1, -1, 0, -1, -1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size",
    #                         "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR",
    #                         "Transfer target TTR distance",
    #                         # "p2n dist", "pron dist", "verb dist", # 3
    #                         "pron dist", "verb dist", # 2
    #                         "Emotion distance", "MWE distance",
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    #     elif feature == 'syn_only':
    #         sort_sign_list = [-1, -1, 0, -1, 1, 1, 1, 1, 1]
    #         feature_name = ["Overlap word-level",
    #                         "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio",
    #                         "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY"]
    #     elif feature == 'cult_only':
    #         sort_sign_list = [-1, -1, 0, 1, 1, 1, 1, 1, 1, 1]
    #         feature_name = ["Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance",
    #                         # "p2n dist", "pron dist", "verb dist", # 3
    #                         "pron dist", "verb dist", # 2
    #                         "Emotion distance", "MWE distance",
    #                         "GEOGRAPHIC"]



    # print("Ranking with single features:")
    # TOP_K=min(3, len(candidate_list))
    # test_inputs = np.array(test_inputs)
    # for j in range(len(feature_name)):
    #     if sort_sign_list[j] != 0:
    #         print(feature_name[j])
    #         values = test_inputs[:, j] * sort_sign_list[j]
    #         best_feat_index = np.argsort(values)
    #         for i in range(TOP_K):
    #             index = best_feat_index[i]
    #             print("%d. %s : score=%.2f" % (i, candidate_list[index][0], test_inputs[index][j]))
    # ind = list(np.argsort(-predict_scores))
    # print("Ranking (top {}):".format(print_topK))
    # for j,i in enumerate(ind[:print_topK]):
    #     print("%d. %s : score=%.2f" % (j+1, candidate_list[i][0], predict_scores[i]))
    #     contrib_scores = predict_contribs[i][:-1]
    #     contrib_ind = list(np.argsort(contrib_scores))[::-1]
    #     # print("\t1. %s : score=%.2f; \n\t2. %s : score=%.2f; \n\t3. %s : score=%.2f" %
    #     #             (feature_name[contrib_ind[0]], contrib_scores[contrib_ind[0]],
    #     #              feature_name[contrib_ind[1]], contrib_scores[contrib_ind[1]],
    #     #              feature_name[contrib_ind[2]], contrib_scores[contrib_ind[2]]))
    cand_langs = [c[0] for c in candidate_list]
    return cand_langs, -predict_scores

