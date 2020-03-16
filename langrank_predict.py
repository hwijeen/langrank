#python3 langrank_predict.py -o sample-data/ted-train.orig.aze -s sample-data/ted-train.orig.spm8000.aze -l aze -n 3 -t MT
#python langrank_predict.py -o datasets/olid/dan.txt -l dan -n 3 -t OLID
#!python langrank_predict.py -o datasets/sa/ara.txt -l ara -c '*ara;' -t SA -m ara
import langrank as lr
from langrank import rank_to_relevance
import os
import argparse
from utils import ndcg
import pickle
from scipy.stats import rankdata
import numpy as np

def evaluate(pred_rank, gold_rank, k=3):
    # NDCG@3 as default
    num_lang = len(pred_rank)
    pred_rel = rank_to_relevance(pred_rank, num_lang)
    gold_rel = rank_to_relevance(gold_rank, num_lang)
    pred_rel = np.expand_dims(pred_rel, axis=0)
    gold_rel = np.expand_dims(gold_rel, axis=0)
    return ndcg(y_score=pred_rel, y_true=gold_rel, k=k)

parser = argparse.ArgumentParser(description='Langrank parser.')
parser.add_argument('-o', '--orig', type=str, required=True, help='unsegmented dataset')
parser.add_argument('-s', '--seg', type=str, help='segmented dataset')
parser.add_argument('-l', '--lang', type=str, required=True, help='language code')
parser.add_argument('-n', '--num', type=int, default=3, help='print top N')
parser.add_argument('-c', '--candidates', type=str, default="all",
                    help="candidates of transfer languages, seperated by ;, use *abc to exclude language abc")
parser.add_argument('-t', '--task', type=str, default="MT", choices=["MT", "POS", "EL", "DEP", "OLID", "SA"],
                    help="The task of interested. Current options support 'MT': machine translation,"
                    "'DEP': Dependency Parsing, 'POS': POS-tagging, 'EL': Entity Linking,"
                    "'OLID': Offensive Language Identification, and 'SA': Sentiment Analysis.")
parser.add_argument('-m', '--model', type=str, default="best", help="model to be used for prediction")

params = parser.parse_args()

assert os.path.isfile(params.orig)
assert (params.seg is None or os.path.isfile(params.seg))

with open(params.orig) as inp:
    lines = inp.readlines()

bpelines = None
if params.seg is not None:
    with open(params.seg) as inp:
        bpelines = inp.readlines()

sort_dict = {'ara': 0, 'zho':1, 'nld':2, 'eng':3, 'fra':4, 'deu':5, 'kor':6, 'rus':7, 'spa':8, 'tam':9, 'tur':10}
def sort_prediction(cand_list, neg_scores):
    cand_list = [c.split('/')[2][:3] for c in cand_list]
    sorted_list = sorted(zip(cand_list, neg_scores), key=lambda x: sort_dict[x[0]])
    pred_neg_scores = [z[1] for z in sorted_list]
    pred = rankdata(pred_neg_scores)
    return pred

def load_gold(task, target_lang):
    dir_ = f'datasets/{task.lower()}'
    filename = 'rankings_wo_jpn_per_tha.pkl'
    f = open(os.path.join(dir_, filename), 'rb')
    gold_list = pickle.load(f)
    for l in gold_list:
        l.pop(l.index(0)) # drop self
    target_lang_idx = sort_dict[target_lang]
    return gold_list[target_lang_idx]

print("read lines")
prepared = lr.prepare_new_dataset(params.lang, task=params.task, dataset_source=lines, dataset_subword_source=bpelines)
print("prepared")
candidates = "all" if params.candidates == "all" else params.candidates.split(";")
task = params.task
print(f'Prediction for lang {params.lang}')
cand_langs, neg_predicted_scores = lr.rank(prepared, task=task, candidates=candidates, print_topK=params.num, model=params.model)
print("ranked")

pred = sort_prediction(cand_langs, neg_predicted_scores)
gold = load_gold(params.task, params.lang)
ndcg = evaluate(pred, gold)
print(f'Prediction is {pred}')
print(f'Gold is {gold}')
print(f'ndcg is {ndcg}')

