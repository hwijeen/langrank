import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
# import pytest
from langrank import prepare_train_file, train
from preprocessing import build_preprocess

def test_train_mt():
    langs = ["aze", "ben", "fin"]
    datasets = [os.path.join(root, "sample-data", "ted-train.orig.{}".format(l)) for l in langs]
    seg_datasets = [os.path.join(root, "sample-data", "ted-train.orig.spm8000.{}".format(l)) for l in langs]
    rank = [[0, 1, 2], [1, 0, 2], [2, 1, 0]] # random
    tmp_dir = "tmp"
    prepare_train_file(datasets=datasets, segmented_datasets=seg_datasets,
                       langs=langs, rank=rank, tmp_dir=tmp_dir, task="MT")
    output_model = "{}/model.txt".format(tmp_dir)
    train(tmp_dir=tmp_dir, output_model=output_model)
    assert os.path.isfile(output_model)

# TODO: automatic way of loading rank
def train_olid():
    langs= ['ara', 'dan', 'ell', 'eng', 'tur']
    data_dir = 'datasets/olid/'
    datasets = [os.path.join(data_dir, f'{l}.txt') for l in langs]
    rank = [[0, 4, 2, 1, 3],
            [2, 0, 4, 1, 3],
            [2, 4, 0, 1, 3],
            [3, 1, 4, 0, 2],
            [3, 1, 4, 2, 0]]
    tmp_dir = "tmp"
    preprocess = build_preprocess()
    prepare_train_file(datasets=datasets, langs=langs, rank=rank,
                       tmp_dir=tmp_dir, task="OLID", preprocess=preprocess)
    output_model = "{}/model_child_1.txt".format(tmp_dir)
    feature_name = ['word_overlap', 'transfer_data_size', 'task_data_size',
                    'ratio_data_size', 'transfer_ttr', 'task_ttr', 'distance_ttr',
                    'genetic', 'syntactic', 'featural', 'phonological', 'inventory', 'geographical']
    train(tmp_dir=tmp_dir, output_model=output_model,
          feature_name=feature_name, task="OLID")
    assert os.path.isfile(output_model)

if __name__ == '__main__':
    train_olid()
