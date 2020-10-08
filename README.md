# Cross-Cultural Similarity Features for Cross-lingual Transfer Learning of Pragmatically Motivated Tasks
This repository contains data and codes used for [the paper](). We proposed pragmatically-motivated features that operationalize linguistic concepts such as language context-level and emotion semantics.
As explained in the paper, we trained a gradient boosted decision tree based ranking model to select transfer languages. The code trains and evaluate the ranking model with the proposed pragmatically-motivated features. The code is based on this [repository](https://github.com/neulab/langrank).


## Dependencies
Below packages are required to run the code.
```
lang2vec
lightgbm
```  

## Data
For sentiment analysis task, we have collected review dataset across 16 languages(one can find details about each dataset in the appendix of the paper). We have formatted the dataset and put it in `datasets/sa` directory. Note that the languages are expressed in terms of ISO 639-3 codes.
The same set of languages were used for dependency parsing task([Universal Dependencies](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837)). Again, the formatted files are located in `datasets/dep`.
The raw zero-shot results that are used to train the ranking model is in `Optimal ranking extraction raw data.xlsx`.


## How to run
To replicate the experiment results in Table 2 & 3, simply run
```
./run.sh
```
This code runs `langrank_train.py` and `langrank_predict.py`. One can specify the task and the feature group of interest in `task` and `featrues` variable in `runs.sh`. We used LambdaRank to train the model.
Note that `langrank_predict.py` prints performance of each cross-validation split as well as the averaged performance, in terms of Mean Average Precision(MAP) and Normalized Discounted Cumulative Gain(NDCG).
