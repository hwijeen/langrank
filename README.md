# Ranking Transfer Languages with Pragmatically-Motivated Features for Multilingual Sentiment Analysis
This repository contains data and codes used for [the paper](). We proposed pragmatically-motivated features that operationalize linguistic concepts such as language context-level and emotion semantics.
As explained in the paper, the experimental set up consists of two steps: Optimal ranking extraction and Ranking model training. We provided the result of ranking model extraction in [this sheet](https://docs.google.com/spreadsheets/d/13qJIcksIbBz4et0vWzM3cdu5_UMrisurHs-K3AtN8Dw/edit#gid=0). Here, we train and evaluate the ranking model with the proposed pragmatically-motivated features.    


## Dependencies
Below packages are required to run the code.
```
lang2vec
lightgbm
```  

## Data
For sentiment analysis task, we have collected review dataset across 16 languages(one can find detail about each dataset in the appendix of the paper). We have formatted the dataset and put it in `datasets/sa` directory. Note that the languages are expressed in terms of ISO 639-3 codes.  
The same set of languages were used for dependency parsing task([Universal Dependencies](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837)). Again, the formatted files are located in `datasets/dep`.   


## How to run
To replicate the experiment results in Table 2 & 3, simple run
```
./run.sh
```
This code runs `langrank_train.py` and `langrank_predict.py`. One can specify the task and the feature group of interest in `task` and `featrues` variable in `runs.sh`.  
Note that `langrank_predict.py` prints performance of each cross-validation split as well as the averaged performance, in terms of Mean Average Precision(MAP) and Normalized Discounted Cumulative Gain(NDCG).
