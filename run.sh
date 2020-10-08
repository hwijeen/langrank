#!/bin/bash
task='sa'
# features=('nocult' 'base' 'pos' 'ltq' 'emot' 'all')
# features=('data_group' 'typo_group' 'geo_group' 'ortho_group' 'cult_group')
features=('base' 'all' 'all_no_lcr' 'all_no_ltq' 'all_no_esd' 'learned' 'learned_ours' 'learned_ours_nolcr' 'learned_ours_noltq' 'learned_ours_noesd') # table1
# features=('data_group' 'typo_group' 'geo_group' 'ortho_group' 'cult_group') # table 2
num_leaves=16
max_depth=-1
learning_rate=0.1
n_estimators=100
min_child_samples=5

python langrank_train.py --task "$task" --features "${features[@]}" --num_leaves="$num_leaves" \
    --max_depth="$max_depth" --learning_rate="$learning_rate" --n_estimators="$n_estimators" \
    --min_child_samples="$min_child_samples"
python langrank_predict.py --task "$task" --features "${features[@]}"
