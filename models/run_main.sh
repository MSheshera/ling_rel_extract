#!/usr/bin/env bash

action=$1
run_path=$2
if [[ $action == '' ]]; then
    echo "Must specify action: run_saved/train_model"
    exit 1
fi
if [[ $run_path == '' ]] && [[ $action == 'run_saved' ]] ; then
    echo "Must specify full path to trained model."
    exit 1
fi


edim=200
hdim=50
dropp=0.3
lr=0.001
run_time=`date '+%Y_%m_%d-%H_%M_%S'`
int_mapped_path="$CUR_PROJ_ROOT/processed/grec-slstm"
glove_path="$CUR_PROJ_ROOT/datasets/embeddings/glove"
if [[ action == 'train_model' ]]; then
    run_path="$CUR_PROJ_ROOT/model_runs/${run_time}-${action}"
    mkdir -p $run_path
fi

if [[ $action == 'train_model' ]]; then
    cmd="python2 -u main.py  train_model --int_mapped_path $int_mapped_path \
         --embedding_path $glove_path \
         --run_path $run_path \
         --edim $edim --hdim $hdim --dropp $dropp \
         --bsize 32 --epochs 5 --lr $lr"
    echo $cmd | tee "$run_path/train_run_log.txt"
    eval $cmd | tee -a "$run_path/train_run_log.txt"
elif [[ $action == 'run_saved' ]]; then
    cmd="python2 -u main.py  run_saved_model --int_mapped_path $int_mapped_path \
         --embedding_path $glove_path \
         --run_path $run_path"
    echo $cmd | tee "$run_path/saved_run_log.txt"
    eval $cmd | tee -a "$run_path/saved_run_log.txt"
fi