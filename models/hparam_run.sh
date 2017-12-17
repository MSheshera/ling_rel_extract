#!/usr/bin/env bash
# Model name which is directory where everything from a specific run is saved.
model_name=$1
edim=$2
hdim=$3
dropp=$4
lr=$5

run_time=`date '+%Y_%m_%d'`
int_mapped_path="$CUR_PROJ_ROOT/processed/grec-slstm"
glove_path="$CUR_PROJ_ROOT/datasets/embeddings/glove"
run_path="$CUR_PROJ_ROOT/model_runs/${model_name}-${run_time}-lr${lr}-dp${dropp}-hd${hdim}-ed${edim}"
mkdir -p $run_path

cmd="python2 -u main.py  train_model --int_mapped_path $int_mapped_path \
     --embedding_path $glove_path \
     --run_path $run_path \
     --edim $edim --hdim $hdim --dropp $dropp \
     --bsize 32 --epochs 5 --lr $lr"

echo $cmd 2>&1 > "$run_path/run_log.txt"
eval $cmd 2>&1 >> "$run_path/run_log.txt"