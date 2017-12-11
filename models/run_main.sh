#!/usr/bin/env bash

# Model name which is directory where everything from a specific run is saved.
model_name=$1
if [[ $model_name == '' ]]; then
	echo "Specify a meaningful model name; dir where all run data is saved."
	exit 1
fi
run_time=`date '+%Y_%m_%d-%H_%M_%S'`
int_mapped_path="$CUR_PROJ_ROOT/processed/grec-slstm"
glove_path="$CUR_PROJ_ROOT/datasets/embeddings/glove"
run_path="$CUR_PROJ_ROOT/model_runs/${run_time}-${model_name}"
mkdir -p $run_path

cmd="python2 -u main.py  train_model --int_mapped_path $int_mapped_path \
     --embedding_path $glove_path \
     --run_path $run_path \
     --edim 200 --hdim 50 --dropp 0.3 \
     --bsize 32 --epochs 5 --lr 0.001"

echo $cmd | tee "$run_path/run_log.txt"
eval $cmd | tee -a "$run_path/run_log.txt"

# Move figures to easy to rsync directory. >_<
cp $run_path/*.png $CUR_PROJ_ROOT/rsync_results/
cp $run_path/*.eps $CUR_PROJ_ROOT/rsync_results/