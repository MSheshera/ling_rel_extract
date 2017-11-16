#!/usr/bin/env bash
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
data_path="$CUR_PROJ_ROOT/datasets/google-relation-extraction/google-relation-extraction"
log_path="$CUR_PROJ_ROOT/logs/pre_process"
mkdir -p $log_path

source_path="$CUR_PROJ_ROOT/source/"

cmd="python2 -u $source_path/pre_process/explore_data.py $data_path"
echo $cmd | tee "$log_path/explore_data_logs.txt"
eval $cmd | tee -a "$log_path/explore_data_logs.txt"