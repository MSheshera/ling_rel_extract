#!/usr/bin/env bash
action=$1
which_processed=$2
if [[ $action == '' ]]; then
    echo "Must specify action: make_split"
    exit 1
fi
if [[ $which_processed == '' ]]; then
    echo "Must specify which processed data: grec-mentionfound-85"
    exit 1
fi
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
data_path="$CUR_PROJ_ROOT/processed/$which_processed"
out_path="$CUR_PROJ_ROOT/processed/grec-split"
log_path="$CUR_PROJ_ROOT/logs/pre_process"
mkdir -p $log_path
mkdir -p $out_path

script_name="build_dataset"
source_path="$CUR_PROJ_ROOT/source/"

cmd="python2 -u $source_path/pre_process/$script_name.py $action \
    --in_path $data_path \
    --out_path $out_path"
echo $cmd | tee "$log_path/${script_name}_${action}_logs.txt"
eval $cmd | tee -a "$log_path/${script_name}_${action}_logs.txt"