#!/usr/bin/env bash
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
data_path="$CUR_PROJ_ROOT/processed/grec-readable"
log_path="$CUR_PROJ_ROOT/logs/pre_process"
mkdir -p $log_path

source_path="$CUR_PROJ_ROOT/source/"
script_name="find_mentions"

cmd="python2 -u $source_path/pre_process/$script_name.py $data_path"
echo $cmd | tee "$log_path/${script_name}_logs.txt"
eval $cmd > "$log_path/${script_name}_logs.txt"