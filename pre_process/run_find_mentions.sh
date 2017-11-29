#!/usr/bin/env bash
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
partial_thresh=$1
if [[ $partial_thresh == '' ]]; then
    partial_thresh=85
fi
data_path="$CUR_PROJ_ROOT/processed/grec-readable"
proc_path="$CUR_PROJ_ROOT/processed/grec-mentionfound-${partial_thresh}"
log_path="$CUR_PROJ_ROOT/logs/pre_process"
mkdir -p $log_path
mkdir -p $proc_path

source_path="$CUR_PROJ_ROOT/source/"
script_name="find_mentions"

cmd="python2 -u $source_path/pre_process/$script_name.py \
    -d $data_path \
    -o $proc_path \
    -t $partial_thresh"
echo $cmd | tee "$log_path/${script_name}_${partial_thresh}_logs.txt"
eval $cmd | tee "$log_path/${script_name}_${partial_thresh}_logs.txt"