#!/usr/bin/env bash
action=$1
which_processed=$2
if [[ $which_processed == '' ]]; then
    echo "Must specify which processed data you want to explore: grec-mentionfound-85/grec-qd/grec-readable"
    exit 1
fi
if [[ $action == '' ]]; then
    echo "Must specify action: count_neg/count_overlap/play_deg/date_obj"
    exit 1
fi
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
data_path="$CUR_PROJ_ROOT/processed/$which_processed"
log_path="$CUR_PROJ_ROOT/logs/pre_process"
mkdir -p $log_path

source_path="$CUR_PROJ_ROOT/source/"

cmd="python2 -u $source_path/pre_process/explore_data.py $action $data_path"
echo $cmd | tee "$log_path/explore_data_logs.txt"
eval $cmd > "$log_path/explore_data_logs.txt"