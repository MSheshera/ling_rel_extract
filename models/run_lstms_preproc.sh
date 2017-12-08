#!/usr/bin/env bash
action=$1
if [[ $action == '' ]]; then
    echo "Must specify action: w2g_map/int_map"
    exit 1
fi
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
log_path="$CUR_PROJ_ROOT/logs/models"
mkdir -p $log_path
glove_path="$CUR_PROJ_ROOT/datasets/embeddings/glove"
splits_path="$CUR_PROJ_ROOT/processed/grec-slstm"

script_name="lstms_preproc"
source_path="$CUR_PROJ_ROOT/source/"

if [[ $action == 'w2g_map' ]]; then
    cmd="python2 -u $source_path/models/$script_name.py $action --glove_path $glove_path"
elif [[ $action == 'int_map' ]]; then
    cmd="python2 -u $source_path/models/$script_name.py $action --in_path $splits_path"
fi

echo $cmd | tee "$log_path/${script_name}_${action}_logs.txt"
eval $cmd | tee -a "$log_path/${script_name}_${action}_logs.txt"