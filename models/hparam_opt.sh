#!/usr/bin/env bash

action=$1
run_name=$2 # All run subdirectories have this as their prefix.
max_jobs=$3
if [[ $action == '' ]]; then
	echo "Must specify action: sweep/pick_best"
	exit 1
fi
if [[ $run_name == '' ]]; then
	echo "Must specify run_name: slstm_hpo/sbilstm_hpo/sdeeplstm_hpo."
	exit 1
fi
if [[ $max_jobs == '' ]] && [[ $action == 'sweep' ]]; then
	max_jobs=10
	echo "Launching 10 jobs in parallel at one time."
fi

# Do a grid search over hyper-params.
edims=(200) # Embedding dimensions.
hdims=(30 50 70) # LSTM hidden vector dimensions.
dropps=(0.0 0.3 0.6) # Drop-out probability.
lrs=(0.01 0.001 0.0001) # Learning rates.

# Run at most max_job jobs in parallel.
# https://unix.stackexchange.com/a/216475/85507
if [[ $action == 'sweep' ]]; then
    N=$max_jobs
    (
    for edim in ${edims[@]};do
        for hdim in ${hdims[@]}; do
            for dropp in ${dropps[@]}; do
                for lr in ${lrs[@]}; do
                    ((i=i%N)); ((i++==0)) && wait
                    echo "Running: lr:${lr}; dropp:${dropp}; hdim:${hdim}; edim:${edim}" && srun --gres=gpu:1 --partition=m40-short --mem=32000 hparam_run.sh "${run_name}" "${edim}" "${hdim}" "${dropp}" "${lr}"  &
                done
            done
        done
    done
    )
elif [[ $action == 'pick_best' ]]; then
    # Do some post train work to collect all result graphs.
    runs_path="$CUR_PROJ_ROOT/model_runs"
    glob_pat=$run_name
    # Move all graphs to rsync_results.
    # TODO: This creates super nested annoying zips in rsync_results >_< look into it.
    rsync -chaz --include="*/" --include="*.png" --exclude="*" "$runs_path/$run_name"* "$CUR_PROJ_ROOT/rsync_results"
    rsync -chaz --include="*/" --include="*.eps" --exclude="*" "$runs_path/$run_name"* "$CUR_PROJ_ROOT/rsync_results"
    # Zip them all.
    zip -qr "$CUR_PROJ_ROOT/rsync_results/graphs_${run_name}.zip" "$CUR_PROJ_ROOT/rsync_results/$run_name"*
    # Remove the directories.
    rm -r "$CUR_PROJ_ROOT/rsync_results/$run_name"*

    # Call a python script to look at all results and print out the best.
    python2 utils.py "pick_best" "$runs_path" "$glob_pat"
fi