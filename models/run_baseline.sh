#!/usr/bin/env bash
cmd="python2 -u baseline.py /mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-vlstm /mnt/nfs/work1/mccallum/smysore/ling_rel_extract/model_runs/baseline_run"
echo $cmd | tee "/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/logs/models/baseline_logs.txt"
eval $cmd | tee -a "/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/logs/models/baseline_logs.txt"