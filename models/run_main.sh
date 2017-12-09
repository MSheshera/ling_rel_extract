#!/usr/bin/env bash
cmd="python2 -u main.py /mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-slstm \
    /mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/embeddings/glove \
    /mnt/nfs/work1/mccallum/smysore/ling_rel_extract/saved_models \
    /mnt/nfs/work1/mccallum/smysore/ling_rel_extract/results"

eval $cmd | tee "/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/logs/models/test.txt"