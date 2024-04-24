#!/bin/bash

cd ..

# custom config
DATA=/mnt/sda3/Data/fs_data2
TRAINER=KgCoOp

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CSC=False  # class-specific context (False or True)
WEIGHT=$4
EP=$5

for SEED in 1 2 3
do
    # new
    MODEL_DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_W${WEIGHT}/base/seed${SEED}
    OUT_DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_W${WEIGHT}/new/seed${SEED}
    if [ -d "$OUT_DIR" ]; then
        echo "Results are available in ${OUT_DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${OUT_DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${OUT_DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${EP} \
        --eval-only \
        TRAINER.KGCOOP.N_CTX ${NCTX} \
        TRAINER.KGCOOP.CSC ${CSC} \
        TRAINER.KGCOOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.KGCOOP.W ${WEIGHT} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES "new"
    fi
done
