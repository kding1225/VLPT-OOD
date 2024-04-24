#!/bin/bash

cd ..

# custom config
DATA=/mnt/sda3/Data/fs_data2
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
CSC=$6  # class-specific context (False or True)

for SEED in 1 2 3
do
    # new
    MODEL_DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/base/seed${SEED}
    OUT_DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/new/seed${SEED}
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
        --load-epoch 50 \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES "new"
    fi
done
