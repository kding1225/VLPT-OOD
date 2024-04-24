#!/bin/bash

cd ..

# custom config
DATA=/mnt/sda3/Data/fs_data2
TRAINER=ProGrad

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
CSC=$6  # class-specific context (False or True)
T=$7
LAM=$8

# new
for SEED in 1 2 3
do
    MODEL_DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_T${T}_lam${LAM}/base/seed${SEED}
    OUT_DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_T${T}_lam${LAM}/new/seed${SEED}
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
        TRAINER.PROGRAD.N_CTX ${NCTX} \
        TRAINER.PROGRAD.CSC ${CSC} \
        TRAINER.PROGRAD.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.PROGRAD.T ${T} \
        TRAINER.PROGRAD.LAMBDA ${LAM} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES "new"
    fi
done
