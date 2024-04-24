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

for SEED in 1 2 3
do
    DIR1=output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_T${T}_lam${LAM}/seed${SEED}
    DIR2=output_dg/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_T${T}_lam${LAM}/seed${SEED}
    if [ -d "$DIR2" ]; then
        echo "Results are available in ${DIR2}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR2}"
      python train.py \
          --root ${DATA} \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR2} \
          --model-dir ${DIR1} \
          --load-epoch 50 \
          --eval-only \
          TRAINER.PROGRAD.N_CTX ${NCTX} \
          TRAINER.PROGRAD.CSC ${CSC} \
          TRAINER.PROGRAD.CLASS_TOKEN_POSITION ${CTP} \
          TRAINER.PROGRAD.T ${T} \
          TRAINER.PROGRAD.LAMBDA ${LAM} \
          DATASET.NUM_SHOTS ${SHOTS}
      fi
done
