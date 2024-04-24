#!/bin/bash

cd ..

# custom config
DATA=/mnt/sda3/Data/fs_data2
TRAINER=ProGrad

CFG=vit_b16_ep50
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
T=$1
LAM=$2
TTF=$3
TTW=$4

for SHOTS in 16
do
  for DATASET in dtd imagenet oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101

  do

    for SEED in 1 2 3
    do
        MODEL_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_T${T}_lam${LAM}/seed${SEED}
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_T${T}_lam${LAM}_ttf${TTF}_ttw${TTW}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch 50 \
            --eval-only \
            TRAINER.PROGRAD.N_CTX ${NCTX} \
            TRAINER.PROGRAD.CSC ${CSC} \
            TRAINER.PROGRAD.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.PROGRAD.T ${T} \
            TRAINER.PROGRAD.LAMBDA ${LAM} \
            TRAINER.PROGRAD.TEST_TIME_FUSE ${TTF} \
            TRAINER.PROGRAD.TEST_TIME_WEIGHT ${TTW} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
  done
done