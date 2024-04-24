#!/bin/bash

cd ..

# custom config
DATA=/mnt/sda3/Data/fs_data2
TRAINER=CoOp

CFG=vit_b16_ep50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)

for SHOTS in 16
do
  for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r imagenet
  do
    for SEED in 1 2 3
    do
        DIR1=output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        DIR2=output_dg/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
    done
  done
done
