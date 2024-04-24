#!/bin/bash

cd ..

# custom config
DATA=/mnt/sda3/Data/fs_data2
TRAINER=CoOp

CFG=vit_b16_ep50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
TTF=$1
TTW=$2

for SHOTS in 16
do
  for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101 imagenet
  do
      for SEED in 1 2 3
      do
          MODEL_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
          DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_ttf${TTF}_ttw${TTW}/seed${SEED}
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
              TRAINER.COOP.N_CTX ${NCTX} \
              TRAINER.COOP.CSC ${CSC} \
              TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
              TRAINER.COOP.TEST_TIME_FUSE ${TTF} \
              TRAINER.COOP.TEST_TIME_WEIGHT ${TTW} \
              DATASET.NUM_SHOTS ${SHOTS}
          fi
      done

  done
done
