# Weak Distribution Detectors Lead to Stronger Generalizability of Vision-Language Prompt Tuning

This repo contains source code of the paper Weak Distribution Detectors Lead to Stronger Generalizability 
of Vision-Language Prompt Tuning, AAAI 2024:

Abstract: We propose a generalized method for boosting the generalization ability of pre-trained vision-language models 
(VLMs) while fine-tuning on downstream few-shot tasks. The idea is realized by exploiting out-of-distribution (OOD) 
detection to predict whether a sample belongs to a base distribution or a novel distribution and then using 
the score generated by a dedicated competition based scoring function to fuse the zero-shot and few-shot classifier. 
The fused classifier is dynamic, which will bias towards the zero-shot classifier if a sample is more likely from the 
distribution pre-trained on, leading to improved base-to-novel generalization ability. Our method is performed only in 
test stage, which is applicable to boost existing methods without time-consuming re-training. Extensive experiments show 
that even weak distribution detectors can still improve VLMs' generalization ability. Specifically, with the help of OOD 
detectors, the harmonic mean of CoOp and ProGrad increase by 2.6 and 1.5 percentage points over 11 recognition datasets 
in the base-to-novel setting.

## How to Install
Please refer to the repo [CoOp](https://github.com/KaiyangZhou/CoOp) for installation. You need to
install Dassl.pytorch, the Dassl.pytorch folder in this repo is from ProGrad.

## How to Run

The shell codes to run the experiments are placed in folder scripts/. Here is an example for running 
the CoOp with OOD detectors:

```bash

cd scripts

# for base to new experiments:
sh coop_ood/exp_coop_base2new.sh && sh coop_ood/exp_coop_base2new_ttf.sh SOFTmsp 64.0

# for domain generalization experiments:
sh coop_ood/main_coop.sh imagenet vit_b16_ep50 16 end 16 False
sh coop_ood/exp_coop_dg.sh
sh coop_ood/exp_coop_dg_ttf.sh SOFTmsp 64.0
```

To parse the results, run:
```
python parse_test_res.py output/base2new/dtd/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/base/ --test-log
python parse_test_res.py output/base2new/dtd/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/new/ --test-log
python parse_test_res.py output/base2new/dtd/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend_ttfSOFTmsp_ttw64.0/base --test-log
python parse_test_res.py output/base2new/dtd/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend_ttfSOFTmsp_ttw64.0/new --test-log
```

## Citation
If you use this code in your research, please kindly cite

```bash
@inproceedings{DingZYWXP24,
  author       = {Kun Ding and
                  Haojian Zhang and
                  Qiang Yu and
                  Ying Wang and
                  Shiming Xiang and
                  Chunhong Pan},
  title        = {Weak Distribution Detectors Lead to Stronger Generalizability of Vision-Language
                  Prompt Tuning},
  booktitle    = {AAAI},
  pages        = {1528--1536},
  year         = {2024},
}
```
