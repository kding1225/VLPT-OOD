backbone=vit_b16_ep100

for shots in 16
do
  for dataset in imagenet caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101
  do
      sh kgcoop_ood/main_kgcoop.sh $dataset $backbone ${shots} False 8.0
  done
done
