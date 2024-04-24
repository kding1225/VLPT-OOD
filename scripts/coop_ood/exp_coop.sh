
backbone=vit_b16_ep50

for shots in 16
do
  for dataset in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 sun397 imagenet
  do
      sh coop_ood/main_coop.sh $dataset $backbone ${shots} end 16 False
  done
done

