
model=vit_b16_ep50

for shots in 16
do
  for dataset in dtd imagenet oxford_pets oxford_flowers fgvc_aircraft eurosat stanford_cars food101 sun397 caltech101 ucf101
  do
    sh coop_ood/main_coop_base2new_train.sh ${dataset} ${model} ${shots} end 16 False
    sh coop_ood/main_coop_base2new_test.sh ${dataset} ${model} ${shots} end 16 False
  done
done
