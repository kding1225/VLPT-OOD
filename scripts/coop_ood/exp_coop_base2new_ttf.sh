
model=vit_b16_ep50

for shots in 16
do
  for dataset in dtd oxford_pets oxford_flowers fgvc_aircraft eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
  do
    sh coop_ood/main_coop_base2new_test_ttf.sh ${dataset} ${model} ${shots} end 16 False $1 $2
  done
done
