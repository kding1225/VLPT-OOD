
model=vit_b16_ep50

for shots in 16
do
  for dataset in dtd imagenet oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101

  do

    sh prograd_ood/main_prograd_base2new_test_ttf.sh ${dataset} ${model} ${shots} end 16 False $1 $2 $3 $4

  done
done