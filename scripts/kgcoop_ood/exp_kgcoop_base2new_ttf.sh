
model=vit_b16_ep100

for shots in 16
do
  for dataset in dtd oxford_pets oxford_flowers fgvc_aircraft eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
  do
    sh kgcoop_ood/main_kgcoop_base2new_test_ttf.sh ${dataset} ${model} ${shots} 8.0 $1 $2 100
  done
done
