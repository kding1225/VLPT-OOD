
model=vit_b16_ep50

for shots in 16
do
  for dataset in imagenet oxford_pets dtd oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101

  do

    sh prograd_ood/main_prograd.sh ${dataset} ${model} ${shots} end 16 False $1 $2

  done
done
