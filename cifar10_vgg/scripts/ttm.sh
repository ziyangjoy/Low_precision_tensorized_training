#!/bin/bash

#may need smaller batch size to fit on 8gb gpu

r=5;
LP=lp;
epoch=150;
kl_mult=5e-5;
for tensor_type in  'TensorTrainMatrix';
do for r in 5;
do 
python train.py --$LP --model-type ${tensor_type} --rank ${r} --kl-multiplier $kl_mult --rank-loss True --epochs  $epoch | tee logs/VGG16_tensor_8b_rank${r}.txt;
# | tee logs/TTM_rank${r}.txt;
done
done




