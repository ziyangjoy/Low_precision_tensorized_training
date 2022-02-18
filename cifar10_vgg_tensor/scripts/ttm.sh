#!/bin/bash

#may need smaller batch size to fit on 8gb gpu

r=5;
for tensor_type in  'TensorTrainMatrix';
do for kl_mult in 5e-5;
do python train.py --model-type ${tensor_type} --rank ${r}  --kl-multiplier $kl_mult --rank-loss True 
# | tee logs/${tensor_type}_${r}_${kl_mult}_bit8_scale.txt;
done
done



