#!/bin/bash

model_name=Linear_LoCo

if [ ! -d "./logs/${model_name}" ]; then
    mkdir -p "./logs/${model_name}"
fi

pred_len=1
seq_len=20
topk=15
lradj=type2

cuda_ids=5

export CUDA_VISIBLE_DEVICES=${cuda_ids}


python -u run.py \
    --task_name detection \
    --model_id ${model_name} \
    --is_training 1 \
    --model ${model_name} \
    --lradj ${lradj} \
    --dataset SWAT \
    --seq_len ${seq_len} \
    --pred_len ${pred_len} \
    --batch_size 1048 \
    --train_epochs 50 \
    --patience 2 \
    --topk 51 \
    --learning_rate 0.001 \
    --gpu 0 \
    --ad_quantile 0.92 \
    2>&1 | tee -a logs/${model_name}/SWAT_${model_name}_${seq_len}_${pred_len}.log