#!/bin/bash

model_name=Lasso

if [ ! -d "./logs/${model_name}" ]; then
    mkdir -p "./logs/${model_name}"
fi

pred_len=1
seq_len=3
topk=15
lradj=type2
dataset=SWAT
train_epochs=100

cuda_ids=5

export CUDA_VISIBLE_DEVICES=${cuda_ids}


python -u run.py \
    --task_name detection \
    --model_id ${model_name} \
    --is_training 1 \
    --model ${model_name} \
    --lradj ${lradj} \
    --dataset ${dataset} \
    --seq_len ${seq_len} \
    --pred_len ${pred_len} \
    --batch_size 1024 \
    --train_epochs ${train_epochs} \
    --patience 3 \
    --topk ${topk} \
    --learning_rate 0.001 \
    --ad_quantile 0.99 \
    --lamda1 0.1 \
    2>&1 | tee -a logs/${model_name}/${dataset}_${model_name}_${seq_len}_${pred_len}.log