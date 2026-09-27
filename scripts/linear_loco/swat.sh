#!/bin/bash

model_name=Linear_LoCo

if [ ! -d "./logs/${model_name}" ]; then
    mkdir -p "./logs/${model_name}"
fi

pred_len=1
seq_len=0
topk=107
lradj=type2

cuda_ids=5

export CUDA_VISIBLE_DEVICES=${cuda_ids}


python -u run.py \
    --task_name loss \
    --model_id ${model_name} \
    --is_training 1 \
    --model ${model_name} \
    --lradj ${lradj} \
    --dataset SWAT \
    --seq_len ${seq_len} \
    --pred_len ${pred_len} \
    --batch_size 2048 \
    --train_epochs 20 \
    --patience 3 \
    --topk 107 \
    --learning_rate 0.0001 \
    --gpu 0 \
    --ad_quantile 0.95 \
    --compress_causal_mat 0 \
    --compress_causal_mat_method MA \
    --root_analysis 0 \
    2>&1 | tee -a logs/${model_name}/SWAT_${model_name}_${seq_len}_${pred_len}.log