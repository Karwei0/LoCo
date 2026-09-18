#!/bin/bash

model_name=Linear_LoCo

if [ ! -d "./logs/${model_name}" ]; then
    mkdir -p "./logs/${model_name}"
fi

pred_len=1
lradj=type2
dataset=PSM
train_epochs=50
cuda_ids=2

seq_len_list=(3 5 10 15 20 30 40 50 60)
topk_list=(1 2 5 10 15 20 25)
lr_list=(0.001 0.002 0.0001 0.0002 0.0005 0.0008)
ad_quantile_list=(0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.90 0.89 0.88 0.87 0.86 0.85 0.84 0.83 0.82 0.81 0.80)
patience_list=(3)

export CUDA_VISIBLE_DEVICES=${cuda_ids}

total_combinations=$(( ${#seq_len_list[@]} * ${#topk_list[@]} * ${#lr_list[@]} * ${#ad_quantile_list[@]} * ${#patience_list[@]} ))
echo "总组合数: $total_combinations"

for seq_len in "${seq_len_list[@]}"; do
  for topk in "${topk_list[@]}"; do
    for lr in "${lr_list[@]}"; do
      for ad_q in "${ad_quantile_list[@]}"; do
        for pat in "${patience_list[@]}"; do
          # 构造带参数的日志文件名
          log_file="logs/${model_name}/${dataset}_${model_name}_seq${seq_len}_topk${topk}_lr${lr}_aq${ad_q}_pat${pat}.log"
          echo "Running: seq_len=$seq_len, topk=$topk, lr=$lr, ad_quantile=$ad_q, patience=$pat"
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
            --patience ${pat} \
            --topk ${topk} \
            --learning_rate ${lr} \
            --ad_quantile ${ad_q} \
            2>&1 | tee -a ${log_file}
        done
      done
    done
  done
done