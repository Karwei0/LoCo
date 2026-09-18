#!/bin/bash

model_name=Kernel_LoCo

# 创建日志目录
if [ ! -d "./logs/${model_name}" ]; then
    mkdir -p "./logs/${model_name}"
fi

# 固定参数
pred_len=1
lradj=type2
dataset=MSL
train_epochs=20
cuda_ids=5

# 网格参数（去重后的值）
seq_len_list=(3 5 10 20 40 60)
topk_list=(1 2 5 10 20 30 40 50 55)
lr_list=(0.0008)
ad_quantile_list=(0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.90 0.89 0.88 0.87 0.86 0.85 0.84 0.83 0.82 0.81 0.80)
patience_list=(3)
D_list=(20 100 150)
sigma_list=(0.5 1.0 1.5 2.0)

export CUDA_VISIBLE_DEVICES=${cuda_ids}

total_combinations=$(( ${#seq_len_list[@]} * ${#topk_list[@]} * ${#lr_list[@]} * ${#ad_quantile_list[@]} * ${#patience_list[@]} * ${#D_list[@]} * ${#sigma_list[@]}))
echo "总组合数: $total_combinations"

for seq_len in "${seq_len_list[@]}"; do
  for topk in "${topk_list[@]}"; do
    for lr in "${lr_list[@]}"; do
      for ad_q in "${ad_quantile_list[@]}"; do
        for pat in "${patience_list[@]}"; do
          for D in "${D_list[@]}"; do
            for sigma in "${sigma_list[@]}"; do
              log_file="logs/${model_name}/${dataset}_${model_name}_seq${seq_len}_topk${topk}.log"
              echo "Running: seq_len=$seq_len, topk=$topk, lr=$lr, ad_quantile=$ad_q, patience=$pat, D=$D, sigma=$sigma"
              python -u run.py \
                --task_name detection \
                --model_id ${dataset}_${model_name} \
                --is_training 1 \
                --model ${model_name} \
                --lradj ${lradj} \
                --dataset ${dataset} \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --batch_size 512 \
                --train_epochs ${train_epochs} \
                --patience ${pat} \
                --topk ${topk} \
                --learning_rate ${lr} \
                --ad_quantile ${ad_q} \
                --D ${D} \
                --sigma ${sigma} \
                2>&1 | tee -a ${log_file}
            done
          done
        done
      done
    done
  done
done