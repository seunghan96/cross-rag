#!/bin/bash
# -*- coding: utf-8 -*-
set -e
cd "$(dirname "$0")/.."  

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL=CrossRAG
export INPUT_LEN=512
export RETRIEVE_SPACE=X

gpu_loc=0

run_file_pretrain=pretrain.py
run_file_zeroshot=zeroshot.py

####################################################################################
# 1. Experimental Setups
####################################################################################
seq_len=512
pred_len=64
batch_size=256

augment_mode=moe
context_length=512
prediction_length=64
top_k=15

####################################################################################
# 2. Model & Pretrained Weights
####################################################################################
backbone="Chronos"
model="ChronosBoltRetrieve"
embedding_model_type="chronos" # for Z-space


base_checkpoint_path="./checkpoints/base"

model_id_pretrain="pretrain_${backbone}_lb${lookback_length}_k${top_k}_CrossRAG"
model_id_zeroshot="zeroshot_${seq_len}_pred_${pred_len}_${context_length}_retrieve_${pred_len}_CrossRAG"

echo ""
echo ">>> Step 2: Starting Zero-shot Prediction for all datasets..."
pretrain_checkpoint_dir="./checkpoints/${model_id_pretrain}"
pretrain_checkpoint_path="${pretrain_checkpoint_dir}/best.pth"
retrieval_database_dir='../retrieval_database/'

latest_checkpoint=$(ls -t ${pretrain_checkpoint_dir}/model_steps*.pth 2>/dev/null | head -1)
if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint" ]; then
    echo ">>> Copying latest checkpoint to best.pth"
    cp "$latest_checkpoint" "$pretrain_checkpoint_path"
fi
                                        
space='X'
filename="zeroshot_${backbone}_lb${context_length}_${space}_space__k${top_k}_CrossRAG.txt"

####################################################################################
# 3. Zero-shot Forecasting
####################################################################################
echo ""
echo ">>> Step 2: Starting Zero-shot Forecasting..."
echo "Model ID: $model_id_pretrain"

datasets="${DATASETS:-"exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather electricity"}"

for dataset in $datasets; do
    retrieve_database_name=$dataset

    if [ $dataset == 'ETTm1' ] || [ $dataset == 'ETTm2' ]; then
        data='ett_m_retrieve'; metadata_frequency='minute'; root_path='../datasets/ETT-small/'
    elif [ $dataset == 'ETTh1' ] || [ $dataset == 'ETTh2' ]; then
        data='ett_h_retrieve'; metadata_frequency='hour'; root_path='../datasets/ETT-small/'
    elif [ $dataset == 'electricity' ] || [ $dataset == 'exchange_rate' ]; then
        data='custom_retrieve'; metadata_frequency='hour'; root_path="../datasets/${dataset}/"
    elif [ $dataset == 'weather' ]; then
        data='custom_retrieve'; metadata_frequency='10minutes'; root_path="../datasets/${dataset}/"
    fi
    model_id_zeroshot="${dataset}_${model_id_zeroshot}"
    result_file="results/forecast_evaluation/${filename}"

    echo "pretrain_checkpoint_path: $pretrain_checkpoint_path"

    python $run_file_zeroshot \
        --root_path $root_path \
        --data_path $dataset'.csv' \
        --model_id $model_id_zeroshot \
        --data $data \
        --top_k $top_k \
        --checkpoint_model_path $pretrain_checkpoint_path \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --lookback_length $context_length \
        --batch_size $batch_size \
        --decay_fac 0.5 \
        --freq 0 \
        --percent 100 \
        --model $model \
        --gpu_loc $gpu_loc \
        --tmax 20 \
        --cos 1 \
        --save_file_name $filename \
        --retrieval_database_dir $retrieval_database_dir \
        --dimension 768 \
        --embedding_model_type $embedding_model_type \
        --metadata_frequency $metadata_frequency \
        --metadata_database_name $retrieve_database_name \
        --augment_mode $augment_mode \
        --pretrained_model_path $base_checkpoint_path                                 

    done

    echo ">>> Zero-shot completed for this combination"
    echo ""


