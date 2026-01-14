#!/bin/bash
# -*- coding: utf-8 -*-
set -e
cd "$(dirname "$0")/.."  

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL=CrossRAG

run_file_pretrain=pretrain.py
run_file_zeroshot=zeroshot.py

####################################################################################
gpu_loc=0

seq_len=512
pred_len=64
batch_size=256
retrieval_database_dir='../retrieval_database/'

augment_mode=moe
context_length=512
prediction_length=64
top_k=15

train_steps=20000
evaluation_steps=10000
optimizer=adamw
lr=0.00003
weight_decay=0.01
tmax=20
drop_prob=0.0
shuffle_buffer_length=10000

####################################################################################
space='X'
retrieve_suffix='X-cosine-norm'
similarity_space='X-cosine-norm'
####################################################################################

datasets="${DATASETS:-"exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather electricity"}"


dual_lambda=0.7


backbone="Chronos"
model="ChronosBoltRetrieve"
embedding_model_type="chronos" # for Z-space


export RETRIEVE_SPACE="$space"
export RETRIEVE_METRIC="$metric"
export TABPFN_DUAL_LAMBDA_INIT="$dual_lambda"

base_checkpoint_path="./checkpoints/base"
pretrain_db_dir="../retrieval_database"
pretrain_db_path="${pretrain_db_dir}/retrieval_database_512.parquet"
pretrain_db_url="https://huggingface.co/datasets/nkh/TS-RAG-Data/resolve/main/retrieval_database_512.parquet"

if [ ! -f "$pretrain_db_path" ]; then
    echo "Downloading pretrain database from Hugging Face..."
    mkdir -p "$pretrain_db_dir"
    wget -O "$pretrain_db_path" "$pretrain_db_url" || {
        echo "ERROR: Failed to download pretrain database"
        echo "Please manually download from: $pretrain_db_url"
        echo "And save it to: $pretrain_db_path"
        exit 1
    }
    echo "Pretrain database downloaded successfully"
else
    echo "Pretrain database already exists: $pretrain_db_path"
fi

####################################################################################
                    
model_id_pretrain="pretrain_${backbone}_lb${lookback_length}_k${top_k}_CrossRAG"

pretrain_checkpoint_dir="./checkpoints/${model_id_pretrain}"
pretrain_checkpoint_path="${pretrain_checkpoint_dir}/best.pth"

echo ""
echo ">>> Step 1: Starting Pretrain..."
echo "Model ID: $model_id_pretrain"

pretrain_skipped=false
checkpoint_found=false
skip_pretrain=false

retrieval_database_path="$pretrain_db_path"

data_path="../datasets/pretrain/pretrain_pairs_ctx${context_length}_X_space"


python $run_file_pretrain \
    --model_id $model_id_pretrain \
    --top_k $top_k \
    --retrieve_lookback_length $context_length \
    --retrieval_database_path $retrieval_database_path \
    --augment_mode $augment_mode \
    --context_length $context_length \
    --prediction_length $prediction_length \
    --data_path $data_path \
    --train_steps $train_steps \
    --evaluation_steps $evaluation_steps \
    --optimizer $optimizer \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --tmax $tmax \
    --drop_prob $drop_prob \
    --batch_size $batch_size \
    --shuffle_buffer_length $shuffle_buffer_length \
    --freeze_chronos_bolt \
    --model $model \
    --pretrained_model_path $base_checkpoint_path \
    --checkpoints ./checkpoints/ \
    --gpu_loc $gpu_loc 

