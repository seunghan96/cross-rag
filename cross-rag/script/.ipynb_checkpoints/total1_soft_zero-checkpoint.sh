#!/bin/bash
# -*- coding: utf-8 -*-
# Total experiment script: Pretrain + Zero-shot for all combinations
# Variables:
# - lookback_length: [96, 256, 512, 1024]
# - Similarity Space: [X-cosine, X-euclidean, X-dtw, Z]
# - K: [1, 3, 5, 10]
# - Backbone: [Chronos, Moment]

set -e
cd "$(dirname "$0")/.."  # ts-rag/ts-rag 디렉터리로 이동

# GPU 설정: 필요시 외부에서 CUDA_VISIBLE_DEVICES override 가능
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Soft-weight Chronos selection
export CHRONOS_SOFT_WEIGHT=1

# Wandb 비활성화 (기본값)
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# 기본 설정
run_file_pretrain=pretrain.py
run_file_zeroshot=zeroshot.py
gpu_loc=0
seq_len=512
pred_len=64
batch_size=256
retrieval_database_dir='../retrieval_database/'
augment_mode=moe
context_length=512
prediction_length=64

# Pretrain 설정
train_steps=20000
evaluation_steps=10000
optimizer=adamw
lr=0.00003
weight_decay=0.01
tmax=20
drop_prob=0.0
shuffle_buffer_length=10000

# Zero-shot 설정
#datasets="${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather"}"  # 모든 데이터셋
datasets="${DATASETS:-"exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather"}"  # 모든 데이터셋
#datasets="${DATASETS:-"exchange_rate"}"  # 모든 데이터셋

# Base checkpoint path
base_checkpoint_path="./checkpoints/base"

# Pretrain database download
pretrain_db_dir="../retrieval_database"
pretrain_db_path="${pretrain_db_dir}/retrieval_database_512.parquet"
pretrain_db_url="https://huggingface.co/datasets/nkh/TS-RAG-Data/resolve/main/retrieval_database_512.parquet"

# Pretrain data directory (contains parquet files)
# Note: This is the base directory. Actual directory is set dynamically based on similarity space.
pretrain_data_dir="../datasets/pretrain/pretrain_pairs_ctx512"
pretrain_data_url="https://huggingface.co/datasets/nkh/TS-RAG-Data/tree/main/pretrain_pairs_ctx512"

# Download pretrain database if not exists
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


lookback_lengths=(512)

similarity_spaces=("X-cosine")
#similarity_spaces=("X-euclidean")
#similarity_spaces=("X-cosine-norm")
#similarity_spaces=("X-euclidean-norm")

#top_ks=(1 3 5)
top_ks=(10 3 5)
#backbones=("Chronos" "Moment")
backbones=("Chronos")
# Output normalization: whether to normalize output sequences
output_norms=(false)
# Output normalization mode: "y" (use y's min/max) or "x" (use x's min/max for y)
output_norm_modes=("x")
# Temperature sweep (override with TEMPERATURES env)
# Note: default must be space-separated without quotes to form an array.
temperatures=(${TEMPERATURES:-0.1 0.01 5.0 10.0 20.0})

# Counter for tracking progress (include temperature sweep)
total_combinations=$((${#lookback_lengths[@]} * ${#similarity_spaces[@]} * ${#top_ks[@]} * ${#backbones[@]} * ${#output_norms[@]} * ${#output_norm_modes[@]} * ${#temperatures[@]}))
current_combination=0

echo "=========================================="
echo "Total combinations: $total_combinations"
echo "Datasets for zero-shot: $datasets"

# Main loop
for lookback_length in "${lookback_lengths[@]}"; do
    for similarity_space in "${similarity_spaces[@]}"; do
        for top_k in "${top_ks[@]}"; do
            for backbone in "${backbones[@]}"; do
                for output_norm in "${output_norms[@]}"; do
                    for output_norm_mode in "${output_norm_modes[@]}"; do
                        for temperature in "${temperatures[@]}"; do
                            # Skip if output_norm is false (mode doesn't matter)
                            
                            current_combination=$((current_combination + 1))
                            
                            echo ""
                            echo "=========================================="
                            echo "Combination $current_combination/$total_combinations"
                            echo "lookback_length: $lookback_length"
                            echo "similarity_space: $similarity_space"
                            echo "top_k: $top_k"
                            echo "backbone: $backbone"
                            echo "output_norm: $output_norm"
                            echo "output_norm_mode: $output_norm_mode"
                            echo "temperature: $temperature"
                            echo "=========================================="
                    
                    # Parse similarity space and set environment variables
                    # Check if normalization is requested (for input similarity)
                    use_norm=false
                    if [[ "$similarity_space" == *"-norm" ]]; then
                        use_norm=true
                        similarity_space_base="${similarity_space%-norm}"  # Remove -norm suffix
                    else
                        similarity_space_base="$similarity_space"
                    fi
                    
                    if [[ "$similarity_space_base" == "Z" ]]; then
                        space="Z"
                        metric=""
                        if [[ "$use_norm" == true ]]; then
                            retrieve_suffix="Z-norm_k${top_k}"
                        else
                            retrieve_suffix="Z_k${top_k}"
                        fi
                        is_reverse=false
                    elif [[ "$similarity_space_base" == "Z-rev" ]]; then
                        space="Z-rev"
                        metric=""
                        if [[ "$use_norm" == true ]]; then
                            retrieve_suffix="Z-rev-norm_k${top_k}"
                        else
                            retrieve_suffix="Z-rev_k${top_k}"
                        fi
                        is_reverse=true
                        reverse_type="Z"
                    elif [[ "$similarity_space_base" =~ ^X-.*-rev$ ]]; then
                        # X-space reverse with metric
                        space="X-rev"
                        metric=$(echo "$similarity_space_base" | sed 's/X-\(.*\)-rev/\1/')  # cosine, euclidean, or dtw
                        if [[ "$use_norm" == true ]]; then
                            retrieve_suffix="X-${metric}-rev-norm_k${top_k}"
                        else
                            retrieve_suffix="X-${metric}-rev_k${top_k}"
                        fi
                        is_reverse=true
                        reverse_type="X"
                    elif [[ "$similarity_space_base" =~ ^X- ]]; then
                        # X-space with metric
                        space="X"
                        metric=$(echo "$similarity_space_base" | cut -d'-' -f2)  # cosine, euclidean, or dtw
                        if [[ "$use_norm" == true ]]; then
                            retrieve_suffix="X-${metric}-norm_k${top_k}"
                        else
                            retrieve_suffix="X-${metric}_k${top_k}"
                        fi
                        is_reverse=false
                    else
                        space="Z"
                        metric=""
                        if [[ "$use_norm" == true ]]; then
                            retrieve_suffix="Z-norm_k${top_k}"
                        else
                            retrieve_suffix="Z_k${top_k}"
                        fi
                        is_reverse=false
                    fi
                    
                    # Set environment variables for retrieval
                    export RETRIEVE_SPACE="$space"
                    export RETRIEVE_METRIC="$metric"
                    
                    # Set model type based on backbone
                    if [[ "$backbone" == "Chronos" ]]; then
                        model="ChronosBoltRetrieve"
                        embedding_model_type="chronos"
                    else
                        model="MOMENTRetrieve"
                        embedding_model_type="chronos"  # Moment also uses chronos for embedding
                    fi
                    
                    # Create model_id for pretrain (includes all parameters except dataset)
                    # Add output_norm suffix (similarity space와 무관)
                    output_norm_suffix=""
                    if [[ "$output_norm" == true ]]; then
                        output_norm_suffix="_outputnorm_${output_norm_mode}"
                    fi
                    model_id_pretrain="pretrain_${backbone}_lb${lookback_length}_${similarity_space}_k${top_k}${output_norm_suffix}"
                
                # Pretrain checkpoint directory
                pretrain_checkpoint_dir="./checkpoints/${model_id_pretrain}"
                pretrain_checkpoint_path="${pretrain_checkpoint_dir}/best.pth"
                
                # ==========================================
                # Step 1: Pretrain (skip - reuse existing checkpoint)
                # ==========================================
                echo ""
                echo ">>> Step 1: Pretrain skipped (soft-weight zero-shot only)"
                echo "Model ID (for lookup): $model_id_pretrain"

                # Always skip training; reuse existing checkpoint if present, otherwise base
                pretrain_skipped=true
                checkpoint_found=false

                if [ -f "$pretrain_checkpoint_path" ]; then
                    echo ">>> Found existing checkpoint: $pretrain_checkpoint_path"
                    checkpoint_found=true
                else
                    echo ">>> No existing pretrain checkpoint found; falling back to base checkpoint."
                    pretrain_checkpoint_path="${base_checkpoint_path}/autogluon_model.pth"
                fi
                if [ ! -f "$pretrain_checkpoint_path" ]; then
                    echo "ERROR: No checkpoint available for zero-shot. Checked: $pretrain_checkpoint_path"
                    continue
                fi
                
                # ==========================================
                # Step 2: Zero-shot Prediction (각 데이터셋별로 실행)
                # ==========================================
                echo ""
                echo ">>> Step 2: Starting Zero-shot Prediction for all datasets..."
                
                # Create filename for results (soft-weight V1: zero-shot-only) + temperature suffix
                filename_base="zeroshot_${backbone}_lb${lookback_length}_${similarity_space}_k${top_k}_softv1_temp${temperature}"
                filename="${filename_base}.txt"
                
                # Run zero-shot for each dataset
                for dataset in $datasets; do
                    retrieve_database_name=$dataset
                    
                    # Set dataset-specific parameters
                    if [ $dataset == 'ETTm1' ] || [ $dataset == 'ETTm2' ]; then
                        data='ett_m_retrieve'
                        metadata_frequency='minute'
                        root_path='../datasets/ETT-small/'
                    elif [ $dataset == 'ETTh1' ] || [ $dataset == 'ETTh2' ]; then
                        data='ett_h_retrieve'
                        metadata_frequency='hour'
                        root_path='../datasets/ETT-small/'
                    elif [ $dataset == 'electricity' ] || [ $dataset == 'exchange_rate' ]; then
                        data='custom_retrieve'
                        metadata_frequency='hour'
                        root_path="../datasets/${dataset}/"
                    elif [ $dataset == 'weather' ]; then
                        data='custom_retrieve'
                        metadata_frequency='10minutes'
                        root_path="../datasets/${dataset}/"
                    fi
                    
                    # Create model_id for zero-shot
                    # Create model_id for zero-shot (includes output_norm suffix to match pretrain)
                    model_id_zeroshot="${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_${retrieve_suffix}${output_norm_suffix}_softv1_temp${temperature}"
                    
                    # Check if result already exists
                    result_file="results/forecast_evaluation/${filename}"
                    result_exists=false
                    
                    if [ -f "$result_file" ]; then
                        # Check if model_id already exists in the result file
                        if grep -q "^${model_id_zeroshot}$" "$result_file"; then
                            result_exists=true
                            echo "  >>> Result already exists for dataset: $dataset (model_id: $model_id_zeroshot)"
                            echo "  >>> Skipping zero-shot prediction for this dataset..."
                        fi
                    fi
                    
                    if [ "$result_exists" = false ]; then
                        echo "  Processing dataset: $dataset"
                        
                        # Run zero-shot
                        output_norm_flag=""
                        if [[ "$output_norm" == true ]]; then
                            output_norm_flag="--output_norm --output_norm_mode $output_norm_mode"
                        fi
                        
                        python "$run_file_zeroshot" \
                        --root_path "$root_path" \
                        --data_path "${dataset}.csv" \
                        --model_id "$model_id_zeroshot" \
                        --data "$data" \
                        --top_k "$top_k" \
                        --checkpoint_model_path "$pretrain_checkpoint_path" \
                        --retrieve_suffix "$retrieve_suffix" \
                        --seq_len "$seq_len" \
                        --label_len 0 \
                        --pred_len "$pred_len" \
                        --lookback_length "$lookback_length" \
                        --batch_size "$batch_size" \
                        --decay_fac 0.5 \
                        --freq 0 \
                        --percent 100 \
                        --model "$model" \
                        --gpu_loc "$gpu_loc" \
                        --tmax 20 \
                        --cos 1 \
                        --save_file_name "$filename" \
                        --retrieval_database_dir "$retrieval_database_dir" \
                        --dimension 768 \
                        --embedding_model_type "$embedding_model_type" \
                        --metadata_frequency "$metadata_frequency" \
                        --metadata_database_name "$retrieve_database_name" \
                        --augment_mode "$augment_mode" \
                        --pretrained_model_path "$base_checkpoint_path" \
                        --temperature "$temperature" \
                        $output_norm_flag
                    else
                        echo "  >>> Zero-shot prediction skipped for dataset: $dataset"
                    fi
                    
                done
                
                echo ">>> Zero-shot completed for all datasets in this combination"
                echo ""
                
                        done
                    done
                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Total combinations processed: $current_combination/$total_combinations"
echo "=========================================="
