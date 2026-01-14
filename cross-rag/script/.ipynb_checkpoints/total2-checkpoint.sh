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

# Download pretrain data files - DISABLED: Files should be pre-generated with similarity measures
# if [ ! -d "$pretrain_data_dir" ] || [ -z "$(ls -A $pretrain_data_dir/*.parquet 2>/dev/null)" ]; then
#     echo "Downloading pretrain data files from Hugging Face..."
#     mkdir -p "$pretrain_data_dir"
#     
#     # Temporary directory for huggingface-cli download
#     temp_download_dir="../datasets/pretrain/temp_download"
#     
#     # Try using huggingface-cli if available
#     if command -v huggingface-cli &> /dev/null; then
#         echo "Using huggingface-cli to download pretrain data..."
#         mkdir -p "$temp_download_dir"
#         # Download to temp directory first, then move files
#         huggingface-cli download nkh/TS-RAG-Data --local-dir "$temp_download_dir" --include "pretrain_pairs_ctx512/*.parquet" || {
#             echo "WARNING: huggingface-cli download failed, trying wget..."
#             rm -rf "$temp_download_dir"
#             # Fallback to wget for individual files
#             for i in {0..29}; do
#                 chunk_file=$(printf "chronos-dataset-sample-50m-chunk-%03d.parquet" $i)
#                 file_url="https://huggingface.co/datasets/nkh/TS-RAG-Data/resolve/main/pretrain_pairs_ctx512/${chunk_file}"
#                 wget -O "${pretrain_data_dir}/${chunk_file}" "$file_url" || echo "Failed to download ${chunk_file}"
#             done
#         }
#         
#         # Move files from temp directory to correct location
#         if [ -d "$temp_download_dir/pretrain_pairs_ctx512" ]; then
#             echo "Moving files to correct location..."
#             mv "$temp_download_dir/pretrain_pairs_ctx512"/*.parquet "$pretrain_data_dir/" 2>/dev/null || true
#             rm -rf "$temp_download_dir"
#         elif [ -d "$temp_download_dir" ]; then
#             # Files might be directly in temp_download_dir
#             find "$temp_download_dir" -name "*.parquet" -exec mv {} "$pretrain_data_dir/" \; 2>/dev/null || true
#             rm -rf "$temp_download_dir"
#         fi
#     else
#         echo "huggingface-cli not found, downloading files individually with wget..."
#         # Download each parquet file individually
#         for i in {0..29}; do
#             chunk_file=$(printf "chronos-dataset-sample-50m-chunk-%03d.parquet" $i)
#             file_url="https://huggingface.co/datasets/nkh/TS-RAG-Data/resolve/main/pretrain_pairs_ctx512/${chunk_file}"
#             if [ ! -f "${pretrain_data_dir}/${chunk_file}" ]; then
#                 echo "Downloading ${chunk_file}..."
#                 wget -O "${pretrain_data_dir}/${chunk_file}" "$file_url" || echo "Failed to download ${chunk_file}"
#             fi
#         done
#     fi
#     
#     # Check if any files were downloaded
#     if [ -z "$(ls -A $pretrain_data_dir/*.parquet 2>/dev/null)" ]; then
#         echo "WARNING: No pretrain data files found in $pretrain_data_dir"
#         echo "Please manually download from: $pretrain_data_url"
#         echo "Or install huggingface-cli: pip install huggingface_hub[cli]"
#     else
#         echo "Pretrain data files downloaded successfully"
#         echo "Number of parquet files: $(ls -1 $pretrain_data_dir/*.parquet 2>/dev/null | wc -l)"
#     fi
# else
#     echo "Pretrain data directory already exists with files: $pretrain_data_dir"
#     echo "Number of parquet files: $(ls -1 $pretrain_data_dir/*.parquet 2>/dev/null | wc -l)"
# fi

# Loop variables
#lookback_lengths=(96 256 512 1024)
lookback_lengths=(512)
#similarity_spaces=("X-cosine" "X-euclidean" "X-dtw" "Z" "reverse")
#similarity_spaces=("Z" "X-cosine" "X-euclidean" "X-dtw" "Z-rev" "X-cosine-rev" "X-euclidean-rev" "X-dtw-rev" "Z-norm" "X-cosine-norm" "X-euclidean-norm" "X-dtw-norm" "Z-rev-norm" "X-cosine-rev-norm" "X-euclidean-rev-norm" "X-dtw-rev-norm")
#similarity_spaces=("X-cosine" "X-euclidean" "X-cosine-rev" "X-euclidean-rev" "X-cosine-norm" "X-euclidean-norm" "X-cosine-rev-norm" "X-euclidean-rev-norm")

#similarity_spaces=("X-cosine")
similarity_spaces=("X-euclidean")
#similarity_spaces=("X-cosine-norm")
#similarity_spaces=("X-euclidean-norm")

top_ks=(1 3 5)
#backbones=("Chronos" "Moment")
backbones=("Chronos")
# Output normalization: whether to normalize output sequences
output_norms=(false)
# Output normalization mode: "y" (use y's min/max) or "x" (use x's min/max for y)
output_norm_modes=("x")

# Counter for tracking progress
total_combinations=$((${#lookback_lengths[@]} * ${#similarity_spaces[@]} * ${#top_ks[@]} * ${#backbones[@]} * ${#output_norms[@]} * ${#output_norm_modes[@]}))
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
                        # Skip if output_norm is false (mode doesn't matter)
#                         if [[ "$output_norm" == false ]] && [[ "$output_norm_mode" == "x" ]]; then
#                             continue
#                         fi
                        
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
                # Step 1: Pretrain (한 번만 실행)
                # ==========================================
                echo ""
                echo ">>> Step 1: Starting Pretrain..."
                echo "Model ID: $model_id_pretrain"
                
                # Check if pretrained checkpoint already exists
                pretrain_skipped=false
                checkpoint_found=false
                
                # Convert to absolute path for reliable checking
                abs_checkpoint_path="$(cd "$(dirname "$pretrain_checkpoint_path")" 2>/dev/null && pwd)/$(basename "$pretrain_checkpoint_path")" 2>/dev/null || echo "$pretrain_checkpoint_path"
                abs_checkpoint_dir="$(cd "$pretrain_checkpoint_dir" 2>/dev/null && pwd)" 2>/dev/null || echo "$pretrain_checkpoint_dir"
                
                if [ -f "$pretrain_checkpoint_path" ]; then
                    echo ">>> Pretrained checkpoint already exists!"
                    echo ">>> Checkpoint path: $pretrain_checkpoint_path"
                    echo ">>> Skipping pretrain step. Using existing checkpoint for zero-shot..."
                    #echo ">>> Not Skipping actually haha"
                    checkpoint_found=true
                elif [ -d "$pretrain_checkpoint_dir" ]; then
                    # Check if there are any checkpoint files in the directory
                    latest_checkpoint=$(ls -t ${pretrain_checkpoint_dir}/model_steps*.pth 2>/dev/null | head -1)
                    if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint" ]; then
                        echo ">>> Pretrained checkpoint directory exists with checkpoint files!"
                        echo ">>> Latest checkpoint: $latest_checkpoint"
                        echo ">>> Copying to best.pth: $pretrain_checkpoint_path"
                        cp "$latest_checkpoint" "$pretrain_checkpoint_path"
                        echo ">>> Skipping pretrain step. Using existing checkpoint for zero-shot..."
                        checkpoint_found=true
                    else
                        echo ">>> Pretrained checkpoint directory exists but no checkpoint files found"
                        echo ">>> Directory: $pretrain_checkpoint_dir"
                        echo ">>> Will run pretrain..."
                    fi
                else
                    echo ">>> No pretrained checkpoint found. Will run pretrain..."
                    echo ">>> Expected checkpoint path: $pretrain_checkpoint_path"
                    echo ">>> Expected checkpoint directory: $pretrain_checkpoint_dir"
                fi
                
                if [ "$checkpoint_found" = false ]; then
                    # Set retrieval database path for pretrain
                    # Note: pretrain database is fixed at 512 (downloaded from Hugging Face)
                    retrieve_lookback_length=512
                    retrieval_database_path="$pretrain_db_path"
                    
                    # Check if this combination should be skipped (Z normalize or X DTW not available yet)
                    skip_pretrain=false
                    if [[ "$space" == "Z" ]] && [[ "$use_norm" == true ]]; then
                        echo ">>> Skipping pretrain: Z-space normalize version not available yet"
                        skip_pretrain=true
                    elif [[ "$space" == "X" ]] && [[ "$metric" == "dtw" ]]; then
                        echo ">>> Skipping pretrain: X-space DTW version not available yet"
                        skip_pretrain=true
                    fi
                    
                    if [ "$skip_pretrain" = true ]; then
                        echo ">>> Skipping pretrain step. Will use base checkpoint for zero-shot..."
                        pretrain_skipped=true
                    else
                        # Set data_path based on similarity space
                        if [[ "$space" == "Z" ]]; then
                            # Z-space: use default folder (no normalize version)
                            data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}"
                        elif [[ "$space" == "X" ]]; then
                            # X-space: use folder with metric and norm suffix
                            if [[ "$use_norm" == true ]]; then
                                data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_X_norm_${metric}"
                            else
                                data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_X_${metric}"
                            fi
                        else
                            # X-rev, Z-rev, etc.: use folder with _rev suffix
                            if [[ "$space" == "X-rev" ]]; then
                                # X-space reverse: use folder with metric and norm suffix + _rev
                                if [[ "$use_norm" == true ]]; then
                                    data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_X_norm_${metric}_rev"
                                else
                                    data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_X_${metric}_rev"
                                fi
                            elif [[ "$space" == "Z-rev" ]]; then
                                # Z-space reverse: use default folder + _rev
                                if [[ "$use_norm" == true ]]; then
                                    data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_Z_norm_rev"
                                else
                                    data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_Z_rev"
                                fi
                            else
                                # Fallback: use default folder
                                data_path="../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}"
                            fi
                        fi
                        
                        echo ">>> Using data_path: $data_path"
                        
                        # Check if data directory exists
                        if [ ! -d "$data_path" ]; then
                            echo "WARNING: Data directory not found: $data_path"
                            echo "Skipping pretrain step. Will use base checkpoint for zero-shot..."
                            pretrain_skipped=true
                        fi
                    fi
                    
                    # Check if pretrain database exists
                    if [ "$pretrain_skipped" = false ] && ([ -z "$retrieval_database_path" ] || [ ! -f "$retrieval_database_path" ]); then
                        echo "WARNING: Pretrain database not found: $retrieval_database_path"
                        echo "Skipping pretrain step. Will use base checkpoint for zero-shot..."
                        pretrain_skipped=true
                    fi
                    
                    if [ "$pretrain_skipped" = false ]; then
                        echo "Using pretrain database: $retrieval_database_path"
                        
                        # Run pretrain
                        output_norm_flag=""
                        if [[ "$output_norm" == true ]]; then
                            output_norm_flag="--output_norm --output_norm_mode $output_norm_mode"
                        fi
                        
                        python $run_file_pretrain \
                            --model_id $model_id_pretrain \
                            --top_k $top_k \
                            --retrieve_lookback_length $retrieve_lookback_length \
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
                            --gpu_loc $gpu_loc \
                            $output_norm_flag
                        
                        # Find the last checkpoint and copy it as best.pth
                        # Wait a bit for file system to sync
                        sleep 2
                        
                        if [ -d "$pretrain_checkpoint_dir" ]; then
                            # Find the checkpoint with the highest step number
                            last_checkpoint=$(ls -t ${pretrain_checkpoint_dir}/model_steps*.pth 2>/dev/null | head -1)
                            if [ -n "$last_checkpoint" ] && [ -f "$last_checkpoint" ]; then
                                echo "Copying last checkpoint to best.pth: $last_checkpoint"
                                cp "$last_checkpoint" "$pretrain_checkpoint_path"
                                echo "Checkpoint saved at: $pretrain_checkpoint_path"
                            else
                                echo "WARNING: No checkpoint found in $pretrain_checkpoint_dir after pretrain"
                                echo "Available files:"
                                ls -la "$pretrain_checkpoint_dir" 2>/dev/null || echo "Directory does not exist"
                                echo "Will use base checkpoint for zero-shot..."
                                pretrain_skipped=true
                            fi
                        else
                            echo "WARNING: Checkpoint directory not found: $pretrain_checkpoint_dir"
                            echo "Will use base checkpoint for zero-shot..."
                            pretrain_skipped=true
                        fi
                    fi
                fi
                
                # Set checkpoint path for zero-shot
                if [ "$pretrain_skipped" = true ]; then
                    # Use base checkpoint if pretrain was skipped or failed
                    echo ">>> Pretrain skipped or failed. Using base checkpoint for zero-shot."
                    pretrain_checkpoint_path="${base_checkpoint_path}/autogluon_model.pth"
                    if [ ! -f "$pretrain_checkpoint_path" ]; then
                        echo "ERROR: Base checkpoint not found: $pretrain_checkpoint_path"
                        echo "Skipping zero-shot for this combination..."
                        continue
                    fi
                else
                    echo ">>> Pretrain completed. Checkpoint saved at: $pretrain_checkpoint_path"
                fi
                
                # ==========================================
                # Step 2: Zero-shot Prediction (각 데이터셋별로 실행)
                # ==========================================
                echo ""
                echo ">>> Step 2: Starting Zero-shot Prediction for all datasets..."
                
                # Create filename for results
                filename_base="zeroshot_${backbone}_lb${lookback_length}_${similarity_space}_k${top_k}"
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
                    model_id_zeroshot="${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_${retrieve_suffix}"
                    
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
                        
                        python $run_file_zeroshot \
                        --root_path $root_path \
                        --data_path $dataset'.csv' \
                        --model_id $model_id_zeroshot \
                        --data $data \
                        --top_k $top_k \
                        --checkpoint_model_path $pretrain_checkpoint_path \
                        --retrieve_suffix $retrieve_suffix \
                        --seq_len $seq_len \
                        --label_len 0 \
                        --pred_len $pred_len \
                        --lookback_length $lookback_length \
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
                        --pretrained_model_path $base_checkpoint_path \
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

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Total combinations processed: $current_combination/$total_combinations"
echo "=========================================="
