#!/bin/bash
# -*- coding: utf-8 -*-
# Random K retrieval experiment script (TabPFN dualhead_R_as_key variant): Zero-shot with random sampling (no pretrain)
# Uses existing pretrained checkpoints and retrieve_X_norm_random.py

set -e
cd "$(dirname "$0")/.."  # ts-rag/ts-rag 디렉터리로 이동

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CHRONOS_TABPFN=1
export CHRONOS_TABPFN_VARIANT=dualhead_R_as_key
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

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

train_steps=20000
evaluation_steps=10000
optimizer=adamw
lr=0.00003
weight_decay=0.01
tmax=20
drop_prob=0.0
shuffle_buffer_length=10000

datasets="${DATASETS:-"exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather electricity"}"

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

lookback_lengths=(512)
similarity_spaces=("X-cosine-norm-random" "X-euclidean-norm-random" "X-correlation-random")
similarity_spaces=("X-cosine-norm-random")
top_ks=(15 3 5 10 20)
backbones=("Chronos")
output_norms=(false)
output_norm_modes=("x")
# hyperparameter: dualhead_R_as_key mix lambda init
dual_lambda_list=(${TABPFN_DUAL_LAMBDA_LIST:-0.7})

total_combinations=$((${#lookback_lengths[@]} * ${#similarity_spaces[@]} * ${#top_ks[@]} * ${#backbones[@]} * ${#output_norms[@]} * ${#output_norm_modes[@]} * ${#dual_lambda_list[@]}))
current_combination=0

echo "=========================================="
echo "Total combinations: $total_combinations"
echo "Datasets for zero-shot: $datasets"

for lookback_length in "${lookback_lengths[@]}"; do
    for similarity_space in "${similarity_spaces[@]}"; do
        for top_k in "${top_ks[@]}"; do
            for backbone in "${backbones[@]}"; do
                for output_norm in "${output_norms[@]}"; do
                    for output_norm_mode in "${output_norm_modes[@]}"; do
                        for dual_lambda in "${dual_lambda_list[@]}"; do
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
                            echo "dual_lambda_init: $dual_lambda"
                            echo "=========================================="

                            # For random sampling: parse -random suffix
                            use_random=false
                            if [[ "$similarity_space" == *"-random" ]]; then
                                use_random=true
                                similarity_space_base="${similarity_space%-random}"  # Remove -random suffix
                            else
                                similarity_space_base="$similarity_space"
                            fi
                            
                            use_norm=false
                            if [[ "$similarity_space_base" == *"-norm" ]]; then
                                use_norm=true
                                similarity_space_base="${similarity_space_base%-norm}"  # Remove -norm suffix
                            fi

                            if [[ "$similarity_space_base" == "Z" ]]; then
                                space="Z"; metric=""
                                if [[ "$use_random" == true ]]; then
                                    retrieve_suffix="Z_random_k${top_k}"
                                elif [[ "$use_norm" == true ]]; then
                                    retrieve_suffix="Z-norm_k${top_k}"
                                else
                                    retrieve_suffix="Z_k${top_k}"
                                fi
                            elif [[ "$similarity_space_base" =~ ^X- ]]; then
                                space="X"
                                metric=$(echo "$similarity_space_base" | cut -d'-' -f2)
                                if [[ "$use_random" == true ]]; then
                                    retrieve_suffix="X-${metric}-norm_random_k${top_k}"
                                elif [[ "$use_norm" == true ]]; then
                                    retrieve_suffix="X-${metric}-norm_k${top_k}"
                                else
                                    retrieve_suffix="X-${metric}_k${top_k}"
                                fi
                            else
                                space="Z"; metric=""
                                if [[ "$use_random" == true ]]; then
                                    retrieve_suffix="Z_random_k${top_k}"
                                elif [[ "$use_norm" == true ]]; then
                                    retrieve_suffix="Z-norm_k${top_k}"
                                else
                                    retrieve_suffix="Z_k${top_k}"
                                fi
                            fi


                            export RETRIEVE_SPACE="$space"
                            export RETRIEVE_METRIC="$metric"
                            export TABPFN_DUAL_LAMBDA_INIT="$dual_lambda"

                            if [[ "$backbone" == "Chronos" ]]; then
                                model="ChronosBoltRetrieve"; embedding_model_type="chronos"
                            else
                                model="MOMENTRetrieve"; embedding_model_type="chronos"
                            fi

                            # For random sampling: use existing pretrained checkpoint (pretrain is already done)
                            # Remove -random suffix to get original similarity_space for checkpoint lookup
                            original_similarity_space="${similarity_space/-random/}"
                            
                            output_norm_suffix=""
                            if [[ "$output_norm" == true ]]; then
                                output_norm_suffix="_outputnorm_${output_norm_mode}"
                            fi
                            model_id_pretrain="pretrain_${backbone}_lb${lookback_length}_${original_similarity_space}_k${top_k}${output_norm_suffix}_TabPFN_dualhead_R_as_key_l${dual_lambda}"

                            pretrain_checkpoint_dir="./checkpoints/${model_id_pretrain}"
                            pretrain_checkpoint_path="${pretrain_checkpoint_dir}/best.pth"

                            echo ""
                            echo ">>> Step 1: Using existing pretrained checkpoint (pretrain skipped for random sampling)"
                            echo ">>> Looking for checkpoint: $pretrain_checkpoint_path"

                            # Check if pretrained checkpoint exists
                            checkpoint_found=false

                            if [ -f "$pretrain_checkpoint_path" ]; then
                                echo ">>> Pretrained checkpoint found!"
                                echo ">>> Checkpoint path: $pretrain_checkpoint_path"
                                checkpoint_found=true
                            elif [ -d "$pretrain_checkpoint_dir" ]; then
                                latest_checkpoint=$(ls -t ${pretrain_checkpoint_dir}/model_steps*.pth 2>/dev/null | head -1)
                                if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint" ]; then
                                    echo ">>> Pretrained checkpoint directory exists with checkpoint files!"
                                    echo ">>> Latest checkpoint: $latest_checkpoint"
                                    echo ">>> Copying to best.pth: $pretrain_checkpoint_path"
                                    cp "$latest_checkpoint" "$pretrain_checkpoint_path"
                                    checkpoint_found=true
                                else
                                    echo ">>> ERROR: Pretrained checkpoint directory exists but no checkpoint files found"
                                    echo ">>> Directory: $pretrain_checkpoint_dir"
                                    echo ">>> Cannot proceed without checkpoint. Skipping this combination."
                                    continue
                                fi
                            else
                                echo ">>> ERROR: No pretrained checkpoint found!"
                                echo ">>> Expected checkpoint path: $pretrain_checkpoint_path"
                                echo ">>> Expected checkpoint directory: $pretrain_checkpoint_dir"
                                echo ">>> Cannot proceed without checkpoint. Skipping this combination."
                                continue
                            fi

                            if [ "$checkpoint_found" = true ]; then
                                echo ">>> Using existing checkpoint for zero-shot with random sampling"
                            else
                                echo ">>> ERROR: Checkpoint not found. Skipping zero-shot for this combination."
                                continue
                            fi

                            echo ""
                            echo ">>> Step 2: Starting Zero-shot Prediction for all datasets..."
                            
                            

                            filename_base="zeroshot_${backbone}_lb${lookback_length}_${similarity_space}_k${top_k}_TabPFN_dualhead_R_as_key_l${dual_lambda}_random"
                            filename="${filename_base}.txt"

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

                                model_id_zeroshot="${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_${retrieve_suffix}${output_norm_suffix}_TabPFN_dualhead_R_as_key_l${dual_lambda}_random"

                                result_file="results/forecast_evaluation/${filename}"
                                result_exists=false
                                if [ -f "$result_file" ]; then
                                    if grep -q "^${model_id_zeroshot}$" "$result_file"; then
                                        result_exists=true
                                    fi
                                fi

                                echo "pretrain_checkpoint_path: $pretrain_checkpoint_path"
                            
                                if [ "$result_exists" = false ]; then
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
                                fi

                            done

                            echo ">>> Zero-shot completed for this combination"
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


