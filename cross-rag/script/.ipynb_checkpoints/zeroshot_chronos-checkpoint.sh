filename_base=zeroshot_chronos
model=ChronosBoltRetrieve
gpu_loc=0
run_file=zeroshot.py
seq_len=512
pred_len=64
# 기본: 모든 지원 데이터셋
#datasets="${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather"}"
datasets="${DATASETS:-"ETTh1"}"
lookback_length=512
augment_mode=moe
top_k=10

batch_size=256
retrieval_database_dir='../retrieval_database/'

# CHECKPOINT_MODEL_PATH 환경변수로 override 가능
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-./checkpoints/chronos-bolt/best.pth}"

# ---------------------------------------------------------------------------
# Retrieval space/metric suffix for filenames
# ---------------------------------------------------------------------------
space="${RETRIEVE_SPACE:-Z}"       # Z or X
metric="${RETRIEVE_METRIC:-}"      # '', 'cosine', 'euclidean', 'dtw'

if [[ "$space" == "Z" || "$space" == "z" ]]; then
    suffix="Z"
elif [[ "$space" == "X" || "$space" == "x" ]]; then
    m="${metric:-cosine}"
    suffix="X-${m}"
else
    suffix="Z"
fi

filename="${filename_base}_${suffix}.txt"

# top_k_h=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20) 

# for top_k in ${top_k_h[@]};
# do
for dataset in $datasets;
do
retrieve_database_name=$dataset

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


python $run_file \
    --root_path $root_path \
    --data_path $dataset'.csv' \
    --model_id $dataset'_zeroshot_'$seq_len'_pred_'$pred_len'_'$lookback_length'_retrieve_'$pred_len'_'$suffix \
    --data $data \
    --top_k $top_k \
    --checkpoint_model_path $checkpoint_model_path \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --lookback_length $lookback_length \
    --batch_size  $batch_size \
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
    --embedding_model_type chronos \
    --metadata_frequency $metadata_frequency \
    --metadata_database_name $retrieve_database_name \
    --augment_mode $augment_mode \

done