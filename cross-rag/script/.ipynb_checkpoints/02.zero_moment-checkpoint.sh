#!/bin/bash
# Case 2: Backbone=moment, Prediction=zero-shot
# 전 데이터셋 반복 수행

set -e
cd "$(dirname "$0")/.."  # ts-rag/ts-rag 디렉터리로 이동

# 필요한 경우 DATASETS, CHECKPOINT_MODEL_PATH 를 외부에서 override 가능
bash script/zeroshot_moment.sh


