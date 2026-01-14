#!/bin/bash
# Case 1: Backbone=chronos, Prediction=zero-shot
# 전 데이터셋 반복 수행

set -e
cd "$(dirname "$0")/.."  # ts-rag/ts-rag 디렉터리로 이동

# GPU 설정: 필요시 외부에서 CUDA_VISIBLE_DEVICES override 가능
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ---------------------------------------------------------------------------
# Retrieval 모드 선택 (인자 1개로 선택, 기본: Z-space)
# 사용법:
#   bash script/01.zero_chronos.sh z        # (1) Z-space (모델 임베딩, 기본)
#   bash script/01.zero_chronos.sh x-cos    # (2) X-space cosine
#   bash script/01.zero_chronos.sh x-euc    # (3) X-space euclidean
#   bash script/01.zero_chronos.sh x-dtw    # (4) X-space DTW
# 선택 결과는 환경변수 RETRIEVE_SPACE / RETRIEVE_METRIC 로 전달됩니다.
# ---------------------------------------------------------------------------
MODE="${1:-z}"
case "$MODE" in
  z|Z)
    export RETRIEVE_SPACE="Z"
    export RETRIEVE_METRIC=""
    echo "[MODE] Z-space (embedding) retrieval"
    ;;
  x-cos|X-COS)
    export RETRIEVE_SPACE="X"
    export RETRIEVE_METRIC="cosine"
    echo "[MODE] X-space retrieval (cosine)"
    ;;
  x-euc|X-EUC)
    export RETRIEVE_SPACE="X"
    export RETRIEVE_METRIC="euclidean"
    echo "[MODE] X-space retrieval (euclidean)"
    ;;
  x-dtw|X-DTW)
    export RETRIEVE_SPACE="X"
    export RETRIEVE_METRIC="dtw"
    echo "[MODE] X-space retrieval (DTW)"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash script/01.zero_chronos.sh [z|x-cos|x-euc|x-dtw]"
    exit 1
    ;;
esac

# 필요한 경우 DATASETS, CHECKPOINT_MODEL_PATH 를 외부에서 override 가능
bash script/zeroshot_chronos.sh


