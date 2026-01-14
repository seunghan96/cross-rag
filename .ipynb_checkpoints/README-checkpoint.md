# Cross-RAG: Zero-Shot Retrieval-Augmented Time Series Forecasting via Cross-Attention

Anonymous submission


The markdown and code below are based on the implementation of [**TS-RAG**](https://github.com/UConn-DSIS/TS-RAG/).

## Introduction

\textbf{Cross-RAG} is a zero-shot retrieval-augmented forecasting framework that selectively attends to query-relevant retrieved samples using cross-attention.

![Figure 1: Cross-attention for RAG in TS](cross-rag/images/crossrag_overview.png)

While previous works aggregate retrieved samples without explicitly modeling the relationship between the query and the retrieved inputs, Cross-RAG performs \textit{input-aware} fusion by using \textit{cross-attention} to weight retrieved samples based on the input similarity.


![Figure 2: Overall framework of Cross-RAG](cross-rag/images/crossrag_details.png)

Cross-RAG fuses retrieved information through two branches:
- (1) \textit{Query--retrieval cross-attention} models relevance between the query and retrieved inputs and aggregates retrieved outputs conditioned on this relevance.
- (2) \textit{Retrieval self-attention} summarizes retrieved outputs in a query-independent manner to capture contextual information among retrieved samples.

The TSFM backbone and predictor are frozen, and only the additional modules are trained on general pretraining datasets.

## Installation

1. **Create a new conda environment**:
   ```bash
   conda create -n crossrag python=3.9
   ```
2. **Activate the environment**:
   ```bash
   conda activate crossrag
   ```
3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Navigate to the cross-rag directory**:
   ```bash
   cd crossrag
   ```

## Data & Model Download

You can download our preprocessed datasets and pretrained models from [Google Drive](https://drive.google.com/drive/folders/12wesXfVwFhdrUY5Kv8yuAWWqN9M77irw?usp=sharing):

## File Structure
After downloading the datasets and code, your file structure should look like this:

```
.
├── datasets
│   ├── ETT-small
│   └── weather
├── retrieval_database
├── ts-rag
│   ├── models
│   ├── results
│   │   └── forecast_evaluation
│   └── checkpoints
│       ├── base
│       ├── chronos-bolt
```

## Usage
- Step 1) Calculate similarity of pretraining dataset
    - Run `01.pretrain_dataset_similarity.ipynb`
- Step 2) Pretrain
  ```bash
  bash script/Cross-Rag-pretrain.sh
  ```
- Step 3) Zero-shot forecasting
  ```bash
  bash script/Cross-Rag-zeroshot.sh
  ```
