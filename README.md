# Cross-RAG: Zero-Shot Retrieval-Augmented Time Series Forecasting via Cross-Attention


## Introduction

**Cross-RAG** is a zero-shot retrieval-augmented forecasting framework that ***selectively*** attends to query-relevant retrieved samples using **cross-attention**.

<img src="cross-rag/images/crossrag_overview.png" alt="Figure 1: Cross-attention for RAG in TS" width="60%">

While previous works aggregate retrieved samples *without explicitly modeling the relationship between the query and the retrieved inputs*, Cross-RAG performs **input-aware** fusion by using **cross-attention** to weight retrieved samples based on the input similarity.

<br>

![Figure 2: Overall framework of Cross-RAG](cross-rag/images/crossrag_details.png)

Cross-RAG fuses retrieved information through two branches:

- (1) **Query--retrieval cross-attention** models relevance between the query and retrieved inputs and aggregates retrieved outputs conditioned on this relevance.
- (2) **Retrieval self-attention** summarizes retrieved outputs in a query-independent manner to capture contextual information among retrieved samples.

The TSFM backbone and predictor are frozen, and only the additional modules are trained on general pretraining datasets.

<br>

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

<br>

## Download datasets & models

You can download our preprocessed datasets and pretrained models from [Google Drive](https://drive.google.com/drive/folders/12wesXfVwFhdrUY5Kv8yuAWWqN9M77irw?usp=sharing):

<br>

## File Structure

After downloading the datasets and code, your file structure should look like this:

```
.
├── datasets
│   ├── ETT-small
│   └── weather
├── retrieval_database
├── cross-rag
│   ├── models
│   ├── results
│   │   └── forecast_evaluation
│   └── checkpoints
│       ├── base
│       ├── chronos-bolt
```

<br>

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


## Citation

If you find this work useful, please cite:

```bibtex
@article{lee2026cross,
  title={Cross-RAG: Zero-Shot Retrieval-Augmented Time Series Forecasting via Cross-Attention},
  author={Lee, Seunghan and Lee, Jaehoon and Seo, Jun and Yoo, Sungdong and Kim, Minjae and Lim, Tae Yoon and Kang, Dongwan and Choi, Hwanil and Lee, SoonYoung and Ahn, Wonbin},
  journal={arXiv preprint arXiv:2603.14709},
  year={2026}
}
```

## Acknowledgements

This codebase builds upon [**TS-RAG**](https://github.com/UConn-DSIS/TS-RAG/).



## Contact

Seunghan Lee — seunghan.lee@lgresearch.ai
