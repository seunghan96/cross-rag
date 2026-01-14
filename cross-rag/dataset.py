import os
import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.itertools import Cyclic
from torch.utils.data import IterableDataset
from gluonts.dataset.common import FileDataset

# Import X-space distance functions
try:
    from retrieve_X import pairwise_distance, minmax_normalize
except ImportError:
    # Fallback if retrieve_X is not available
    def pairwise_distance(query_batch, db_slices, metric="euclidean"):
        """Fallback pairwise distance calculation"""
        if metric == "euclidean":
            return np.linalg.norm(query_batch[:, None, :] - db_slices[None, :, :], axis=-1)
        elif metric == "cosine":
            qn = np.linalg.norm(query_batch, axis=1, keepdims=True) + 1e-8
            dn = np.linalg.norm(db_slices, axis=1, keepdims=True).T + 1e-8
            sim = (query_batch @ db_slices.T) / (qn * dn + 1e-8)
            return 1 - sim
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def minmax_normalize(x):
        """Fallback minmax normalization"""
        x_min = x.min(axis=1, keepdims=True)
        x_max = x.max(axis=1, keepdims=True)
        x_range = x_max - x_min
        x_range = np.where(x_range == 0, 1.0, x_range)
        return (x - x_min) / x_range


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class CustomPretrainDataset(IterableDataset, ShuffleMixin):
    def __init__(
        self,
        dataset_path,
        retriever,
        mode="training",
        drop_prob=0.2,
        context_length=512,
        prediction_length=64,
        retrieve_lookback_length=64,
        top_k=5,
        output_norm=False,
        output_norm_mode="y",  # "y" (use y's min/max) or "x" (use x's min/max for y)
        retrieve_suffix=None,  # Suffix for retrieval method (e.g., Z_random_k10, X-cosine-random_k10)
    ):
        super().__init__()

        assert mode in ("training", "validation", "test")

        self.drop_prob = drop_prob
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.retriever = retriever
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.retrieve_lookback_length = retrieve_lookback_length
        self.top_k = top_k
        self.output_norm = output_norm  # Whether to normalize output sequences (y) using MinMax scaling
        self.output_norm_mode = output_norm_mode  # "y" (use y's min/max) or "x" (use x's min/max for y)
        self.retrieve_suffix = retrieve_suffix  # Suffix for retrieval method
        
        assert output_norm_mode in ("y", "x"), f"output_norm_mode must be 'y' or 'x', got {output_norm_mode}"

        # Check if random retrieval is requested
        self.use_random_retrieval = retrieve_suffix is not None and 'random' in retrieve_suffix
        # Check if this is a rev version (reverse retrieval based on output TS)
        # Rev versions have '_rev' in the dataset path
        self.is_rev_version = '_rev' in str(self.dataset_path)
        
        # Check if this is X-space or Z-space rev version
        # X-space rev: path contains '_X_' (e.g., pretrain_pairs_ctx512_X_cosine_rev)
        # Z-space rev: path contains '_Z_' or no space indicator (default to Z-space)
        dataset_path_str = str(self.dataset_path)
        self.is_x_space_rev = self.is_rev_version and '_X_' in dataset_path_str
        self.is_z_space_rev = self.is_rev_version and not self.is_x_space_rev
        
        # For X-space rev, extract metric and normalization info from path
        if self.is_x_space_rev:
            # Extract metric: _X_cosine_rev, _X_euclidean_rev, _X_dtw_rev, etc.
            if '_X_cosine' in dataset_path_str:
                self.x_space_metric = 'cosine'
            elif '_X_euclidean' in dataset_path_str:
                self.x_space_metric = 'euclidean'
            elif '_X_dtw' in dataset_path_str:
                self.x_space_metric = 'dtw'
            else:
                self.x_space_metric = 'euclidean'  # default
            
            # Check if normalization is used: _X_norm_cosine_rev, etc.
            self.x_space_normalize = '_norm_' in dataset_path_str or '_X_norm_' in dataset_path_str
        else:
            self.x_space_metric = None
            self.x_space_normalize = False

        # Ensure the dataset path exists
        if not self.dataset_path.is_dir():
            raise ValueError(f"Provided dataset_path {dataset_path} is not a directory.")
        
        # check files, all should be parquet (only check files, not directories)
        parquet_files = [f for f in self.dataset_path.iterdir() if f.is_file() and f.suffix == ".parquet"]
        non_parquet_files = [f for f in self.dataset_path.iterdir() if f.is_file() and f.suffix != ".parquet"]
        
        if len(non_parquet_files) > 0:
            raise ValueError(f"All files in the dataset_path should be parquet files. Found non-parquet files: {[f.name for f in non_parquet_files]}")
        
        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found in the dataset_path: {dataset_path}")

        # lazy loading
        self.dataset = FileDataset(self.dataset_path, freq="1H")

        if self.mode == "training":
            self.dataset = Cyclic(self.dataset)
        
        # For rev version in validation/test mode, we need to load the non-rev version
        # (X-based indices/distances) to find the most similar X
        if self.is_rev_version and self.mode != "training":
            # Get the non-rev version path by removing '_rev' from the path
            dataset_path_str = str(self.dataset_path)
            non_rev_path = dataset_path_str.replace('_rev', '')
            self.non_rev_dataset_path = Path(non_rev_path)
            
            # Check if non-rev version exists
            if not self.non_rev_dataset_path.is_dir():
                raise ValueError(f"Non-rev version dataset not found: {self.non_rev_dataset_path}. "
                               f"Rev version requires the corresponding non-rev version (X-based indices/distances).")
            
            # Load non-rev version dataset (X-based indices/distances)
            self.non_rev_dataset = FileDataset(self.non_rev_dataset_path, freq="1H")
            self.non_rev_cache = None  # Will be populated on first access


    def __iter__(self):
        iterable = iter(self.dataset)
        if self.mode == "training":
            while True:
                entry = next(iterable)
                entry = {f: entry[f] for f in ['target', 'distances', 'indices']}

                # Split stored target (max length: retrieve_lookback_length + full y)
                # Then slice to requested horizons:
                #  - x: take tail of stored x with length context_length
                #  - y: take head of stored y with length prediction_length
                full_target = entry['target']
                full_x = full_target[: self.retrieve_lookback_length]
                full_y = full_target[self.retrieve_lookback_length :]

                entry['x'] = full_x[-self.context_length :] if self.context_length > 0 else full_x
                entry['y'] = full_y[: self.prediction_length]
                
                entry['distances'] = entry['distances'][:self.top_k]
                entry['indices'] = entry['indices'][:self.top_k]

                if self.drop_prob > 0:
                    target = entry['target'].copy()
                    drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
                    mask = np.random.choice(
                        [True, False], size=len(target), p=[drop_p, 1 - drop_p]
                    )
                    target[mask] = np.nan
                    entry['target'] = target
                yield entry

        else:
            # For validation/test mode
            for entry in iterable:
                entry = {f: entry[f] for f in ['target', 'distances', 'indices']}

                full_target = entry['target']
                full_x = full_target[: self.retrieve_lookback_length]
                full_y = full_target[self.retrieve_lookback_length :]

                entry['x'] = full_x[-self.context_length :] if self.context_length > 0 else full_x
                entry['y'] = full_y[: self.prediction_length]
                entry['distances'] = entry['distances'][:self.top_k]
                entry['indices'] = entry['indices'][:self.top_k]
                
                yield entry


class Retriever_for_pretrain():
    def __init__(self, retrieval_database_path, dimension, embedding_model):
        self.retrieval_database_path = retrieval_database_path
        self.d = dimension #768
        self.index = None
        self.Y = None
        self.embedding_model = embedding_model

    def build_index(self):
        self.index = faiss.IndexFlatL2(self.d)  # euclidean distance

        database = pd.read_parquet(self.retrieval_database_path)
        embeddings = np.vstack(database["embedding"].to_numpy())
        self.x = database['x'].values
        self.y = database['y'].values
        self.whole_seq = np.concatenate([self.x.tolist(), self.y.tolist()], axis=-1)
        self.index.add(embeddings)

    def embedding(self, x_tensor):
        embeddings, _ = self.embedding_model.embed(x_tensor)
        return embeddings[:, -1, :].float().numpy()

    def search(self, query_vector, top_k, params=None):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # drop first or last
        if params is None:
            distances, indices = self.index.search(query_vector, top_k + 1)
        else:
            distances, indices = self.index.search(query_vector, top_k + 1, params=params)
        # drop first if first distance is 0
        mask = distances[:, 0] == 0
        distances = np.where(
            mask[:, None],
            distances[:, 1:], 
            distances[:, :-1]
        )
        indices = np.where(
            mask[:, None],
            indices[:, 1:], 
            indices[:, :-1]
        )
        
        return indices, distances


