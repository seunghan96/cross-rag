import os
import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. Falling back to naive DTW implementation.")

from utils.tools import get_borders

frequency_dict = {'ETTh1': 'hour', 'ETTh2': 'hour', 'ETTm1': 'minute', 'ETTm2': 'minute',
                  'electricity': 'hour', 'weather': '10minutes', 'traffic': 'hour', 'exchange_rate': 'hour', 'illness': 'hour'}
subdir_name_dict = {'ETTh1': 'ETT-small', 'ETTh2': 'ETT-small', 'ETTm1': 'ETT-small', 'ETTm2': 'ETT-small',
                    'electricity': 'electricity', 'weather': 'weather', 'traffic': 'traffic'}


# ---------------------------------------------------------------------
# MinMax Normalization (0-1 range)
# ---------------------------------------------------------------------
def minmax_normalize(x):
    """
    MinMax normalize each time series to [0, 1] range.
    x: (N, L) or (L,) array
    Returns: normalized array with same shape
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    # Avoid division by zero (constant time series)
    x_range = x_max - x_min
    x_range = np.where(x_range == 0, 1.0, x_range)
    normalized = (x - x_min) / x_range
    
    if was_1d:
        normalized = normalized.squeeze(0)
    
    return normalized


# ---------------------------------------------------------------------
# Similarity / distance functions (data space)
# ---------------------------------------------------------------------
def cosine_distance(a, b):
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8
    sim = np.sum(a * b, axis=-1) / (a_norm.squeeze() * b_norm.squeeze() + 1e-8)
    return 1 - sim  # distance = 1 - cosine_similarity


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


def dtw_distance(a, b):
    """
    Efficient DTW distance calculation using dtaidistance package.
    Falls back to naive implementation if package is not available.
    """
    if DTW_AVAILABLE:
        return dtw.distance(a, b)
    else:
        # Fallback to naive O(n^2) DTW implementation
        n, m = len(a), len(b)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i - 1] - b[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return dp[n, m]


def pairwise_distance(query_batch, db_slices, metric="euclidean"):
    """
    query_batch: (B, L) - already normalized
    db_slices:   (N, L) - already normalized
    return: distances (B, N)
    """
    if metric == "euclidean":
        # (B, N)
        return np.linalg.norm(query_batch[:, None, :] - db_slices[None, :, :], axis=-1)
    elif metric == "cosine":
        q = query_batch
        d = db_slices
        qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        dn = np.linalg.norm(d, axis=1, keepdims=True).T + 1e-8  # (1, N)
        sim = (q @ d.T) / (qn * dn + 1e-8)  # (B, N)
        return 1 - sim
    elif metric == "dtw":
        # Efficient DTW calculation using dtaidistance
        if DTW_AVAILABLE:
            # Use distance_matrix for efficient batch computation
            # Combine all queries and db_slices for batch computation
            B, L = query_batch.shape
            N = db_slices.shape[0]
            combined = np.vstack([query_batch, db_slices])  # (B+N, L)
            
            # Compute full distance matrix
            dist_matrix = dtw.distance_matrix_fast(combined, parallel=True)
            
            # Extract distances: query_batch (first B rows) to db_slices (last N rows)
            # dist_matrix[i, B+j] is distance from query[i] to db_slices[j]
            distances = dist_matrix[:B, B:].astype(np.float32)
            return distances
        else:
            # Fallback to naive implementation
            dist_list = []
            for q in query_batch:
                dist_list.append(np.array([dtw_distance(q, db) for db in db_slices]))
            return np.stack(dist_list, axis=0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


# ---------------------------------------------------------------------
# DB 생성/저장/로드 (raw X-space slices)
# ---------------------------------------------------------------------
def create_database(raw_data, timestamps, lookback_length, metadata):
    """
    raw_data: list/array, 1D 시계열
    timestamps: 동일 길이 타임스탬프
    """
    slices = []
    sliced_timestamps = []
    for start in range(0, len(raw_data) - lookback_length + 1):
        end = start + lookback_length
        slices.append(raw_data[start:end])
        sliced_timestamps.append(timestamps[end - 1])  # 슬라이스 끝 시점

    slices = np.array(slices, dtype=np.float32)
    sliced_timestamps = np.array(sliced_timestamps)

    database = {
        'slices': slices,              # (num_slices, lookback_length)
        'timestamps': sliced_timestamps,
        'metadata': metadata
    }
    return database


def save_database(database, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(database, f)


def load_database(file_path):
    with open(file_path, 'rb') as f:
        database = pickle.load(f)
    return database


def generate_retrieval_database(dataset_name, lookback_length, database_dir, root_dir):
    root_dir = Path(root_dir)
    database_dir = Path(database_dir)
    data_path = root_dir / (dataset_name + '.csv')
    frequency = frequency_dict[dataset_name]
    df = pd.read_csv(data_path)
    variables = df.columns[1:]

    databases = {}
    for variable in variables:
        raw_data = df[variable].tolist()
        timestamps = df['date'].tolist()
        metadata = {
            'dataset_name': dataset_name,
            'variable_name': variable,
            'lookback_length': lookback_length,
            'frequency': frequency,
        }
        database = create_database(raw_data, timestamps, lookback_length, metadata)
        databases[variable] = database

    # Database file contains raw data, so it's independent of similarity calculation method
    save_database(databases, os.path.join(database_dir, f'{dataset_name}_{frequency}_{lookback_length}.pkl'))


# ---------------------------------------------------------------------
# Retriever (X space with MinMax normalization)
# ---------------------------------------------------------------------
class RetrieverX:
    def __init__(self, database_dir, root_dir, metadata, seed, lookback_length, metric="euclidean"):
        self.database_dir = database_dir
        self.metadata = metadata
        self.metric = metric
        self.lookback_length = lookback_length
        self.root_dir = root_dir

    def build_index(self, y_length, begin=None, end=None, variable_filter=None):
        self.slices = []
        self.timestamps = []
        self.retrieved_metadata = []
        self.boundary = [0]

        database_paths = []
        for database_name in self.metadata['database_name']:
            # Database file contains raw data, independent of similarity calculation method
            database_path = f'{database_name}_{self.metadata["frequency"]}_{self.metadata["lookback_length"]}.pkl'
            if not os.path.exists(self.database_dir):
                print(f'{self.database_dir} does not exist, building the dir...')
                os.makedirs(self.database_dir)
            db_full_path = os.path.join(self.database_dir, database_path)
            if os.path.exists(db_full_path):
                # Check if database has correct format (X-space with 'slices')
                try:
                    test_db = load_database(db_full_path)
                    # Check if it's X-space format (has 'slices' for at least one variable)
                    has_correct_format = False
                    for test_key in test_db.keys():
                        if isinstance(test_db[test_key], dict) and 'slices' in test_db[test_key]:
                            has_correct_format = True
                            break
                    if not has_correct_format:
                        print(f'{database_path} exists but has wrong format (Z-space or other). Regenerating as X-space format...')
                        os.remove(db_full_path)
                        generate_retrieval_database(
                            dataset_name=database_name,
                            lookback_length=self.metadata['lookback_length'],
                            database_dir=self.database_dir,
                            root_dir=self.root_dir
                        )
                except Exception as e:
                    print(f'Error checking {database_path}: {e}. Regenerating...')
                    if os.path.exists(db_full_path):
                        os.remove(db_full_path)
                    generate_retrieval_database(
                        dataset_name=database_name,
                        lookback_length=self.metadata['lookback_length'],
                        database_dir=self.database_dir,
                        root_dir=self.root_dir
                    )
                database_paths.append(database_path)
            else:
                print(f'{database_path} does not exist, building the database...')
                generate_retrieval_database(
                    dataset_name=database_name,
                    lookback_length=self.metadata['lookback_length'],
                    database_dir=self.database_dir,
                    root_dir=self.root_dir
                )
                database_paths.append(database_path)

        print(f'Build X-space index (with MinMax normalization) with database: {database_paths}')

        for database_path in database_paths:
            print(f'load database: {database_path}')
            database = load_database(os.path.join(self.database_dir, database_path))
            # Debug: print database structure
            if len(database) == 0:
                raise ValueError(f"Database {database_path} is empty")
            print(f"Database keys: {list(database.keys())[:5]}...")  # Print first 5 keys
            for key in database.keys():
                if variable_filter is None or key in variable_filter:
                    # Debug: check database[key] structure
                    if not isinstance(database[key], dict):
                        raise ValueError(
                            f"Database {database_path} for variable {key} is not a dict. "
                            f"Got type: {type(database[key])}, value: {database[key]}"
                        )
                    if 'slices' not in database[key]:
                        raise ValueError(
                            f"Database {database_path} for variable {key} missing 'slices' key. "
                            f"Available keys: {list(database[key].keys())}"
                        )
                    slices = database[key]['slices']
                    # begin/end 필터 (예측 구간 제외)
                    if begin is None:
                        filter_begin = 0
                    else:
                        filter_begin = begin
                    if end is None:
                        filter_end = -y_length
                    else:
                        filter_end = end
                    slices = slices[filter_begin:filter_end, :]
                    
                    # MinMax normalize each slice individually
                    slices = minmax_normalize(slices)

                    self.slices.append(slices)
                    self.timestamps.append(database[key]['timestamps'])
                    self.retrieved_metadata.append(database[key]['metadata'])
                    self.boundary.append(slices.shape[0])

            self.boundary = [sum(self.boundary[:i]) for i in range(1, len(self.boundary) + 1)]

        if len(self.slices) == 0:
            raise ValueError("No slices loaded. Check variable_filter or database paths.")

        self.slices = np.concatenate(self.slices, axis=0)  # (N, L)
        self.timestamps = np.concatenate(self.timestamps, axis=0)

    def search(self, query_batch, top_k, drop_first=False):
        """
        query_batch: (B, lookback_length)
        return: distances (B, top_k), boundary_idx (B, top_k), timestamp_idx (B, top_k)
        """
        if query_batch.ndim == 1:
            query_batch = query_batch.reshape(1, -1)
        
        # MinMax normalize query batch
        query_batch = minmax_normalize(query_batch)

        distances = pairwise_distance(query_batch, self.slices, metric=self.metric)  # (B, N)

        # top_k: cosine/euclidean -> 최소값, dtw -> 최소값 동일
        idx = np.argpartition(distances, top_k, axis=1)[:, :top_k]
        # 정렬
        row_indices = np.arange(distances.shape[0])[:, None]
        sorted_order = np.argsort(distances[row_indices, idx], axis=1)
        top_indices = idx[row_indices, sorted_order]
        top_distances = distances[row_indices, top_indices]

        # boundary / timestamp index 계산
        boundary_array = np.array(self.boundary)
        boundary_idx_batch = np.digitize(top_indices, boundary_array) - 1
        timestamp_idx_batch = top_indices - boundary_array[boundary_idx_batch]

        return top_distances, boundary_idx_batch, timestamp_idx_batch


# ---------------------------------------------------------------------
# do_retrieve (X space with MinMax normalization)
# ---------------------------------------------------------------------
def do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode,
                top_k, context_length, prediction_length, seed, metric="euclidean",
                save=True, retrieve_suffix=""):
    """
    input: original data, retrieval database, metadata and retrieve mode
    output: retrieved_data (with boundary_idx / timestamp_idx / distance)
    """
    # load original data
    original_data_path = os.path.join(root_dir, original_data_name + '.csv')
    original_data = pd.read_csv(original_data_path)
    variable_names = original_data.columns[1:]  # exclude 'date'
    print(f'There are {len(variable_names)} variables in the original data')

    # initialize the retrieved data
    boundary_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    timestamp_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    distance_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)

    # get borders
    border1s, border2s = get_borders(original_data_name, context_length, len(original_data))

    if mode == 'only_self_train':
        print(f'----------For one variable, we retrieve data from itself training set (X space with MinMax normalization)----------')
        for var_idx, var_name in enumerate(variable_names):
            print(f'----------Retrieving for variable: {var_name}')
            retriever = RetrieverX(
                database_dir=retrieval_database_dir,
                root_dir=root_dir,
                metadata=metadata,
                seed=seed,
                lookback_length=context_length,
                metric=metric,
            )

            retriever.build_index(
                y_length=prediction_length,
                variable_filter=[var_name],
                begin=border1s[0],
                end=border2s[0]
            )

            sequence = original_data[var_name].values

            # batch search
            start_idx_list = list(range(0, len(sequence) - context_length - prediction_length + 1))
            end_idx_list = [start_idx + context_length for start_idx in start_idx_list]

            search_batch_size = 256
            batch_num = math.ceil(len(start_idx_list) / search_batch_size)
            for batch_idx in tqdm(range(batch_num)):
                start_idx_batch = start_idx_list[batch_idx * search_batch_size:min((batch_idx + 1) * search_batch_size, len(start_idx_list))]
                end_idx_batch = end_idx_list[batch_idx * search_batch_size:min((batch_idx + 1) * search_batch_size, len(start_idx_list))]

                # train/val 구간은 검색 불필요
                if end_idx_batch[-1] <= border2s[0]:
                    boundary_idx_batch = np.zeros((len(start_idx_batch), top_k))
                    timestamp_idx_batch = np.zeros((len(start_idx_batch), top_k))
                    distance_batch = np.zeros((len(start_idx_batch), top_k))
                    boundary_idx_matrix[start_idx_batch, var_idx, :] = boundary_idx_batch
                    timestamp_idx_matrix[start_idx_batch, var_idx, :] = timestamp_idx_batch
                    distance_matrix[start_idx_batch, var_idx, :] = distance_batch
                else:
                    seq_x_batch = np.array([sequence[start_idx:end_idx] for start_idx, end_idx in zip(start_idx_batch, end_idx_batch)], dtype=np.float32)
                    distances_batch, boundary_idx_batch, timestamp_idx_batch = retriever.search(seq_x_batch, top_k=top_k)
                    boundary_idx_matrix[start_idx_batch, var_idx, :] = boundary_idx_batch
                    timestamp_idx_matrix[start_idx_batch, var_idx, :] = timestamp_idx_batch
                    distance_matrix[start_idx_batch, var_idx, :] = distances_batch

    boundary_idx_df = pd.DataFrame(boundary_idx_matrix.reshape(len(original_data), -1),
                                   columns=[f'boundary_idx_{var}_{k}' for var in variable_names for k in range(top_k)])
    timestamp_idx_df = pd.DataFrame(timestamp_idx_matrix.reshape(len(original_data), -1),
                                    columns=[f'timestamp_idx_{var}_{k}' for var in variable_names for k in range(top_k)])
    distance_df = pd.DataFrame(distance_matrix.reshape(len(original_data), -1),
                               columns=[f'distance_{var}_{k}' for var in variable_names for k in range(top_k)])
    retrieved_data = pd.concat([original_data, boundary_idx_df, timestamp_idx_df, distance_df], axis=1)

    assert (pd.concat([boundary_idx_df.isna().sum().reset_index(drop=True),
                       timestamp_idx_df.isna().sum().reset_index(drop=True),
                       distance_df.isna().sum().reset_index(drop=True)], axis=1).nunique(axis=1) == 1).all(), "NaN counts are not the same in all columns"

    if save:
        retrieval_database_names = '_'.join(metadata['database_name'])
        suffix = retrieve_suffix if retrieve_suffix is not None else ''
        suffix = suffix if suffix == '' else f'_{suffix}'
        # Match the format expected by zeroshot.py: {name}_retrieve_{db_names}_{len}_{mode}_{embedding_tuning}{suffix}.csv
        # For X-space, embedding_tuning is None, so we use None
        retrieved_data_path = os.path.join(root_dir, f'{original_data_name}_retrieve_{retrieval_database_names}_{metadata["lookback_length"]}_{mode}_None{suffix}.csv')
        retrieved_data_path = os.path.join(root_dir, f'{original_data_name}_retrieve_{retrieval_database_names}_{metadata["lookback_length"]}_{mode}_None.csv')
        print(f'Saving the retrieved data to {retrieved_data_path}')
        retrieved_data.to_csv(retrieved_data_path, index=False)

    return retrieved_data


if __name__ == "__main__":
    # 예시 실행 (필요 시 수정)
    original_data_name = 'ETTh1'
    retrieval_database_dir = '../retrieval_database_X/'
    root_dir = './datasets/ETT-small'
    metadata = {
        'database_name': ['ETTh2'],
        'lookback_length': 96,
        'frequency': 'hour',
    }
    mode = 'only_self_train'
    save = False
    seed = 42
    top_k = 5
    context_length = 96
    prediction_length = 96
    metric = 'cosine'  # 'euclidean', 'cosine', 'dtw' 중 선택

    do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode,
                top_k, context_length, prediction_length, seed, metric=metric, save=save)


