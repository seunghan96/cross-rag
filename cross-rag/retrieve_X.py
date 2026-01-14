import os
import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils.tools import get_borders

frequency_dict = {'ETTh1': 'hour', 'ETTh2': 'hour', 'ETTm1': 'minute', 'ETTm2': 'minute',
                  'electricity': 'hour', 'weather': '10minutes', 'traffic': 'hour', 'exchange_rate': 'hour', 'illness': 'hour'}
subdir_name_dict = {'ETTh1': 'ETT-small', 'ETTh2': 'ETT-small', 'ETTm1': 'ETT-small', 'ETTm2': 'ETT-small',
                    'electricity': 'electricity', 'weather': 'weather', 'traffic': 'traffic'}

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


def pairwise_distance(query_batch, db_slices):
    """
    query_batch: (B, L) - already normalized
    db_slices:   (N, L) - already normalized
    return: distances (B, N)
    """
    q = query_batch
    d = db_slices
    qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    dn = np.linalg.norm(d, axis=1, keepdims=True).T + 1e-8  # (1, N)
    sim = (q @ d.T) / (qn * dn + 1e-8)  # (B, N)
    return 1 - sim


def create_database(raw_data, timestamps, lookback_length, metadata):
    """
    raw_data: list/array, 1D time series
    timestamps: timestamps of same length
    """
    slices = []
    sliced_timestamps = []
    for start in range(0, len(raw_data) - lookback_length + 1):
        end = start + lookback_length
        slices.append(raw_data[start:end])
        sliced_timestamps.append(timestamps[end - 1])  # End timestamp of slice

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
    save_database(databases, os.path.join(database_dir, f'{dataset_name}_{frequency}_{lookback_length}_X_space.pkl'))


# ---------------------------------------------------------------------
# Retriever (X space with MinMax normalization)
# ---------------------------------------------------------------------
class RetrieverX:
    def __init__(self, database_dir, root_dir, metadata, seed, lookback_length):
        self.database_dir = database_dir
        self.metadata = metadata
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
            database_path = f'{database_name}_{self.metadata["frequency"]}_{self.metadata["lookback_length"]}_X_space.pkl'
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
                    # Filter by begin/end (exclude prediction period)
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

        distances = pairwise_distance(query_batch, self.slices)  # (B, N)

        # Get top_k smallest distances
        idx = np.argpartition(distances, top_k, axis=1)[:, :top_k]
        # Sort
        row_indices = np.arange(distances.shape[0])[:, None]
        sorted_order = np.argsort(distances[row_indices, idx], axis=1)
        top_indices = idx[row_indices, sorted_order]
        top_distances = distances[row_indices, top_indices]

        # Calculate boundary / timestamp index
        boundary_array = np.array(self.boundary)
        boundary_idx_batch = np.digitize(top_indices, boundary_array) - 1
        timestamp_idx_batch = top_indices - boundary_array[boundary_idx_batch]

        return top_distances, boundary_idx_batch, timestamp_idx_batch


# ---------------------------------------------------------------------
# do_retrieve (X space with MinMax normalization)
# ---------------------------------------------------------------------
def do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode,
                top_k, context_length, prediction_length, seed, 
                save=True, knowledge_base_root_dir=None):
    """
    Input: original data, retrieval database, metadata and retrieve mode
    Output: retrieved_data (with boundary_idx, timestamp_idx, distance)
    
    knowledge_base_root_dir: root directory for knowledge base dataset (for transfer learning).
                            If None, uses root_dir. If different from root_dir, enables transfer learning mode.
    """
    # Use knowledge_base_root_dir for knowledge base if provided, otherwise use root_dir
    if knowledge_base_root_dir is None:
        knowledge_base_root_dir = root_dir
    
    # Check if this is transfer learning (knowledge base and target are different)
    is_transfer = knowledge_base_root_dir != root_dir or (metadata.get('database_name') and original_data_name not in metadata['database_name'])
    
    # load original data
    original_data_path = os.path.join(root_dir, original_data_name + '.csv')
    original_data = pd.read_csv(original_data_path)
    variable_names = original_data.columns[1:]  # exclude 'date'
    print(f'There are {len(variable_names)} variables in the original data')
    if is_transfer:
        print(f'Transfer learning mode: Knowledge base dataset(s): {metadata.get("database_name", [])}, Target dataset: {original_data_name}')

    # initialize the retrieved data
    boundary_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    timestamp_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    distance_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)

    # get borders
    border1s, border2s = get_borders(original_data_name, context_length, len(original_data))
    if mode == 'only_self_train':
        for var_idx, var_name in enumerate(variable_names):
            print(f'----------Retrieving for variable: {var_name}')
            retriever = RetrieverX(
                database_dir=retrieval_database_dir,
                root_dir=knowledge_base_root_dir,  # Use knowledge_base_root_dir for knowledge base
                metadata=metadata,
                seed=seed,
                lookback_length=context_length,
            )

            # For transfer learning, check if variable exists in knowledge base
            # If not, use all variables from knowledge base
            variable_filter = [var_name]
            if is_transfer:
                # Check if variable exists in knowledge base
                sample_db_name = metadata.get('database_name', [original_data_name])[0]
                sample_db_path = f'{sample_db_name}_{metadata.get("frequency", "hour")}_{metadata.get("lookback_length", context_length)}_X_space.pkl'
                sample_db_full_path = os.path.join(retrieval_database_dir, sample_db_path)
                if os.path.exists(sample_db_full_path):
                    from retrieve import load_database
                    sample_db = load_database(sample_db_full_path)
                    if var_name not in sample_db.keys():
                        # Variable doesn't exist in knowledge base, use all variables
                        variable_filter = None
                        print(f'  Variable {var_name} not found in knowledge base, using all variables')
                    else:
                        print(f'  Using matched variable {var_name} from knowledge base')
                else:
                    # Database doesn't exist yet, will be created
                    variable_filter = [var_name]

            # For transfer learning, don't filter by borders of knowledge base
            if is_transfer:
                retriever.build_index(
                    y_length=prediction_length,
                    variable_filter=variable_filter,
                    begin=None,  # Use all data from knowledge base
                    end=None
                )
            else:
                retriever.build_index(
                    y_length=prediction_length,
                    variable_filter=variable_filter,
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

                # No search needed for train/val interval
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
        # For X-space, embedding_tuning is None, so we use None
        retrieved_data_path = os.path.join(root_dir, f'{original_data_name}_retrieve_{retrieval_database_names}_{metadata["lookback_length"]}_{mode}_None.csv')
        print(f'Saving the retrieved data to {retrieved_data_path}')
        retrieved_data.to_csv(retrieved_data_path, index=False)

    return retrieved_data


if __name__ == "__main__":
    # Example execution (modify if needed)
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

    do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode,
                top_k, context_length, prediction_length, seed,  save=save)


