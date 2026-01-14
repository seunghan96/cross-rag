import numpy as np
import torch
from tqdm import tqdm


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


def compute_topk_indices_distances_cosine(query_array, database_array, k=20, batch_size=50000, 
                                          gpu_id=0, device=None, USE_GPU=True):
    """
    Compute top-k indices and distances using cosine similarity with mandatory MinMax normalization.
    Query and database are normalized to [0, 1] range before computing cosine similarity.
    """
    # Set device default
    if device is None:
        device = torch.device(f'cuda:{gpu_id}' if USE_GPU and torch.cuda.is_available() else 'cpu')

    query_for_search = minmax_normalize(query_array)
    database_for_search = minmax_normalize(database_array)

    n_queries, dim = query_for_search.shape
    n_db, dim_db = database_for_search.shape


    # GPU path
    if USE_GPU and torch.cuda.is_available():
        query_cpu = torch.from_numpy(query_for_search.astype('float32'))
        database_cpu = torch.from_numpy(database_for_search.astype('float32'))
        
        # Normalize for cosine similarity
        database_cpu = torch.nn.functional.normalize(database_cpu, p=2, dim=1)

        all_indices = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)

        safe_batch_size = min(batch_size, 20000)
        max_distance_matrix_gb = 4
        max_distance_matrix_elements = (max_distance_matrix_gb * 1024**3) // 2  # float16
        db_chunk_size = max(200000, max_distance_matrix_elements // safe_batch_size)
        db_chunk_size = min(db_chunk_size, n_db)


        num_batches = (n_queries + safe_batch_size - 1) // safe_batch_size

        for batch_start in tqdm(range(0, n_queries, safe_batch_size),
                               desc="Cosine similarity (GPU)",
                               total=num_batches):
            batch_end = min(batch_start + safe_batch_size, n_queries)
            batch_queries = query_cpu[batch_start:batch_end]

            # Normalize queries for cosine similarity
            batch_queries_gpu = torch.nn.functional.normalize(batch_queries, p=2, dim=1).to(device, dtype=torch.float16, non_blocking=True)
            del batch_queries

            batch_topk_distances = torch.full((batch_end - batch_start, k), float('inf'), device=device, dtype=torch.float16)
            batch_topk_indices = torch.zeros((batch_end - batch_start, k), dtype=torch.long, device=device)

            for db_start in range(0, n_db, db_chunk_size):
                db_end = min(db_start + db_chunk_size, n_db)
                data_db_chunk = database_cpu[db_start:db_end]

                # Cosine similarity: compute dot product, convert to distance (1 - similarity)
                data_db_chunk = data_db_chunk.to(device, dtype=torch.float16, non_blocking=True)
                chunk_distances = torch.mm(batch_queries_gpu, data_db_chunk.t())
                chunk_distances.neg_().add_(1.0)

                chunk_topk_size = min(k, chunk_distances.shape[1])
                chunk_topk_distances, chunk_topk_indices_local = torch.topk(chunk_distances, chunk_topk_size, dim=1, largest=False)
                del chunk_distances

                chunk_topk_indices = chunk_topk_indices_local + db_start

                combined_distances = torch.cat([batch_topk_distances, chunk_topk_distances], dim=1)
                combined_indices = torch.cat([batch_topk_indices, chunk_topk_indices], dim=1)
                del chunk_topk_distances, chunk_topk_indices, chunk_topk_indices_local

                final_topk_distances, final_topk_indices_local = torch.topk(combined_distances, k, dim=1, largest=False)
                batch_topk_distances.copy_(final_topk_distances)
                batch_topk_indices.copy_(combined_indices.gather(1, final_topk_indices_local))
                del combined_distances, combined_indices, final_topk_distances, final_topk_indices_local, data_db_chunk
                torch.cuda.empty_cache()

            all_indices[batch_start:batch_end] = batch_topk_indices.cpu().numpy()
            all_distances[batch_start:batch_end] = batch_topk_distances.float().cpu().numpy()
            del batch_queries_gpu, batch_topk_distances, batch_topk_indices
            torch.cuda.empty_cache()
            gc.collect()

        del query_cpu, database_cpu
        torch.cuda.empty_cache()
        gc.collect()

    else:
        # CPU path
        query_tensor = torch.from_numpy(query_for_search.astype('float32'))
        database_tensor = torch.from_numpy(database_for_search.astype('float32'))
        if USE_GPU and torch.cuda.is_available():
            query_tensor = query_tensor.to(device)
            database_tensor = database_tensor.to(device)

        all_indices = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)

        for batch_start in tqdm(range(0, n_queries, batch_size), desc="Cosine similarity (CPU)"):
            batch_end = min(batch_start + batch_size, n_queries)
            batch_queries = query_tensor[batch_start:batch_end]

            # Cosine similarity
            batch_queries_norm = torch.nn.functional.normalize(batch_queries, p=2, dim=1)
            database_norm = torch.nn.functional.normalize(database_tensor, p=2, dim=1)
            similarities = torch.mm(batch_queries_norm, database_norm.t())
            distances = 1 - similarities

            top_distances, top_indices = torch.topk(distances, k, dim=1, largest=False)
            all_indices[batch_start:batch_end] = top_indices.cpu().numpy()
            all_distances[batch_start:batch_end] = top_distances.cpu().numpy()

        del query_tensor, database_tensor
        if USE_GPU and torch.cuda.is_available():
            torch.cuda.empty_cache()

    indices_list = [all_indices[i] for i in range(n_queries)]
    distances_list = [all_distances[i] for i in range(n_queries)]
    return indices_list, distances_list

