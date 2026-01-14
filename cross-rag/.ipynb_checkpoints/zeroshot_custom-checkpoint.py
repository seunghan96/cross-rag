import os
import time
import torch
import random
import argparse
import warnings
import numpy as np
import importlib

from transformers import AutoConfig
from chronos import ChronosPipeline
from sklearn.preprocessing import StandardScaler

from models.moment import MOMENTPipelineWithRetrieval
from utils.tools import test, test_retrieve
from retrieve import load_database
from retrieve_Z import do_retrieve as do_retrieve_Z
from retrieve_X import do_retrieve as do_retrieve_X
from retrieve_X_reverse import do_retrieve as do_retrieve_X_reverse
from retrieve_Z_reverse import do_retrieve as do_retrieve_Z_reverse
from retrieve_X_norm import do_retrieve as do_retrieve_X_norm
from retrieve_X_reverse_norm import do_retrieve as do_retrieve_X_reverse_norm
from retrieve_Z_reverse_norm import do_retrieve as do_retrieve_Z_reverse_norm

# Patch retrieval datasets to use custom slicing (seq_len tail for x, pred_len head for y)
try:
    from data_provider import data_loader as _dl_base
    from data_provider.data_loader_custom import (
        Dataset_ETT_hour_retrieve_custom,
        Dataset_ETT_minute_retrieve_custom,
        Dataset_Custom_retrieve_custom,
    )
    _dl_base.Dataset_ETT_hour_retrieve = Dataset_ETT_hour_retrieve_custom
    _dl_base.Dataset_ETT_minute_retrieve = Dataset_ETT_minute_retrieve_custom
    _dl_base.Dataset_Custom_retrieve = Dataset_Custom_retrieve_custom
    # Reload data_factory so that it picks up patched classes
    import data_provider.data_factory as _df
    importlib.reload(_df)
    data_provider = _df.data_provider
except Exception as e:
    warnings.warn(f"Failed to patch custom retrieval loaders: {e}")
    from data_provider.data_factory import data_provider

# Patch metric to align prediction length with ground-truth length
try:
    import utils.tools as _tools_mod
    import utils.metrics as _metrics_mod

    _orig_metric = getattr(_tools_mod, "metric", None)
    _orig_metric_metrics = getattr(_metrics_mod, "metric", None)

    def _metric_trim(pred, true):
        # Align last dimension to ground-truth
        if pred.shape[-1] != true.shape[-1]:
            pred = pred[..., : true.shape[-1]]
        return (_orig_metric or _orig_metric_metrics)(pred, true)

    if _orig_metric:
        _tools_mod.metric = _metric_trim
    if _orig_metric_metrics:
        _metrics_mod.metric = _metric_trim
except Exception as e:
    warnings.warn(f"Failed to patch metric trimming: {e}")

# Select ChronosBolt implementation (default vs soft-weight vs TabPFN) via env flag.
USE_TABPFN = os.environ.get("CHRONOS_TABPFN", "0") == "1"
TABPFN_VARIANT = os.environ.get("CHRONOS_TABPFN_VARIANT", "").lower()
USE_SOFT_WEIGHT = os.environ.get("CHRONOS_SOFT_WEIGHT", "0") == "1" and not USE_TABPFN
if USE_TABPFN:
    if TABPFN_VARIANT == "concat":
        from models.ChronosBolt_TabPFN_concat import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "cross":
        from models.ChronosBolt_TabPFN_cross import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "gate":
        from models.ChronosBolt_TabPFN_gate import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "kernel":
        from models.ChronosBolt_TabPFN_kernel import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "twostage":
        from models.ChronosBolt_TabPFN_twostage import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "slot":
        from models.ChronosBolt_TabPFN_slot import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "adaptivetemp":
        from models.ChronosBolt_TabPFN_adaptivetemp import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "residual":
        from models.ChronosBolt_TabPFN_residual import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "prototype":
        from models.ChronosBolt_TabPFN_prototype import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "posbias":
        from models.ChronosBolt_TabPFN_posbias import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "multiscale":
        from models.ChronosBolt_TabPFN_multiscale import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead":
        from models.ChronosBolt_TabPFN_dualhead import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_custom":
        from models.ChronosBolt_TabPFN_dualhead_custom import ChronosBoltPipelineWithRetrieval as ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "motif":
        from models.ChronosBolt_TabPFN_motif import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_dropout":
        from models.ChronosBolt_TabPFN_dualhead_dropout import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_learnable":
        from models.ChronosBolt_TabPFN_dualhead_learnable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_custom_learnable":
        from models.ChronosBolt_TabPFN_dualhead_custom_learnable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore                
    elif TABPFN_VARIANT == "dualhead_kv_kplusx":
        from models.ChronosBolt_TabPFN_dualhead_kv_kplusx import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_kplusx_vy":
        from models.ChronosBolt_TabPFN_dualhead_kplusx_vy import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_kplusx_vy_learnable":
        from models.ChronosBolt_TabPFN_dualhead_kplusx_vy_learnable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_xgate":
        from models.ChronosBolt_TabPFN_dualhead_xgate import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_xgate_learnable":
        from models.ChronosBolt_TabPFN_dualhead_xgate_learnable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_xself":
        from models.ChronosBolt_TabPFN_dualhead_xself import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_xself_learnable":
        from models.ChronosBolt_TabPFN_dualhead_xself_learnable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_wo_flow":
        from models.ChronosBolt_TabPFN_dualhead_wo_flow import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_xbias":
        from models.ChronosBolt_TabPFN_dualhead_xbias import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_xbias_learnable":
        from models.ChronosBolt_TabPFN_dualhead_xbias_learnable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_stable":
        from models.ChronosBolt_TabPFN_dualhead_stable import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "triplehead":
        from models.ChronosBolt_TabPFN_triplehead import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    elif TABPFN_VARIANT == "dualhead_raw":
        from models.ChronosBolt_TabPFN_dualhead_raw import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
    else:
        from models.ChronosBolt_TabPFN import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
elif USE_SOFT_WEIGHT:
    from models.ChronosBolt_soft_weight import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval  # type: ignore
else:
    from models.ChronosBolt import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Chronos-bolt')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)
parser.add_argument('--all', type=int, default=0)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=20)

parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=1.0 , required=False)
parser.add_argument('--save_file_name', type=str, default=None)
parser.add_argument('--temperature', type=float, default=1.0, help='temperature for distance-based attention (soft-weight variant)')
parser.add_argument('--gpu_loc', type=int, default=1)
parser.add_argument('--n_scale', type=float, default=-1)
parser.add_argument('--method', type=str, default='')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--metadata', type=dict, default={})
parser.add_argument('--metadata_database_name', type=str, default='ETTh2')
parser.add_argument('--metadata_frequency', type=str, default='hour')
parser.add_argument('--mode', type=str, default='only_self_train')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--retrieval_database_dir', type=str, default='../retrieval_database/')
parser.add_argument('--dimension', type=int, default=768)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--lookback_length', type=int, default=512)
parser.add_argument('--retrieve_suffix', type=str, default='')
parser.add_argument('--output_norm', action='store_true', default=False, help='whether to normalize output sequences')
parser.add_argument('--output_norm_mode', type=str, default='y', choices=['y', 'x'], 
                    help='statistics for output normalization: "y" (use y\'s min/max) or "x" (use x\'s min/max for y)')

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')

parser.add_argument('--checkpoint_model_path', type=str, default='None')
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base')

args = parser.parse_args()


if args.save_file_name is not None : 
    log_fine_name = args.save_file_name

device_address = 'cuda:'+str(args.gpu_loc)
        
SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}
mses = []
maes = []
print(args.model_id)

args.metadata['lookback_length'] = args.lookback_length
args.metadata['frequency'] = args.metadata_frequency
args.metadata['database_name'] = args.metadata_database_name.split(' ')
ori_data_path = args.data_path

best_model_path = args.checkpoint_model_path
print(f'best_model_path: {best_model_path}')
if not os.path.exists(best_model_path):
    exit('no corresponding checkpoint!!')
    
if args.freq == 0:
    args.freq = 'h'

if 'retrieve' in args.model_id:
    retrieval_database_names = '_'.join(args.metadata['database_name'])
    suffix = args.retrieve_suffix if args.retrieve_suffix is not None else ''
    suffix = suffix if suffix == '' else f'_{suffix}'
    ori_data_path_base = ori_data_path.split(".")[0]
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------#
    retrieved_data_path = os.path.join(args.root_path, f'{ori_data_path_base}_retrieve_{retrieval_database_names}_{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}{suffix}.csv')
    #retrieved_data_path = os.path.join(args.root_path, f'{ori_data_path_base}_retrieve_{ori_data_path_base}_{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}{suffix}.csv')
    retrieved_data_path = os.path.join(args.root_path, f'{ori_data_path_base}_retrieve_{retrieval_database_names}_{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}.csv')
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------#

    if os.path.exists(retrieved_data_path):
        print(f'----------retrieval for {args.model_id} has done!!----------')
    else:
        print(f'----------retrieving for {args.model_id} ...----------')
        if 'chronos' in args.embedding_model_type:
            if args.embedding_tuning == None:
                model_path = "amazon/chronos-t5-base"
            else: 
                model_path = f"../tuning_results/{args.metadata_database_name}_{str(args.seq_len)}_chronos_{args.embedding_tuning}"
                if not os.path.exists(model_path):
                    exit('embedding model path does not exist!!')
            embedding_model = ChronosPipeline.from_pretrained(
                model_path,
                device_map=device_address,
                torch_dtype=torch.bfloat16,
            )
        else:
            print('embedding model type error!!')
            exit()
        top_k = args.top_k if args.top_k > 20 else 20
        
        # Check if normalization is requested
        use_norm = args.retrieve_suffix and 'norm' in args.retrieve_suffix
        
        # Check if reverse retrieval is requested and which type
        if args.retrieve_suffix and 'rev' in args.retrieve_suffix:
            # Parse reverse type and metric from retrieve_suffix
            if 'Z-rev' in args.retrieve_suffix:
                # Use Z-space reverse retrieval
                # Extract metric from suffix (e.g., "Z-cosine-rev-norm_k3" -> "cosine")
                if 'cosine' in args.retrieve_suffix:
                    metric = 'cosine'
                elif 'euclidean' in args.retrieve_suffix:
                    metric = 'euclidean'
                elif 'dtw' in args.retrieve_suffix:
                    metric = 'dtw'
                else:
                    metric = 'euclidean'  # default
                if use_norm:
                    do_retrieve_Z_reverse_norm(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix, metric=metric)
                else:
                    do_retrieve_Z_reverse(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix, metric=metric)
            elif 'X-' in args.retrieve_suffix and 'rev' in args.retrieve_suffix:
                # Use X-space reverse retrieval
                # Extract metric from suffix (e.g., "X-cosine-rev_k3" -> "cosine")
                if 'cosine' in args.retrieve_suffix:
                    metric = 'cosine'
                elif 'euclidean' in args.retrieve_suffix:
                    metric = 'euclidean'
                elif 'dtw' in args.retrieve_suffix:
                    metric = 'dtw'
                else:
                    metric = 'euclidean'  # default
                if use_norm:
                    do_retrieve_X_reverse_norm(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix, metric=metric)
                else:
                    do_retrieve_X_reverse(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix, metric=metric)
            else:
                # Fallback to standard retrieval
                if use_norm:
                    # For standard retrieval with norm, check if it's X-space or Z-space
                    if 'X-' in args.retrieve_suffix:
                        # Extract metric
                        if 'cosine' in args.retrieve_suffix:
                            metric = 'cosine'
                        elif 'euclidean' in args.retrieve_suffix:
                            metric = 'euclidean'
                        elif 'dtw' in args.retrieve_suffix:
                            metric = 'dtw'
                        else:
                            metric = 'euclidean'
                        do_retrieve_X_norm(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, metric=metric, save=args.save, retrieve_suffix=args.retrieve_suffix)
                    else:
                        do_retrieve_Z_norm(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix)
                else:
                    do_retrieve(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix)
        else:
            # Use standard retrieval (Z or X-space)
            # Check if it's X-space or Z-space first
            if 'X-' in args.retrieve_suffix:
                # Extract metric
                if 'cosine' in args.retrieve_suffix:
                    metric = 'cosine'
                elif 'euclidean' in args.retrieve_suffix:
                    metric = 'euclidean'
                elif 'dtw' in args.retrieve_suffix:
                    metric = 'dtw'
                else:
                    metric = 'euclidean'
            # X-space: use do_retrieve_X_norm or do_retrieve_X based on normalize
                if use_norm:
                    do_retrieve_X_norm(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, metric=metric, save=args.save, retrieve_suffix=args.retrieve_suffix)
                else:
                    do_retrieve_X(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, metric=metric, save=args.save, retrieve_suffix=args.retrieve_suffix)
            else:
                # Z-space: use do_retrieve_Z_norm or do_retrieve based on normalize
                if use_norm:
                    do_retrieve_Z_norm(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix)
                else:
                    do_retrieve_Z(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning, args.retrieve_suffix)
    print('retrieved_data_path = {}'.format(retrieved_data_path))
    args.data_path = retrieved_data_path.split('/')[-1]

    # load retrieved raw data, it will be used to reconstruct the retrieved data
    retriever_rawdata = []
    
    # Determine database file suffix based on retrieval type (Z-space vs X-space)
    # Z-space uses _Z.pkl, X-space uses .pkl (or other suffixes for X-space with metric)
    db_suffix = '.pkl'
    if args.retrieve_suffix:
        # Check if it's Z-space (not X-space)
        if 'X-' not in args.retrieve_suffix:
            # Z-space: use _Z.pkl suffix
            db_suffix = '_Z.pkl'
        # X-space: use .pkl (or could be _X_cosine.pkl, etc., but raw_data is in base .pkl)
        # For X-space, raw_data is typically in the base .pkl file
    
    db_filename = f'{args.metadata["database_name"][0]}_{args.metadata["frequency"]}_{args.metadata["lookback_length"]}{db_suffix}'
    
    if args.mode == 'only_self' or args.mode == 'only_self_train':
        # database: {var1: {}, var2: {}, ...}
        # retriever_rawdata: [var1_raw_data, var2_raw_data, ...]
        db_path = os.path.join(args.retrieval_database_dir, db_filename)
        # Try to load the database, fallback to base .pkl if not found
        if not os.path.exists(db_path) and db_suffix == '_Z.pkl':
            # Fallback: try base .pkl file (X-space format)
            base_db_path = os.path.join(args.retrieval_database_dir, f'{args.metadata["database_name"][0]}_{args.metadata["frequency"]}_{args.metadata["lookback_length"]}.pkl')
            if os.path.exists(base_db_path):
                db_path = base_db_path
        database = load_database(db_path)
        for variable in database.keys():
            # Check if 'raw_data' exists (for backward compatibility)
            if 'raw_data' in database[variable]:
                retriever_rawdata.append(database[variable]['raw_data'])
            else:
                # If 'raw_data' doesn't exist, reconstruct from 'slices'
                # This happens when using X-space databases
                slices = database[variable]['slices']
                # Reconstruct raw_data from slices (assuming overlapping windows)
                # This is a simplified reconstruction - may need adjustment based on actual use case
                raw_data = slices[0].tolist()  # Start with first slice
                for i in range(1, len(slices)):
                    # Append only the last element of each subsequent slice
                    raw_data.append(slices[i][-1])
                retriever_rawdata.append(np.array(raw_data))
    elif args.mode == 'all_vars':
        for database_name in args.metadata['database_name']:
            db_path = os.path.join(args.retrieval_database_dir, db_filename)
            # Try to load the database, fallback to base .pkl if not found
            if not os.path.exists(db_path) and db_suffix == '_Z.pkl':
                # Fallback: try base .pkl file (X-space format)
                base_db_path = os.path.join(args.retrieval_database_dir, f'{database_name}_{args.metadata["frequency"]}_{args.metadata["lookback_length"]}.pkl')
                if os.path.exists(base_db_path):
                    db_path = base_db_path
            database = load_database(db_path)
            for variable in database.keys():
                # Check if 'raw_data' exists (for backward compatibility)
                if 'raw_data' in database[variable]:
                    retriever_rawdata.append(database[variable]['raw_data'])
                else:
                    # If 'raw_data' doesn't exist, reconstruct from 'slices'
                    slices = database[variable]['slices']
                    raw_data = slices[0].tolist()
                    for i in range(1, len(slices)):
                        raw_data.append(slices[i][-1])
                    retriever_rawdata.append(np.array(raw_data))

    # scale transform for retrieved data
    scaler = StandardScaler()
    retriever_rawdata = np.array(retriever_rawdata).T
    scaler.fit(retriever_rawdata)
    retriever_rawdata = scaler.transform(retriever_rawdata)         #(n_samples, n_features)
    retriever_rawdata = retriever_rawdata.T
    test_data, test_loader = data_provider(args, 'test', retriever_rawdata=retriever_rawdata)

else:
    test_data, test_loader = data_provider(args, 'test')

if args.freq != 'h':
    args.freq = SEASONALITY_MAP[test_data.freq]
    print("freq = {}".format(args.freq))
device = torch.device(device_address)

time_now = time.time()

if args.model == 'ChronosBolt':
    model = ChronosBoltPipeline.from_pretrained(args.pretrained_model_path)
    model.model.load_state_dict(torch.load(args.pretrained_model_path+'autogluon_model.pth'))
    model.model.to(device)
elif args.model == 'ChronosBoltRetrieve':
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.pretrained_model_path, config=config, augment=args.augment_mode)


    state_dict = torch.load(best_model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)

    model.to(device)
elif args.model == "MOMENTRetrieve":
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    state_dict = torch.load(best_model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
else:
    print('model error')
    exit()

print("------------------------------------")

if 'retrieve' in args.model_id:
    mse, mae = test_retrieve(model, test_data, test_loader, args, device)
else:
    mse, mae = test(model, test_data, test_loader, args, device)
mses.append(round(mse,5))
maes.append(round(mae,5))

if len(maes)==0 : exit()
maes = np.array(maes)
mses = np.array(mses)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

# shift-aware logging: if query dataset (from data_path) differs from retrieval DB, log to results_shift
query_ds_name = os.path.splitext(os.path.basename(ori_data_path))[0]
retrieval_db_name = args.metadata['database_name'][0] if isinstance(args.metadata.get('database_name'), list) else args.metadata.get('database_name')
is_shift = retrieval_db_name is not None and query_ds_name != retrieval_db_name

# Scope result directories by horizons so outputs from different runs do not collide
base_results_dir = f'results_{args.seq_len}_{args.pred_len}'
shift_results_dir = f'results_shift_{args.seq_len}_{args.pred_len}'
log_dir = f'{shift_results_dir}/forecast_evaluation' if is_shift else f'{base_results_dir}/forecast_evaluation'
os.makedirs(log_dir, exist_ok=True)
file_path = os.path.join(log_dir, log_fine_name)

with open(file_path, 'a') as f : 
    f.write("{}\n".format(args.model_id))
    f.write("mse:{:.4f}, std:{:.4f} ---- mae:{:.4f}, std:{:.4f}\n".format(np.mean(mses), np.std(mses) , np.mean(maes), np.std(maes)))
        
print(log_fine_name)
            
