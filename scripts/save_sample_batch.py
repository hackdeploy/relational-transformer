"""
Save a sample batch to inspect the data structure and values.
"""
import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import os
import pickle
import numpy as np
import json
import ml_dtypes  # Required for bfloat16 support in NumPy
from rustler import Sampler

# Set USERPROFILE explicitly for Rust code (pixi might not pass Windows env vars)
if 'USERPROFILE' not in os.environ:
    os.environ['USERPROFILE'] = r'C:\Users\User'

print(f"USERPROFILE: {os.environ.get('USERPROFILE', 'NOT SET')}")

# Load table_info to get valid node offsets
db_name = "rel-f1"
table_name = "driver-dnf"
split = "Train"

table_info_path = f"{os.environ['USERPROFILE']}/scratch/pre/{db_name}/table_info.json"
with open(table_info_path) as f:
    table_info = json.load(f)

# Get table info
table_key = f"{table_name}:{split}"
if table_key not in table_info:
    # Try Db variant
    table_key = f"{table_name}:Db"
    
info = table_info[table_key]
node_idx_offset = info["node_idx_offset"]
num_nodes = info["num_nodes"]

print(f"Loading {table_key}: offset={node_idx_offset}, num_nodes={num_nodes}")

# Load column index
column_index_path = f"{os.environ['USERPROFILE']}/scratch/pre/{db_name}/column_index.json"
with open(column_index_path) as f:
    column_index = json.load(f)

# Get target column index
target_column_name = "did_not_finish"  # For driver-dnf task
target_key = f"{target_column_name} of {table_name}"
target_idx = column_index[target_key]

print(f"Target column: {target_key} = index {target_idx}")

# Create a sampler with your dataset
sampler = Sampler(
    dataset_tuples=[(db_name, node_idx_offset, num_nodes)],
    batch_size=4,
    seq_len=1024,
    rank=0,
    world_size=1,
    max_bfs_width=256,
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    seed=42,
    target_columns=[target_idx],
    columns_to_drop=[[]],
)

print(f"Sampler has {sampler.len_py()} batches available")
print(f"Total items: {sampler.len_py() * 4}")  # approximate

if sampler.len_py() == 0:
    print("\nError: No valid items found in the sampler.")
    print("Make sure you have preprocessed data at ~/scratch/pre/rel-f1/")
    print("Or adjust the dataset_tuples and target_columns parameters.")
    exit(1)

# Get a sample batch
batch_idx = 0
batch_data = sampler.batch_py(batch_idx)
batch_dict = dict(batch_data)

# Print batch structure
print("Batch structure:")
print("-" * 50)
for key, value in batch_dict.items():
    if isinstance(value, np.ndarray):
        print(f"{key:20s}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key:20s}: {value}")

# Save batch to pickle file
output_path = "sample_batch.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(batch_dict, f)

print(f"\nBatch saved to: {output_path}")

# Also save as numpy archive for easier inspection
np.savez("sample_batch.npz", **{k: v for k, v in batch_dict.items() if isinstance(v, np.ndarray)})

print(f"Batch arrays saved to: sample_batch.npz")

# Save batch to JSON file (convert numpy arrays to lists)
json_output_path = "sample_batch.json"
batch_dict_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in batch_dict.items()}
with open(json_output_path, "w") as f:
    json.dump(batch_dict_json, f, indent=2)
print(f"Batch saved to: {json_output_path}")

# Print some sample values
print("\nSample values from first sequence:")
print(f"node_idxs[0][:10] = {batch_dict['node_idxs'][:10]}")
print(f"is_targets[0][:10] = {batch_dict['is_targets'][:10]}")
print(f"is_padding[0][:10] = {batch_dict['is_padding'][:10]}")
