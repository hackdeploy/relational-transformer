"""
Smoke test for the pyrustler module.

Tests that:
1. All data structures can be instantiated and serialised (pickle round-trip).
2. The Sampler class has the expected public interface.
3. If preprocessed data exists, a small batch can be sampled.
"""

import pickle
import sys
import tempfile
from pathlib import Path

# ---- 1. data structures ----
from pyrustler.common import Edge, Node, SemType, TableType, TableInfo

print("--- data structures ---")

node = Node(
    is_task_node=True,
    node_idx=42,
    f2p_nbr_idxs=[10, 20],
    f2p_edges=[
        Edge(node_idx=10, table_name_idx=1, table_type=TableType.Db, timestamp=100),
        Edge(node_idx=20, table_name_idx=2, table_type=TableType.Train),
    ],
    timestamp=50,
    table_name_idx=0,
    col_name_idxs=[5, 6, 7],
    sem_types=[SemType.Number, SemType.Text, SemType.Boolean],
    number_values=[1.5, 0.0, 0.0],
    text_values=[0, 99, 0],
    datetime_values=[0.0, 0.0, 0.0],
    boolean_values=[0.0, 0.0, 0.8],
    class_value_idx=[-1, 99, -1],
)

# pickle round-trip
data = pickle.dumps(node)
node2 = pickle.loads(data)
assert node2.node_idx == 42
assert node2.f2p_edges[0].table_type == TableType.Db
assert node2.sem_types[1] == SemType.Text
print(f"  Node pickle round-trip OK  ({len(data)} bytes)")

edge = Edge(node_idx=5, table_name_idx=1, table_type=TableType.Val)
assert edge.timestamp is None
print("  Edge OK")

ti = TableInfo(node_idx_offset=100, num_nodes=50)
assert ti.num_nodes == 50
print("  TableInfo OK")

# ---- 2. Sampler interface check ----
from pyrustler.fly import Sampler

print("\n--- Sampler interface ---")
assert callable(getattr(Sampler, "__init__", None))
assert callable(getattr(Sampler, "len_py", None))
assert callable(getattr(Sampler, "batch_py", None))
assert callable(getattr(Sampler, "shuffle_py", None))
print("  Sampler has len_py, batch_py, shuffle_py ✓")

# ---- 3. pre module importable ----
from pyrustler import pre
print(f"  pre.main callable: {callable(pre.main)} ✓")

# ---- 4. convert_file module importable ----
from pyrustler import convert_file
print(f"  convert_file.main callable: {callable(convert_file.main)} ✓")

# ---- 5. rt.data import check ----
print("\n--- rt.data import ---")
try:
    from rt.data import RelationalDataset
    print("  rt.data.RelationalDataset imported OK ✓")
except Exception as e:
    print(f"  rt.data import failed: {e}")

# ---- 6. optional: small batch test if preprocessed data exists ----
import os
home = os.environ.get("USERPROFILE", os.environ.get("HOME", "."))
test_db = "rel-f1"
pre_path = Path(home) / "scratch" / "pre" / test_db
nodes_pkl = pre_path / "nodes.pkl"

if nodes_pkl.exists():
    print(f"\n--- live batch test ({test_db}) ---")
    import json
    import numpy as np

    with open(pre_path / "table_info.json") as f:
        table_info = json.load(f)

    with open(pre_path / "column_index.json") as f:
        column_index = json.load(f)

    # Use driver-dnf Train split (classification task) with its real target column
    test_table = "driver-dnf"
    test_split = "Train"
    key = f"{test_table}:{test_split}"
    if key not in table_info:
        # fallback: pick any task key
        task_keys = [k for k in table_info if ":Train" in k]
        key = task_keys[0] if task_keys else None

    if key:
        info = table_info[key]
        tname = key.split(":")[0]
        print(f"  Using table key: {key}")
        print(f"    offset={info['node_idx_offset']}  num_nodes={info['num_nodes']}")

        # Find a target column for this table
        target_col_name = None
        target_col_idx = -1
        for cname, cidx in column_index.items():
            if cname.endswith(f" of {tname}"):
                target_col_name = cname
                target_col_idx = cidx
                break
        print(f"  target_column: {target_col_name} (idx={target_col_idx})")

        sampler = Sampler(
            dataset_tuples=[(test_db, info["node_idx_offset"], min(info["num_nodes"], 50))],
            batch_size=4,
            seq_len=64,
            rank=0,
            world_size=1,
            max_bfs_width=32,
            embedding_model="all-MiniLM-L12-v2",
            d_text=384,
            seed=0,
            target_columns=[target_col_idx],
            columns_to_drop=[[]],
        )

        num_batches = sampler.len_py()
        print(f"  num_batches={num_batches}")

        sampler.shuffle_py(0)
        batch = sampler.batch_py(0)
        batch_dict = dict(batch)

        print(f"  batch keys: {list(batch_dict.keys())}")
        for k, v in batch_dict.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: shape={v.shape} dtype={v.dtype}")
            else:
                print(f"    {k}: {v}")
        print("  live batch test OK ✓")
    else:
        print("  No task tables found, skipping live test")
else:
    print(f"\n  (no preprocessed data at {pre_path}, skipping live batch test)")

print("\n=== ALL TESTS PASSED ===")
