# pyrustler

A pure-Python port of the Rust [`rustler`](../rustler) crate.  
It provides the same preprocessing pipeline and on-the-fly context sampler used by `rt.data.RelationalDataset`, with no Rust compiler or Maturin build step required.

---

## Overview

The relational transformer operates on **graph-structured tabular data** derived from relational databases.  
`pyrustler` handles two jobs:

1. **Preprocessing** (`pre.py`) — reads raw relbench parquet tables, builds a node graph, and serialises it to disk.
2. **Sampling** (`fly.py`) — at training time, loads the preprocessed graph and samples fixed-length sequences of cells via BFS for each mini-batch.

```
relbench parquets
       │
       ▼
  pyrustler.pre            ← run once per dataset
       │
       ├── nodes.pkl        ← list[Node], one per database row
       ├── p2f_adj.pkl      ← list[list[Edge]], parent→foreign adjacency
       ├── text.json        ← string vocabulary (table/column/cell names)
       ├── text_map.json    ← reverse: string → index
       ├── column_index.json← column display-name → vocabulary index
       └── table_info.json  ← per-table {node_idx_offset, num_nodes}
                                   (also consumed by rt.data)
       │
       ▼
  pyrustler.fly (Sampler)   ← called every training step
       │
       └── batch_py(i) → list of named numpy arrays → RelationalDataset → torch tensors
```

---

## Data structures (`common.py`)

Every row in every table becomes a **`Node`**:

```python
@dataclass
class Node:
    is_task_node: bool          # True for train/val/test split rows
    node_idx: int               # global row index across all tables
    f2p_nbr_idxs: list[int]    # foreign→primary neighbour node indices
    f2p_edges: list[Edge]       # typed edges to parent nodes
    timestamp: Optional[int]    # Unix seconds (from the table's time column)
    table_name_idx: int         # index into the string vocabulary
    col_name_idxs: list[int]    # one entry per non-PK, non-FK cell
    sem_types: list[SemType]    # Number | Text | DateTime | Boolean
    number_values: list[float]  # z-scored numeric value (or 0.0)
    text_values: list[int]      # vocabulary index of the string value
    datetime_values: list[float]# globally z-scored datetime (nanoseconds)
    boolean_values: list[float] # z-scored boolean (0.0 / 1.0)
    class_value_idx: list[int]  # vocab index for classification targets (-1 otherwise)
```

Foreign-key relationships are encoded as directed **`Edge`** objects:

```python
@dataclass
class Edge:
    node_idx: int           # the other end of the edge
    table_name_idx: int     # vocabulary index of that table's name
    table_type: TableType   # Db | Train | Val | Test
    timestamp: Optional[int]
```

Cell semantic types:

| `SemType` | Source dtype | Stored in |
|---|---|---|
| `Number`   | int / float       | `number_values` |
| `Text`     | string            | `text_values` (vocab index) |
| `DateTime` | datetime (ns)     | `datetime_values` |
| `Boolean`  | bool              | `boolean_values` |

---

## Preprocessing (`pre.py`)

Run once per dataset:

```bash
python -m pyrustler.pre rel-f1
python -m pyrustler.pre rel-amazon --skip-db   # skip Db-type tables
```

### What it does

**Step 1 — Read tables.**  
All `.parquet` files under `~/scratch/relbench/<db_name>/db/` (database tables) and `tasks/*/` (task split tables) are loaded with Polars.  
Dataset-specific fixups are applied (binarising string columns, casting booleans, dropping/renaming columns) to match the exact transformations in the original Rust code.

**Step 2 — Column statistics.**  
For `Number` and `Boolean` columns the (mean, std) are computed from the training split and used to z-score values at node-build time.  
For `DateTime` columns a single global (mean, std) is computed across all datetime values in all tables. Val/Test splits reuse the Train split's statistics.

**Step 3 — Build the node vector and adjacency list.**  
Tables are iterated column by column.  
- **Primary-key columns** are skipped (they carry no semantic content).  
- **Foreign-key columns** produce `Edge` objects in both directions:
  - `f2p_edge` on the child node → the parent row.
  - `p2f_edge` on the parent → the child row (stored in `p2f_adj`).
- **Value columns** contribute one cell per non-null value, normalised and tagged with a `SemType`.

All string values (table names, column names, cell text) are interned into a shared vocabulary (`text_to_idx`).

**Step 4 — Write outputs.**  
- `nodes.pkl` / `p2f_adj.pkl` — pickle files consumed by the sampler.  
- `text.json` / `text_map.json` — string vocabulary.  
- `column_index.json` — maps `"<col> of <table>"` → vocabulary index (used by `rt.data.get_column_index`).  
- `table_info.json` — `{node_idx_offset, num_nodes}` per table key (used by `rt.data.RelationalDataset`).

---

## Sampling (`fly.py`)

The `Sampler` class is used by `rt.data.RelationalDataset` and exposes the same interface as the original Rust PyO3 extension:

```python
sampler = Sampler(
    dataset_tuples   = [("rel-f1", node_idx_offset, num_nodes)],
    batch_size       = 32,
    seq_len          = 1024,
    rank             = 0,           # DDP rank
    world_size       = 1,
    max_bfs_width    = 256,
    embedding_model  = "all-MiniLM-L12-v2",
    d_text           = 384,
    seed             = 0,
    target_columns   = [target_col_idx],
    columns_to_drop  = [[drop_col_idx, ...]],
)

sampler.shuffle_py(epoch)           # re-shuffle item order for this epoch
n = sampler.len_py()                # number of batches
batch = sampler.batch_py(batch_idx) # list[(name, np.ndarray)]
```

### How a batch is built

Each batch item is one *task node* — a row from a train/val/test split table.  
For each task node a **sequence of cells** is assembled by BFS over the graph:

```
seed node (task row)
  │
  ├── f2p edges → parent rows (same depth + 1)
  │       └── their f2p edges → grandparents …
  └── p2f edges → child rows (sibling task rows only if same table)
          └── Db-type p2f edges → related Db rows (subsampled to max_bfs_width)
```

BFS respects two constraints:
- **Temporal**: edges whose timestamp is later than the seed node's timestamp are skipped (prevents label leakage).
- **Task boundary**: p2f edges to task tables are only followed if the edge points back to the seed node's own table.

Each visited node contributes one sequence position per cell.  
Cells belonging to `columns_to_drop` on the seed node (or any node with the same timestamp) are excluded — this is how the target column is masked during training.

Once `seq_len` positions are filled (or the reachable subgraph is exhausted), the function returns.  
Remaining positions stay as padding (`is_padding=True`).

### Output arrays

`batch_py` returns a list of `(name, array)` pairs that `RelationalDataset.__getitem__` converts to named torch tensors:

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `node_idxs` | `[B·L]` | int32 | Global node index for each position |
| `f2p_nbr_idxs` | `[B·L·5]` | int32 | Up to 5 foreign→primary neighbour indices |
| `table_name_idxs` | `[B·L]` | int32 | Vocabulary index of the node's table |
| `col_name_idxs` | `[B·L]` | int32 | Vocabulary index of the column |
| `class_value_idxs` | `[B·L]` | int32 | Vocab index for Text cells (-1 otherwise) |
| `col_name_values` | `[B·L·D]` | bfloat16 | Text embedding of the column name |
| `sem_types` | `[B·L]` | int32 | 0=Number, 1=Text, 2=DateTime, 3=Boolean |
| `number_values` | `[B·L]` | bfloat16 | Z-scored numeric value |
| `text_values` | `[B·L·D]` | bfloat16 | Text embedding of the cell string value |
| `datetime_values` | `[B·L]` | bfloat16 | Z-scored datetime |
| `boolean_values` | `[B·L]` | bfloat16 | Z-scored boolean |
| `masks` | `[B·L]` | bool | True on the target cell (same as `is_targets`) |
| `is_targets` | `[B·L]` | bool | True on the prediction target cell |
| `is_task_nodes` | `[B·L]` | bool | True for task-table nodes or the target column |
| `is_padding` | `[B·L]` | bool | True for unfilled positions |
| `true_batch_size` | scalar | int | Actual samples in this batch (may be < B at end of epoch) |

`B` = `batch_size`, `L` = `seq_len`, `D` = `d_text` (embedding dimension).

Text embeddings are read directly from the pre-computed `text_emb_<model>.bin` file (raw bfloat16, array shape `[vocab_size, D]`), which is shared with the Rust version and produced by `rt.embed`.

---

## Utility: `convert_file.py`

Converts pickle files to JSON for inspection:

```bash
python -m pyrustler.convert_file ~/scratch/pre/rel-f1          # convert all .pkl files
python -m pyrustler.convert_file ~/scratch/pre/rel-f1/nodes.pkl
```

---

## Differences from the Rust version

| Aspect | Rust (`rustler`) | Python (`pyrustler`) |
|---|---|---|
| Node storage | `rkyv` zero-copy archives | Python `pickle` |
| Adjacency storage | `rkyv` archives | Python `pickle` |
| Random number generator | ChaCha12 (`rand` crate) | Python `random.Random` (Mersenne Twister) |
| Batching speed | ~5–20× faster | Adequate for CPU-bound experiments |
| Build requirement | Rust + Maturin | None (pure Python) |
| Shared files | `text.json`, `text_emb_*.bin`, `table_info.json`, `column_index.json` | same format |

Because different RNGs are used, batch orderings will differ from the Rust sampler even with the same seed.  
All output tensor shapes and dtypes are identical.
