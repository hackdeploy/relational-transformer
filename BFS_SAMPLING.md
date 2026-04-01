# BFS Sampling: How Context Is Built for Each Prediction

This document explains the **BFS (Breadth-First Search) sampling algorithm** that turns a relational
database into a flat token sequence for the transformer. The implementation lives in
`pyrustler/fly.py` (`_seq` method) and its Rust equivalent `rustler/src/fly.rs`.

---

## Why BFS Sampling Exists

A relational database is a **graph**. Rows in one table point to rows in another via foreign keys.
To make a prediction about a row (e.g., "will this user churn?"), the model needs context from
related rows across multiple tables.

The transformer requires a **flat sequence of tokens**. BFS sampling converts one prediction's
relational neighbourhood into that flat sequence.

---

## Key Data Structures

### Node vs. Edge — the core distinction

- **Node** = a **row** in a table. It holds all the column values you want to feed into the model.
- **Edge** = a **pointer** from one row to another. It holds just enough metadata to decide *whether and how* to traverse to the next Node.

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  Node: orders row #101      │        │  Node: users row #42         │
│  ├── amount = 59.99         │        │  ├── name = "Alice"           │
│  ├── date   = 2024-03-01    │──────▶ │  ├── country = "US"           │
│  └── user_id = 42           │  Edge  │  └── signup = 2023-01-10      │
└─────────────────────────────┘        └──────────────────────────────┘
```

The **Edge** between them contains:

| Field | Value | Why it's needed |
|---|---|---|
| `node_idx` | 42 | *where to go* — the destination row |
| `table_name_idx` | users | *what table* — needed for the table-type filter |
| `table_type` | `Train` / `Db` | *which split* — prevents Val/Test rows leaking into training context |
| `timestamp` | 2024-03-01 | *when* — the timestamp of the **child row** (order #101), used for temporal leakage filter |

The **timestamp on the Edge** is the child row's timestamp, not the parent's. When BFS expands
from `users#42` downward to its orders, it checks each edge's timestamp: "was this order placed
before the prediction cutoff?" If not, the edge is skipped entirely.

### Full field reference

```
Node
├── node_idx          int           unique global ID across all tables in the DB
├── is_task_node      bool          True if this node is from the prediction target table
├── timestamp         Optional[int] unix ns (used for temporal leakage prevention)
├── table_name_idx    int           which table this row belongs to
├── col_name_idxs     list[int]     one entry per column in this row
├── sem_types         list[SemType] Number | Text | DateTime | Boolean (per column)
├── number_values     list[float]   z-scored numeric values
├── boolean_values    list[float]   z-scored 0/1 values
├── datetime_values   list[float]   z-scored timestamps
├── text_values       list[int]     indices into text_emb lookup table
├── f2p_nbr_idxs      list[int]     ≤5 FK parent node IDs (for feat attention mask — see below)
└── f2p_edges         list[Edge]    FK parent edges to follow during BFS

Edge
├── node_idx          int           destination node
├── table_name_idx    int           destination table
├── table_type        TableType     Db | Train | Val | Test
└── timestamp         Optional[int] child row's timestamp (for temporal leakage filter)
```

### What are `f2p_nbr_idxs`?

These are the FK parent node IDs precomputed at preprocessing time and stored directly on each
Node. A row with multiple FK columns will have multiple parent IDs here:

```
orders row #101
  ├── orders.user_id    → users row #42      ← FK parent #1
  └── orders.product_id → products row #7   ← FK parent #2

f2p_nbr_idxs for order#101 = [42, 7]
```

They are not used during BFS traversal. Instead, they are copied into the output tensor so the
model can build the **`feat` attention mask**:

```
feat[q, k] = True  if  node_idxs[q] == node_idxs[k]        ← same row
                   OR  node_idxs[q] IN f2p_nbr_idxs[k]      ← q is a FK parent of k's row
                   OR  node_idxs[k] IN f2p_nbr_idxs[q]      ← k is a FK parent of q's row
```

This allows all cells belonging to the **same row or its direct FK parents** to attend to each
other — capturing everything describing one entity at one moment. The cap of 5 (`MAX_F2P_NBRS`)
is a hard constant set at preprocess time; in practice most tables have 1–3 FK columns.

### Two adjacency structures

| Direction | Storage | Description |
|---|---|---|
| **f2p** (foreign → primary) | Stored on each `Node.f2p_edges` | FK pointer going *up*: `orders.user_id → users` |
| **p2f** (primary → foreign) | Separate `p2f_adj[node_idx]` list | FK pointer going *down*: `users → [order#1, order#2, ...]` |

---

## Pseudocode

```
function sample_sequence(seed_node, dataset, seq_len, max_bfs_width, epoch, seed):

    visited  = set()
    f2p_ftr  = stack[(depth=0, node=seed_node)]   # deterministic, depth-first
    p2f_ftr  = list_of_lists_by_depth[]            # stochastic, per-depth buckets
    cells    = []

    while len(cells) < seq_len:

        # --- 1. Pick next node ---
        if f2p_ftr is not empty:
            (depth, node) = f2p_ftr.pop()          # always drain f2p first
        else:
            depth = first depth bucket with candidates in p2f_ftr
            if none found: STOP                     # no more nodes to visit
            node = random_pop(p2f_ftr[depth])       # stochastic for p2f

        if node in visited: continue
        visited.add(node)

        # --- 2. Enqueue neighbours ---

        # f2p: follow FK *up* to parent rows (e.g. order → user)
        for edge in node.f2p_edges:
            f2p_ftr.push((depth + 1, edge.node))

        # p2f: follow FK *down* to child rows (e.g. user → [orders])
        candidates = []
        for edge in p2f_adj[node]:
            if temporal_leak(edge, seed_node): continue    # skip future edges
            if table_filter(edge, seed_node):  continue    # skip unrelated tables
            candidates.append(edge.node)

        if len(candidates) > max_bfs_width:
            candidates = random_sample(candidates, max_bfs_width)  # subsample

        for child in candidates:
            p2f_ftr[depth + 1].append(child)

        # --- 3. Emit one cell per column in this row ---
        for each column col in node:
            if col is in columns_to_drop for this seed_node: skip
            cells.append(Cell(node, col))
            if len(cells) == seq_len: STOP

    return cells   # flat list, length ≤ seq_len
```

---

## Step-by-Step Walkthrough

### Step 0 — Initialisation

```python
f2p_ftr = [(0, seed_node_idx)]   # start with the task node at depth 0
p2f_ftr = []                     # empty; will grow as we expand downward

visited  = [False] * num_nodes
seq_i    = 0                     # how many cells have been emitted so far
```

The RNG is seeded deterministically from `epoch + seed_node_idx + seed`, so the same example
produces different random subsamples each epoch (stochastic coverage).

---

### Step 1 — Pick the Next Node

Priority rule: **f2p always drains before p2f**.

```python
if f2p_ftr:
    depth, nidx = f2p_ftr.pop()        # stack pop → depth-first within f2p
else:
    # find shallowest p2f bucket that still has candidates
    depth = next(i for i, lst in enumerate(p2f_ftr) if lst)
    r     = rng.randrange(len(p2f_ftr[depth]))
    nidx  = p2f_ftr[depth].swap_remove(r)   # random pop → stochastic for p2f

if visited[nidx]: continue
```

Why two separate frontiers? FK *up* traversals (f2p) are cheap and deterministic — there is at
most one parent per FK column. FK *down* traversals (p2f) can fan out enormously (a popular user
might have 50,000 orders), so they are randomised and capped.

---

### Step 2a — Enqueue f2p Neighbours (FK up)

```python
for edge in node.f2p_edges:
    f2p_ftr.append((depth + 1, edge.node_idx))
```

These are the nodes the current row *points to* via FK — e.g., `order.user_id → users row`.
They are always enqueued without filtering (no temporal check needed; FK parents cannot be in
the future relative to the child).

---

### Step 2b — Enqueue p2f Neighbours (FK down) with filtering

Two filters are applied before enqueuing:

#### Temporal leakage filter

```python
if edge.timestamp is not None and seed_node.timestamp is not None:
    if edge.timestamp > seed_node.timestamp:
        continue   # skip: this child row exists in the future — leakage!
```

This ensures the model only sees data that would have been available *at prediction time*.

#### Table-type filter

```python
if edge.table_name_idx != seed_node.table_name_idx and edge.table_type != TableType.Db:
    continue
```

During training, the task table is split into Train / Val / Test. This filter prevents rows
from the Val or Test splits leaking into a training example's context.

#### Width cap (subsampling)

```python
if len(candidates) > max_bfs_width:   # default: 256
    candidates = rng.sample(candidates, max_bfs_width)
```

Without this cap, a single highly-connected node (e.g., a warehouse in a supply chain) could
consume the entire 1024-cell budget, leaving no room for the rest of the neighbourhood.

---

### Step 3 — Emit Cells

Once a node is visited, every column becomes one token in the sequence:

```python
for cell_i, col_idx in enumerate(node.col_name_idxs):
    if col_idx in columns_to_drop: continue    # drop target/leakage columns

    si = offset + seq_i            # flat index into the (B × seq_len) tensor

    node_idxs[si]       = node.node_idx
    table_name_idxs[si] = node.table_name_idx
    col_name_idxs[si]   = col_idx
    sem_types[si]       = node.sem_types[cell_i]
    number_values[si]   = node.number_values[cell_i]
    text_values[...]    = text_emb[node.text_values[cell_i]]
    datetime_values[si] = node.datetime_values[cell_i]
    boolean_values[si]  = node.boolean_values[cell_i]
    f2p_nbr_idxs[...]   = node.f2p_nbr_idxs   # ≤5 parent node IDs
    is_padding[si]      = False                # mark this slot as real

    seq_i += 1
    if seq_i >= seq_len: STOP
```

The `f2p_nbr_idxs` written here are not BFS-traversal data — they are precomputed during
preprocessing and stored directly on each `Node`. They are used later in the model to build
the `feat` attention mask (cells from the same row or its FK parents can attend to each other).

---

## The Output: A Flat Token Sequence

### What "flat" means

A normal table is **2D** — rows × columns. BFS visits multiple tables, each also 2D. The
transformer can only accept a **1D sequence**, so all of those cells are unrolled into a single
list, reading left-to-right across each row's columns, then moving to the next row:

```
Normal 2D view:
          age    country   signup      is_premium
user#42    34     "US"     2023-01-10    True       ← row 1
order#101  —      —        2024-03-01    —          ← row 2 (different columns)
...

After flattening → 1D token sequence:
┌────────────┬─────────────────┬─────────────────┬──────────────────┬────────────────┬──────────────┐
│user#42.age │user#42.country  │user#42.signup   │user#42.is_premium│order#101.amount│order#101.date│ ...
│  slot 0    │    slot 1       │    slot 2       │     slot 3       │    slot 4      │   slot 5     │
└────────────┴─────────────────┴─────────────────┴──────────────────┴────────────────┴──────────────┘
```

Each slot is **one cell** — one column of one row. Not a row, not a column — a single value.
A subgraph with 10 rows × 5 columns = 50 tokens.

The relational structure (which cells came from the same row, which rows are FK-related) is
**not** encoded in position. It is encoded in the attention masks built afterward from
`node_idxs`, `col_name_idxs`, `f2p_nbr_idxs` etc. This is why the model has no positional
encodings — position in the sequence is meaningless; structure comes from the masks.

---

After `_seq` returns, the flat arrays look like this (example with seq_len=12, 3 rows of 4 cols each):

```
Slot  0: user#42.age          node_idx=42, col=age,      sem=Number,   is_padding=False
Slot  1: user#42.country      node_idx=42, col=country,  sem=Text,     is_padding=False
Slot  2: user#42.signup_date  node_idx=42, col=signup,   sem=DateTime, is_padding=False
Slot  3: user#42.is_premium   node_idx=42, col=premium,  sem=Boolean,  is_padding=False, is_target=True
Slot  4: order#101.amount     node_idx=101, ...
Slot  5: order#101.date       ...
...
Slot 11: (last real cell)
```

Zero-padding slots (if the subgraph had fewer than seq_len real cells):

```
Slot 12–1023: is_padding=True, all values=0
```

---

## How the Sequence Becomes Attention Masks

The flat arrays are reshaped to `(B, seq_len)` and passed to `rt/model.py`, which builds four
boolean masks of shape `(B, seq_len, seq_len)`:

| Mask | Allows attention between… | Built from |
|---|---|---|
| `feat` | cells in the same row *or* its f2p parents | `node_idxs` + `f2p_nbr_idxs` |
| `nbr` | cells in f2p-neighbour rows (reverse FK) | `node_idxs` + `f2p_nbr_idxs` |
| `col` | cells in the same column across all rows | `col_name_idxs` + `table_name_idxs` |
| `full` | all real cells (global attention) | always True (within `pad`) |

All four masks are AND-ed with `pad`, which is:

```python
# True only where BOTH query and key positions are real (non-padding)
pad[b, q, kv] = (not is_padding[b, q]) AND (not is_padding[b, kv])
```

This means padding positions are silently blocked from all four attention heads simultaneously.

---

## Parameter Reference

| Parameter | Default | Effect |
|---|---|---|
| `seq_len` | 1024 | Total cells per example. Increase = more context, quadratic memory cost. |
| `max_bfs_width` | 256 | Max p2f children per node. Reduce to control fanout from wide tables. |
| `MAX_F2P_NBRS` | 5 | Max f2p parent IDs stored per cell (hard constant, set at preprocess time). |
| `batch_size` | 32 | Number of subgraphs stacked. Memory ∝ `batch_size × seq_len²`. |
| `seed` | — | Mixed with `epoch + node_idx` for reproducible but epoch-varying subsamples. |

---

## Common Questions

**Q: Is BFS guaranteed to visit the closest nodes first?**
Yes — f2p is depth-ordered (stack) and p2f uses depth-indexed buckets. Nodes at depth 0 are
always emitted before depth 1 etc. When `seq_len` is exhausted, truncation removes the
*deepest* nodes, keeping the task node and its closest context.

**Q: What if the same node is reachable via two paths?**
The `visited` array prevents double-processing. The first path wins.

**Q: Does the model know which row each cell came from?**
Yes — via `node_idxs` and `f2p_nbr_idxs`. These are used to build the `feat` and `nbr` masks.
There are no positional encodings; row identity is conveyed entirely through the structural masks.

**Q: How does the model see the full graph if only 1024 cells fit?**
It doesn't in a single forward pass. Over many training epochs, different random subsets of
wide neighbourhoods are sampled (controlled by `epoch` in the RNG seed), so the model learns
from the full graph distribution stochastically.

**Q: What is `columns_to_drop`?**
It's the leakage column list — columns on the task node that would reveal the answer or are
recorded simultaneously (e.g., the target column itself, or a "churn date" column that only
exists once churn has happened). These are specified per task in `rt/tasks.py`.
