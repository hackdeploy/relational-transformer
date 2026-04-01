# Relational Transformer — Architecture Deep Dive

> **Assumed background:** You know how standard transformers (BERT, GPT) work — attention, FFN, residual connections. This doc explains what's *new* here.

---

## 1. The Core Problem

Standard transformers work on sequences (text, tokens). Relational databases aren't sequences — they're **graphs** of tables joined by foreign keys. A single prediction task (e.g. "will this Amazon user churn?") requires reasoning over:

- Multiple rows across multiple tables
- Columns with very different data types (numbers, strings, dates, booleans)
- Join relationships (foreign → primary key links)

The Relational Transformer turns a slice of a relational database into a flat sequence of **cells** (one token per cell), then applies a specially masked transformer over them.

---

## 2. Data Representation

### The Sequence = A Bag of Cells

Each token in the sequence is a **single cell** from a table row. If a row has 10 columns, it contributes 10 tokens. The sequence is built by BFS-sampling the relational graph starting from the "task node" (the row you're predicting on).

```
Task row (e.g. user #42)
  └── user.age, user.country, user.signup_date   ← 3 tokens
      └── order #101 (FK from orders.user_id)
            └── order.amount, order.date          ← 2 tokens
                └── product #7 (FK from orders.product_id)
                      └── product.name, product.price  ← 2 tokens
```

### How BFS Sampling Works

The BFS traversal starts at the task node and expands the relational graph in two directions:

- **Foreign → Primary (f2p)**: Follow FK links upward — e.g. from `order.user_id` up to the `users` row. These are traversed deterministically (depth-first stack).
- **Primary → Foreign (p2f)**: Follow FK links downward — e.g. from the `users` row down to all their `orders`. These are sampled randomly when there are more than `max_bfs_width` candidates.

Two important constraints are enforced during sampling:
1. **Temporal correctness**: edges with a timestamp *after* the task node's timestamp are excluded — no future leakage.
2. **Width cap**: if a node has more than `max_bfs_width=256` child edges, they are **randomly subsampled** to 256. This is the key control for wide graphs (e.g., a user with 50,000 orders).

The result is a flat list of rows, ordered by BFS distance from the task node. Each row emits one token per column. The sequence is filled until `seq_len` cells are reached, then truncated — closest nodes are always kept.

**Sequence length** is the total number of cells the transformer sees per training example — not per row, but across the whole sampled subgraph:

```
seq_len = 1024  →  at most 1024 cells total per example

user #42 row      (5 cols)  →   5 cells     ← always included (task node)
order #101        (4 cols)  →   4 cells
order #102        (4 cols)  →   4 cells
product #7        (6 cols)  →   6 cells
...BFS continues until 1024 cells are filled; remaining rows dropped
```

Default is **1024 cells**. Over many training steps, different random subsets of wide neighborhoods are sampled — the model learns from the full graph stochastically, not all at once.

### Subgraphs vs. Batches

A **subgraph** is the context for one prediction — one task node and its BFS neighborhood, flattened into up to 1024 cells. A **batch** is `batch_size` independent subgraphs stacked together.

```
One subgraph  =  context for ONE task node (one prediction)
                 └── up to 1024 cells, shape: (1024,)

One batch     =  batch_size=32 subgraphs stacked
                 └── tensor shape: (32, 1024)
```

For a single forward pass with `batch_size=32`:
- 32 different task nodes are sampled (e.g. user #42, user #99, user #301 ...)
- Each gets its own independent BFS traversal and its own 1024-cell sequence
- The 32 sequences are padded to the same length and stacked into `(32, 1024)`
- Cross-batch isolation is **inherent** — every tensor is shaped `(B, S, S)`, so `b=0` and `b=1` are completely separate array slices. There is no mechanism by which a token in one subgraph can reach a token in another.

**What the `pad` mask actually does** is handle *variable-length subgraphs*. A small subgraph (e.g. 200 real cells out of 1024) is zero-padded to fill the full sequence length. The sampler initializes `is_padding = True` for all positions, then sets it `False` only as BFS fills real cells in:

```python
# Shape (B, S, S): True where BOTH q and kv are real (non-padding) cells
pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
```

This prevents three unwanted attention patterns within a single example:
- Real cells attending **to** padding positions
- Padding positions attending **to** real cells  
- Padding positions attending to each other

All 4 structural masks (`feat`, `nbr`, `col`, `full`) are AND-ed with `pad` at construction time, so padding is blocked from every attention head simultaneously.

This means:
- **Loss** is computed per-cell within each subgraph, then averaged across the batch
- **Padding** is added when a subgraph has fewer than 1024 cells (small neighborhoods) — those positions are masked out everywhere
- **Batch size and seq_len are independent knobs** — you can have 32 examples of 1024 cells, or 64 examples of 512 cells. Memory scales with `batch_size × seq_len²` due to the `(B, S, S)` attention masks.

### 5 Semantic Types

Every token has a **semantic type** that determines how it's encoded:

| Type | Raw input | Preprocessing | Encoder |
|---|---|---|---|
| `number` | float scalar | **z-score** per column: `(x − μ) / σ` (computed over training split) | `Linear(1 → d_model)` |
| `datetime` | unix timestamp (nanoseconds) | **z-score** globally across all datetime columns | `Linear(1 → d_model)` |
| `boolean` | 0.0 / 1.0 | **z-score** per column (mean/std of the 0/1 distribution) | `Linear(1 → d_model)` |
| `text` | raw string | embedded with `sentence-transformers/all-MiniLM-L12-v2` → 384-dim | `Linear(384 → d_model)` |
| `col_name` | column name string | same sentence-transformer embedding | `Linear(384 → d_model)` |

> **Where normalization happens:** in the Rust/Python preprocessor (`pre.rs` / `pre.py`), not in the model. Stats (mean, std) are computed over the **training split** of each column and stored in the preprocessed binary files. `std=0` columns get `std=1` to avoid division by zero.

Every token's embedding = **col_name embedding + value embedding**. Masked tokens (the target) replace the value with a learned `mask_emb` vector.

---

## 3. The 4-Head Attention Pattern (The Novel Part)

This is the key architectural innovation. Instead of one attention pattern, each transformer block has **4 separate attention modules**, each with its own learned weights and a **different structural mask**:

```
Input x
  │
  ├─ feat_attn(x, feat_mask)   ─── same row OR foreign-key neighbors
  ├─ nbr_attn(x, nbr_mask)    ─── cells that point TO this row
  ├─ col_attn(x, col_mask)    ─── same column name across any row
  └─ full_attn(x, full_mask)  ─── all non-padding tokens
  │
  └─ FFN(x)
  │
Output x
```

All 4 outputs are summed into `x` with residual connections (pre-norm with RMSNorm).

### The 4 Masks Defined Precisely

Let `q` and `kv` be two tokens in the batch. The mask `M[q, kv] = True` if attention is allowed:

| Head | Mask condition |
|---|---|
| **`feat`** | `same_node(q,kv)` OR `kv` is among `q`'s foreign→primary neighbors |
| **`nbr`** | `q` is among `kv`'s foreign→primary neighbors (reverse of feat) |
| **`col`** | same column name AND same table name |
| **`full`** | any non-padding token |

From the code (`model.py`):
```python
attn_masks = {
    "feat": (same_node | kv_in_f2p) & pad,   # intra-row + FK lookup
    "nbr":  q_in_f2p & pad,                  # reverse FK
    "col":  same_col_table & pad,             # same column, any row
    "full": pad,                              # global
}
```

### Why This Design?

Each head captures a different **relational inductive bias**:

- **`feat`**: "What values are in the same record or directly joined to me?" → row-level context
- **`nbr`**: "What other records reference me?" → aggregate-like reasoning (e.g., all orders for a user)
- **`col`**: "What does this column look like across all sampled rows?" → column-level statistics
- **`full`**: Global context, lets signals propagate across the whole sequence

This replaces the need for explicit feature engineering (GROUP BY, COUNT(*), AVG(price)) — the model learns to do it.

---

## 4. Full Model Architecture

```
Input batch (B × S tokens)
│
├─ Encoders: for each token, project its value to d_model
│     number/datetime/boolean: Linear(1, d_model)
│     text: Linear(d_text, d_model)        (d_text=384)
│     col_name: Linear(d_text, d_model)    (always added)
│
├─ 12× RelationalBlock:
│     ├─ RMSNorm + feat_attn  (residual)
│     ├─ RMSNorm + nbr_attn   (residual)
│     ├─ RMSNorm + col_attn   (residual)
│     ├─ RMSNorm + full_attn  (residual)
│     └─ RMSNorm + SwiGLU FFN (residual)
│
├─ RMSNorm (final)
│
└─ Decoders: per semantic type, project d_model → output
      number/datetime: Linear(d_model, 1)    → Huber loss
      boolean:         Linear(d_model, 1)    → BCE loss
      text:            Linear(d_model, d_text) (not trained, text masking unsupported)
```

**Default hyperparameters** (from `scripts/example_pretrain.py`):

| Parameter | Value |
|---|---|
| `num_blocks` | 12 |
| `d_model` | 256 |
| `num_heads` | 8 |
| `d_ff` | 1024 |
| `d_text` | 384 |
| `seq_len` | 1024 |
| Precision | bfloat16 |

Total parameters: ~few million (small by LLM standards, but data-efficient).

### FFN: SwiGLU (same as LLaMA)

```python
def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

A standard FFN computes `w2(relu(w1(x)))` — one gate, one projection. SwiGLU uses **two parallel projections** (`w1`, `w3`) multiplied together, with SiLU (a smooth ReLU) as the activation on one of them. The product acts as a learned gate: `w3(x)` decides *how much* of `w1(x)` to pass through. `w2` then projects back to `d_model`. This gives the FFN more expressive power without adding much cost, and is used in LLaMA, PaLM, and other modern transformers.

---

## 5. Pretraining Objective: Masked Cell Modeling

Analogous to BERT's masked language modeling, but for cells:

1. Randomly **mask a subset of cells** (replace value with `mask_emb`)
2. Predict the original value from context
3. Loss depends on type:
   - **Number / datetime** → Huber loss
   - **Boolean** → Binary cross-entropy
   - **Text** → skipped (too expensive to reconstruct embeddings)

During fine-tuning, the **target column** is always masked, and the loss is computed only on those cells.

---

## 6. Training Pipeline

```
Pretrain (6 databases, 1 held-out)
    ↓  ~2h on 8×A100
Continued pretrain (held-out DB, hold out test task)
    ↓  ~15min
Fine-tune (single task)
    ↓  ~1.5h
Evaluate (val/test splits, AUC or R²)
```

**Optimizer:** AdamW (`lr=1e-3`, `wd=0.1`, fused CUDA kernel)  
**Scheduler:** OneCycleLR with 20% warmup, linear decay  
**Multi-GPU:** PyTorch DDP via `torchrun --nproc_per_node=8`  
**Compilation:** `torch.compile()` + `flex_attention` compiled separately

---

## 7. The 7 Benchmark Databases (Relbench)

| DB | Domain | Example task |
|---|---|---|
| `rel-amazon` | E-commerce | user churn, LTV |
| `rel-hm` | Fashion retail | item sales |
| `rel-stack` | Q&A platform | post votes, badges |
| `rel-avito` | Classifieds | ad CTR |
| `rel-event` | Events | user attendance |
| `rel-trial` | Clinical trials | study outcome |
| `rel-f1` | Formula 1 racing | driver DNF, position |

Tasks are split into:
- **Forecast**: predict future behavior from historical data (temporal split)
- **Autocomplete**: predict a masked column value from other columns

---

## 8. Attention Implementation Details

Masking is implemented with **PyTorch `flex_attention`** (sparse, custom mask support) rather than additive masking. This is more efficient for the highly structured sparsity patterns here.

- `full` head (dense): uses Flash Attention (`sdpa_kernel(FLASH_ATTENTION)`)
- All other heads: use `flex_attention` with `create_block_mask`

Foreign→primary neighbor indices (`f2p_nbr_idxs`) are precomputed during the Rust preprocessing step. Up to **5 FK neighbors** per cell are stored (`MAX_F2P_NBRS = 5`).

---

## 9. How It Differs from Prior Work

| Approach | How it handles relational data |
|---|---|
| GNN-based (e.g., RGCN) | Aggregates node features on a graph |
| Feature engineering + XGBoost | Manual GROUP BY, COUNT, AVG |
| **Relational Transformer** | Sequence of cells with structural attention masks; learns aggregations |

The key claim of the paper: **zero-shot transfer** across databases. A model pretrained on 6 databases can make reasonable predictions on the 7th without fine-tuning, because the column-level and neighbor-level attention heads learn *generic* relational reasoning patterns.

---

## 10. Practical Assessment

### Where this model shines

**Cold start on a new database.**
You have a new relational DB and need predictions *today*, before you've collected enough labels to train a supervised model. Zero-shot RT gives you ~93% of fully supervised AUROC with a single forward pass. This is the model's single strongest use case.

**Low-label regime.**
When you do start collecting labels, fine-tuning RT needs far fewer examples than training a GNN or gradient-boosted model from scratch. The pretrained weights already encode general relational reasoning.

**No feature engineering team.**
RT replaces weeks of manual SQL aggregations (GROUP BY customer_id, COUNT orders, AVG spend, ...). The `col` and `nbr` attention heads learn to do this automatically from raw FK structure.

**Many tasks on the same database.**
One pretrained model can be fine-tuned for churn, LTV, fraud, upsell — all on the same DB — without rerunning preprocessing or re-engineering features per task.

**Boolean and numeric targets across structured schemas.**
If your prediction targets are binary outcomes (churn/no-churn, fraud/no-fraud) or numeric quantities (revenue, count), and your data lives in a normalized relational schema, this is a strong fit.

---

### Where it won't replace your current stack

**Mature pipelines with abundant labels and domain experts.**
XGBoost with carefully engineered features trained on millions of labeled rows is extremely hard to beat. When you have data and expertise, RT's zero-shot advantage disappears and the infrastructure overhead becomes a liability.

**Predicting text/string columns.**
"Classification" in this model means predicting a *boolean* column (0 or 1). Predicting what a string value will be (e.g., a product category, a status message) is **not supported** — text columns can only be used as input features, never as targets.

**Deep graph dependencies.**
If the predictive signal sits 6+ FK hops away from the target row, BFS truncation at 1024 cells may cut it off entirely. There's no guarantee the most important context fits within the sequence length.

**Low-latency inference.**
Running a 22M-parameter transformer over 1024 cells per prediction is heavier than a decision tree or a lookup table. Not suitable for sub-millisecond serving without batching or distillation.

**Highly irregular or dirty schemas.**
The 7 Relbench benchmark databases are well-curated. Production schemas with inconsistent column naming, schema drift over time, deeply nested JSON blobs, or sparse FK coverage may hurt both preprocessing and transfer quality.

---

### What to be cautious about

**"93% zero-shot" is an average.**
Individual task performance varies. Some tasks may be at 75%, others at 99%. Always evaluate on your specific task before relying on the aggregate number.

**The FK neighbor cap of 5 is a hard heuristic.**
Each cell stores at most 5 foreign→primary neighbors (`MAX_F2P_NBRS = 5`). High-cardinality joins (a user with 10,000 orders) are subsampled. The model may miss important aggregation signals in very wide relationships.

**Pretraining domain matters.**
The pretrained checkpoints were trained on 7 specific domains (e-commerce, F1, clinical trials, etc.). Zero-shot transfer degrades if your domain is structurally very different from all of them. Continued pretraining on your own DB is strongly recommended before fine-tuning.

**Infrastructure is non-trivial.**
This is not `pip install` → `model.predict()`. It requires: Rust preprocessing, sentence-transformer embedding generation, GPU training (8×A100 for pretraining), and W&B for monitoring. Budget for setup time.

**Benchmark ≠ production distribution.**
Relbench uses clean temporal splits. Real production data has concept drift, missing FK links, delayed labels, and schema changes that the benchmark doesn't test for.

---

### Supported use case summary

| Use case | Fit |
|---|---|
| Churn prediction on a new product | ✅ Strong — zero-shot or quick fine-tune |
| Fraud detection (boolean target) | ✅ Strong — boolean classification |
| Revenue / LTV forecasting | ✅ Good — numeric regression |
| Autocomplete of missing boolean/numeric fields | ✅ Good — autocomplete tasks |
| Cold start recommendations (CTR prediction) | ✅ Reasonable — rel-avito shows this |
| Predicting free-text fields | ❌ Not supported |
| Sub-10ms latency serving | ❌ Too heavy without optimization |
| Schemas without FK relationships | ❌ No relational structure to exploit |
| Tasks with >1024-cell relevant context | ⚠️ Risk of truncating key signals |
| High-label mature pipelines | ⚠️ Probably not worth the infra cost |

---

## 11. Code Entry Points

| What you want | Where to look |
|---|---|
| Model definition | `rt/model.py` — `RelationalTransformer`, `RelationalBlock` |
| Attention masks | `rt/model.py` — `forward()` of `RelationalTransformer` |
| Batch format | `rt/data.py` — `RelationalDataset.__getitem__()` |
| Training loop | `rt/main.py` — `main()` |
| BFS sampler (Python) | `pyrustler/fly.py` — `Sampler` |
| BFS sampler (Rust) | `rustler/src/fly.rs` |
| Preprocessing | `rustler/src/pre.rs` or `pyrustler/pre.py` |
| Task definitions | `rt/tasks.py` |
| Example scripts | `scripts/example_pretrain.py`, `example_finetune.py` |
