# pysqlrustler

Preprocesses Postgres tables into the relational-transformer node graph format (same output as `pyrustler`).

## Setup

**Local (Windows)**
```powershell
uv venv .venv --python 3.12
uv pip install -e ".[sql]"
.\.venv\Scripts\Activate.ps1
```

**Google Colab**
```python
import os
os.environ['HOME']        = "/content/drive/MyDrive/Colab_Data"
os.environ['USERPROFILE'] = "/content/drive/MyDrive/Colab_Data"

!pip install uv -q
!git clone --branch dev https://github.com/hackdeploy/relational-transformer.git /content/relational-transformer
!uv pip install --system -e "/content/relational-transformer[sql]"
!uv pip install --system sentence-transformers ml_dtypes orjson strictfire -q
```

## Schema config

Create a `schema.json` describing your tables:

```json
{
  "db_name": "mydb",
  "tables": [
    {
      "table_name":  "users",
      "table_type":  "Db",
      "primary_key": "user_id",
      "foreign_keys": {},
      "time_col": null,
      "sql": null
    },
    {
      "table_name":  "orders",
      "table_type":  "Db",
      "primary_key": "order_id",
      "foreign_keys": {"user_id": "users"},
      "time_col": "created_at",
      "sql": null
    },
    {
      "table_name":  "train_labels",
      "table_type":  "Train",
      "primary_key": null,
      "foreign_keys": {"user_id": "users"},
      "time_col": null,
      "sql": "SELECT user_id, label FROM labels WHERE split = 'train'"
    }
  ]
}
```

- `table_type`: `Db` | `Train` | `Val` | `Test`
- `sql`: optional — defaults to `SELECT * FROM <table_name>`. Use it to filter rows, cast unsupported types (`NUMERIC`, etc.), or rename columns.

## Full pipeline

### Step 1 — Preprocess

```powershell
python -m pysqlrustler.pre `
    --dsn "postgresql://user:pass@host:5432/mydb" `
    --config pysqlrustler/schema.json `
    --out-dir "$HOME/scratch/pre/mydb"
```

### Step 2 — Embed text tokens

```powershell
python -m rt.embed mydb
```

Reads `text.json`, runs `all-MiniLM-L12-v2` on every unique string, saves `text_emb_all-MiniLM-L12-v2.bin`.

### Step 3 — Train

```powershell
$env:WANDB_MODE="disabled"
python scripts/train_motolote.py
```

See `scripts/train_motolote.py` for hyperparameters. Checkpoints saved to `ckpts/`.

### Step 4 — Predict

```python
from pysqlrustler.predict import predict

results = predict(
    dsn="postgresql://...",
    ckpt_path="$HOME/scratch/ckpts/motolote_best.pt",
    base_schema_path="pysqlrustler/schema.json",
    task_table="listing_model_matches",
    target_column="label",
    foreign_keys={"listing_id": "listings", "model_id": "models"},
    prediction_sql="""
        SELECT l.id AS listing_id, m.id AS model_id, false::boolean AS label
        FROM listings l CROSS JOIN models m
        WHERE l.id IN ('uuid-1', 'uuid-2')
    """,
    group_by="listing_id",
    id_columns=["listing_id", "model_id"],
    top_k=3,
    out_path="predictions.csv",
)
```

Or use the task-specific wrapper:

```powershell
python scripts/predict_motolote.py `
    --listing-ids "uuid-1,uuid-2" `
    --top-k 3 `
    --out predictions.csv
```

**How prediction works:**
1. Injects your candidate SQL as a temporary `Test` split alongside the Db tables
2. Runs `pysqlrustler.pre` in a temp directory — re-fetches Db tables so FK graph edges can be built for the new candidates. Reuses the existing `text_emb.bin` (no re-embedding).
3. Loads the checkpoint, scores every candidate pair, ranks by probability
4. Returns top-k per `group_by` column

**Adding a new task** only requires a new thin wrapper — the inference engine in `pysqlrustler/predict.py` is reused as-is.

## Output files

Written to `--out-dir`:

| File | Contents |
|---|---|
| `nodes.pkl` | `list[Node]` — one node per table row |
| `p2f_adj.pkl` | Parent→children adjacency `list[list[Edge]]` |
| `text.json` | Flat list of all unique string tokens |
| `text_map.json` | Token → index mapping |
| `column_index.json` | `"col of table"` → index mapping |
| `table_info.json` | Per-table node offset and count |
| `text_emb_<model>.bin` | bfloat16 text embeddings (after `rt.embed`) |
