# pysqlrustler

Preprocesses Postgres tables into the relational-transformer node graph format (same output as `pyrustler`).

## Setup

**Local (Windows)**
```powershell
uv venv .venv --python 3.12
uv pip install -e ".[sql]"
```

**Google Colab**
```python
!pip install uv -q
# clone repo or mount Drive, then:
!uv pip install --system -e "/content/relational-transformer[sql]"
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

## Run

```powershell
 .\.venv\Scripts\Activate.ps1    

python -m pysqlrustler.pre `
    --dsn "postgresql://postgres.zetsjztchfmihqfzobok:M%40satepe123@aws-1-us-east-1.pooler.supabase.com:5432/postgres" `
    --config ./pysqlrustler/schema.json `
    --out-dir ~/scratch/pre/mydb
```

Or from Python:

```python
from pysqlrustler.pre import main

main(
    dsn="postgresql://user:pass@host:5432/mydb",
    config_path="schema.json",
    out_dir="~/scratch/pre/mydb",  # optional, defaults to ~/scratch/pre/<db_name>
)
```

## Output

Written to `--out-dir`:

| File | Contents |
|---|---|
| `nodes.pkl` | `list[Node]` — one node per table row |
| `p2f_adj.pkl` | Parent→children adjacency `list[list[Edge]]` |
| `text.json` | Flat list of all unique string tokens |
| `text_map.json` | Token → index mapping |
| `column_index.json` | `"col of table"` → index mapping |
| `table_info.json` | Per-table node offset and count |
