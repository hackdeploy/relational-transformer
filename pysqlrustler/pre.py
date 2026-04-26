"""
pysqlrustler/pre.py

PostgreSQL-backed preprocessing — reads from Postgres tables instead of
relbench parquet files, builds the same node graph, and serialises
everything to disk in the same Python-native format (pickle + JSON) that
pyrustler/pre.py produces.

Dependencies (beyond the base pixi env):
    pip install psycopg2-binary pandas

Usage:
    python -m pysqlrustler.pre --dsn "postgresql://user:pass@host:5432/db" \\
                               --config schema.json

    python -m pysqlrustler.pre --dsn "postgresql://..." \\
                               --config schema.json \\
                               --out-dir ~/scratch/pre/mydb

schema.json format
------------------
{
  "db_name": "mydb",
  "tables": [
    {
      "table_name":  "users",
      "table_type":  "Db",           // Db | Train | Val | Test
      "primary_key": "user_id",      // null if the table has no PK
      "foreign_keys": {              // FK column -> parent table name
        "account_id": "accounts"
      },
      "time_col": null,              // null if there is no timestamp column
      "sql": null                    // null => "SELECT * FROM <table_name>"
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
      "sql": "SELECT user_id, label FROM label_table WHERE split = 'train'"
    }
  ]
}

Output (written to --out-dir or ~/scratch/pre/<db_name>/)
---------------------------------------------------------
  text.json          — flat list of all unique string tokens
  text_map.json      — token -> index mapping
  column_index.json  — "col of table" -> index mapping
  table_info.json    — per-table node offset + count
  nodes.pkl          — list[Node]
  p2f_adj.pkl        — parent->children adjacency list[list[Edge]]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import polars as pl

from .common import Edge, Node, SemType, TableType


# ---------------------------------------------------------------------------
# PostgreSQL loading
# ---------------------------------------------------------------------------

def _load_table_postgres(dsn: str, sql: str) -> pl.DataFrame:
    """Execute *sql* against *dsn* and return a normalised Polars DataFrame."""
    import decimal

    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "pysqlrustler requires 'psycopg2-binary'.\n"
            "Install with:  pip install psycopg2-binary"
        ) from exc

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            col_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    if not rows:
        return pl.DataFrame({col: pl.Series(col, [], dtype=pl.Utf8) for col in col_names})

    # Build column-wise lists, converting Decimal → float so polars can infer types.
    series_list = []
    for i, col_name in enumerate(col_names):
        values = [row[i] for row in rows]
        if any(isinstance(v, decimal.Decimal) for v in values if v is not None):
            values = [float(v) if v is not None else None for v in values]
        series_list.append(pl.Series(col_name, values))

    df = pl.DataFrame(series_list)

    # Normalise datetime precision and integer widths.
    cast_exprs = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if isinstance(dtype, pl.Datetime) and dtype.time_unit != "ns":
            cast_exprs.append(pl.col(col_name).dt.cast_time_unit("ns"))
        elif dtype in (pl.Int8, pl.Int16):
            cast_exprs.append(pl.col(col_name).cast(pl.Int32))
        elif dtype in (pl.UInt8, pl.UInt16, pl.UInt64):
            cast_exprs.append(pl.col(col_name).cast(pl.Int64))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


# ---------------------------------------------------------------------------
# Internal table container (mirrors pyrustler._Table)
# ---------------------------------------------------------------------------

class _Table:
    __slots__ = (
        "table_name",
        "df",
        "col_stats",
        "pcol_name",
        "fcol_name_to_ptable_name",
        "tcol_name",
        "node_idx_offset",
        "pk_to_row",
    )

    def __init__(self) -> None:
        self.table_name: str = ""
        self.df: pl.DataFrame = pl.DataFrame()
        self.col_stats: list[tuple[float, float]] = []
        self.pcol_name: str | None = None
        self.fcol_name_to_ptable_name: dict[str, str] = {}
        self.tcol_name: str | None = None
        self.node_idx_offset: int = 0
        self.pk_to_row: dict = {}  # pk_value → 0-based row index


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(
    dsn: str,
    config_path: str,
    out_dir: str | None = None,
) -> None:
    # ------------------------------------------------------------------
    # 0. Load schema config
    # ------------------------------------------------------------------
    with open(config_path) as f:
        cfg = json.load(f)

    db_name: str = cfg.get("db_name", "pgsql_db")
    table_cfgs: list[dict] = cfg["tables"]

    if out_dir is None:
        home = os.environ.get("USERPROFILE", os.environ.get("HOME", "."))
        out_dir = str(Path(home) / "scratch" / "pre" / db_name)

    pre_path = Path(out_dir)
    pre_path.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {pre_path}")

    # ------------------------------------------------------------------
    # 1. Read tables from Postgres
    # ------------------------------------------------------------------
    print("reading tables from postgres...")
    tic = time.time()

    table_map: dict[tuple[str, TableType], _Table] = {}
    num_rows_sum = 0
    num_cells_sum = 0

    for tcfg in table_cfgs:
        table_name: str = tcfg["table_name"]
        table_type = TableType[tcfg["table_type"]]
        pcol_name: str | None = tcfg.get("primary_key") or None
        fcol_name_to_ptable_name: dict[str, str] = tcfg.get("foreign_keys") or {}
        tcol_name: str | None = tcfg.get("time_col") or None
        sql: str = tcfg.get("sql") or f"SELECT * FROM {table_name}"

        print(f"  {table_name!r} ({table_type.to_label()})  ← {sql[:80]}")
        df = _load_table_postgres(dsn, sql)

        num_rows = df.height
        num_cells = num_rows * df.width
        print(f"    {num_rows:,} rows × {df.width} cols")

        tbl = _Table()
        tbl.table_name = table_name
        tbl.df = df
        tbl.pcol_name = pcol_name
        tbl.fcol_name_to_ptable_name = fcol_name_to_ptable_name
        tbl.tcol_name = tcol_name
        tbl.node_idx_offset = num_rows_sum
        # Map each PK value → its 0-based row index so FK lookups work with
        # any PK type (UUID, non-sequential int, etc.), not just 0-based ints.
        if pcol_name and pcol_name in df.columns:
            tbl.pk_to_row = {pk: i for i, pk in enumerate(df[pcol_name].to_list())}

        table_map[(table_name, table_type)] = tbl
        num_rows_sum += num_rows
        num_cells_sum += num_cells

    print(f"done in {time.time() - tic:.2f}s.  "
          f"({num_rows_sum:,} rows total, {num_cells_sum:,} cells total)")

    # ------------------------------------------------------------------
    # 2. Column stats
    # ------------------------------------------------------------------
    print("computing column stats...")
    tic = time.time()

    dt_cnt = 0
    dt_sum = 0.0
    dt_sum_sq = 0.0

    for tbl in table_map.values():
        for col_series in tbl.df.get_columns():
            col = col_series.rechunk()
            dtype = col.dtype

            if dtype == pl.Boolean:
                col_float = col.cast(pl.Float64).drop_nulls()
                col_mean = col_float.mean() or 0.0
                col_std = col_float.std() or 0.0
                tbl.col_stats.append((col_mean, col_std))

            elif dtype in (pl.UInt32, pl.Int32, pl.Int64, pl.Float64, pl.Float32):
                col_f = col.cast(pl.Float64).drop_nulls()
                col_f = col_f.filter(col_f.is_not_nan())
                mean = col_f.mean() or 0.0
                std = col_f.std() or 1.0
                if std == 0.0:
                    std = 1.0
                tbl.col_stats.append((mean, std))

            elif isinstance(dtype, pl.Datetime):
                col_f = col.cast(pl.Int64).drop_nulls().cast(pl.Float64)
                col_f = col_f.filter(col_f.is_not_nan())
                dt_cnt += len(col_f)
                vals = col_f.to_list()
                dt_sum += sum(vals)
                dt_sum_sq += sum(v * v for v in vals)
                tbl.col_stats.append((0.0, 0.0))  # placeholder, filled below

            elif isinstance(dtype, (pl.List, pl.Array)):
                tbl.col_stats.append((0.0, 0.0))  # treated as text

            else:
                tbl.col_stats.append((0.0, 0.0))

    dt_mean = dt_sum / dt_cnt if dt_cnt else 0.0
    # Clamp to avoid sqrt of negative due to floating-point rounding
    dt_var = max(dt_sum_sq / dt_cnt - dt_mean * dt_mean, 0.0) if dt_cnt else 1.0
    dt_std = math.sqrt(dt_var) if dt_var > 0.0 else 1.0
    print(f"  dt_cnt={dt_cnt}  dt_mean={dt_mean:.2f}  dt_std={dt_std:.2f}")

    # Propagate Train col_stats → Val / Test (same table name, different split)
    col_stats_by_name: dict[str, list[tuple[float, float]]] = {}
    for (name, tt), tbl in table_map.items():
        if tt == TableType.Train:
            col_stats_by_name[name] = tbl.col_stats
    for (name, tt), tbl in table_map.items():
        if tt in (TableType.Val, TableType.Test) and name in col_stats_by_name:
            tbl.col_stats = col_stats_by_name[name]

    print(f"done in {time.time() - tic:.2f}s.")

    # ------------------------------------------------------------------
    # 3. Build node vector + adjacency
    # ------------------------------------------------------------------
    print("making node vector...")
    tic = time.time()

    text_to_idx: dict[str, int] = {}
    column_name_to_idx: list[tuple[str, int]] = []
    node_vec: list[Node] = [Node() for _ in range(num_rows_sum)]
    p2f_adj: list[list[Edge]] = [[] for _ in range(num_rows_sum)]

    def _get_text_idx(text: str) -> int:
        if text not in text_to_idx:
            text_to_idx[text] = len(text_to_idx)
        return text_to_idx[text]

    cells_done = 0

    for (_table_name, table_type), tbl in table_map.items():
        table_name_idx = _get_text_idx(tbl.table_name)

        for col_i, col_series in enumerate(tbl.df.get_columns()):
            col = col_series.rechunk()
            col_name_str = col.name
            col_stat_mean, col_stat_std = tbl.col_stats[col_i]

            full_col_name = f"{col_name_str} of {tbl.table_name}"
            old_len = len(text_to_idx)
            col_name_idx = _get_text_idx(full_col_name)
            if col_name_idx >= old_len:
                column_name_to_idx.append((full_col_name, col_name_idx))

            # Skip primary-key column
            if col_name_str == (tbl.pcol_name or ""):
                cells_done += len(col)
                continue

            # Foreign-key column → build graph edges instead of cell values
            if col_name_str in tbl.fcol_name_to_ptable_name:
                ptable_name = tbl.fcol_name_to_ptable_name[col_name_str]
                ptable = table_map.get((ptable_name, TableType.Db))
                if ptable is None:
                    print(
                        f"    WARNING: parent table '{ptable_name}' (Db) not found, "
                        f"skipping FK column '{col_name_str}'"
                    )
                    cells_done += len(col)
                    continue

                ptable_offset = ptable.node_idx_offset
                ptable_name_idx = _get_text_idx(ptable_name)

                # Pre-cast timestamp columns to nanosecond ints for speed
                tc_vals: list | None = None
                if tbl.tcol_name:
                    if tbl.tcol_name not in tbl.df.columns:
                        print(f"    WARNING: time_col '{tbl.tcol_name}' not in '{tbl.table_name}' columns, ignoring")
                    else:
                        tc_vals = tbl.df[tbl.tcol_name].cast(pl.Int64).to_list()

                ptc_vals: list | None = None
                if ptable.tcol_name:
                    if ptable.tcol_name not in ptable.df.columns:
                        print(f"    WARNING: time_col '{ptable.tcol_name}' not in '{ptable.table_name}' columns, ignoring")
                    else:
                        ptc_vals = ptable.df[ptable.tcol_name].cast(pl.Int64).to_list()

                for r, val in enumerate(col.to_list()):
                    cells_done += 1
                    if val is None:
                        continue

                    node_idx = tbl.node_idx_offset + r
                    node = node_vec[node_idx]
                    node.is_task_node = table_type != TableType.Db
                    node.node_idx = node_idx
                    node.table_name_idx = table_name_idx

                    if ptable.pk_to_row:
                        local_row = ptable.pk_to_row.get(val)
                        if local_row is None:
                            continue
                        pnode_idx = ptable_offset + local_row
                    else:
                        try:
                            pnode_idx = ptable_offset + int(val)
                        except (ValueError, TypeError):
                            continue

                    node.f2p_nbr_idxs.append(pnode_idx)

                    timestamp: int | None = None
                    if tc_vals is not None and tc_vals[r] is not None:
                        timestamp = int(tc_vals[r] // 1_000_000_000)
                    node.timestamp = timestamp

                    ptimestamp: int | None = None
                    if ptc_vals is not None:
                        local_row = ptable.pk_to_row.get(val) if ptable.pk_to_row else None
                        if local_row is None:
                            try:
                                local_row = int(val)
                            except (ValueError, TypeError):
                                local_row = None
                        if local_row is not None:
                            pt_raw = ptc_vals[local_row]
                            if pt_raw is not None:
                                ptimestamp = int(pt_raw // 1_000_000_000)

                    node.f2p_edges.append(Edge(
                        node_idx=pnode_idx,
                        table_name_idx=ptable_name_idx,
                        table_type=TableType.Db,
                        timestamp=ptimestamp,
                    ))
                    p2f_adj[pnode_idx].append(Edge(
                        node_idx=node_idx,
                        table_name_idx=table_name_idx,
                        table_type=table_type,
                        timestamp=timestamp,
                    ))

                continue  # next column

            # Regular value column
            dtype = col.dtype

            for r, val in enumerate(col.to_list()):
                cells_done += 1
                node_idx = tbl.node_idx_offset + r
                node = node_vec[node_idx]
                node.is_task_node = table_type != TableType.Db
                node.node_idx = node_idx
                node.table_name_idx = table_name_idx

                if val is None:
                    continue

                if dtype == pl.Boolean:
                    val_float = 1.0 if val else 0.0
                    val_normed = (
                        (val_float - col_stat_mean) / col_stat_std
                        if col_stat_std != 0.0
                        else 0.0
                    )
                    node.boolean_values.append(val_normed)
                    node.number_values.append(0.0)
                    node.text_values.append(0)
                    node.datetime_values.append(0.0)
                    node.sem_types.append(SemType.Boolean)
                    node.col_name_idxs.append(col_name_idx)
                    node.class_value_idx.append(-1)

                elif dtype in (pl.UInt32, pl.Int32, pl.Int64, pl.Float64, pl.Float32):
                    fval = float(val)
                    if math.isnan(fval):
                        continue
                    fval = (fval - col_stat_mean) / col_stat_std
                    if math.isinf(fval):
                        raise ValueError(
                            f"Infinite value after normalisation in "
                            f"{tbl.table_name}.{col_name_str} "
                            f"(mean={col_stat_mean}, std={col_stat_std})"
                        )
                    node.boolean_values.append(0.0)
                    node.number_values.append(fval)
                    node.text_values.append(0)
                    node.datetime_values.append(0.0)
                    node.sem_types.append(SemType.Number)
                    node.col_name_idxs.append(col_name_idx)
                    node.class_value_idx.append(-1)

                elif isinstance(dtype, pl.Datetime):
                    if hasattr(val, "timestamp"):
                        try:
                            ns_val = val.timestamp() * 1e9
                        except OSError:
                            import datetime as _dt
                            epoch = _dt.datetime(1970, 1, 1, tzinfo=val.tzinfo)
                            ns_val = (val - epoch).total_seconds() * 1e9
                    else:
                        ns_val = float(val)
                    normed = (ns_val - dt_mean) / dt_std if dt_std != 0.0 else 0.0
                    node.boolean_values.append(0.0)
                    node.number_values.append(0.0)
                    node.text_values.append(0)
                    node.datetime_values.append(normed)
                    node.sem_types.append(SemType.DateTime)
                    node.col_name_idxs.append(col_name_idx)
                    node.class_value_idx.append(-1)

                elif dtype in (pl.Utf8, pl.String) or isinstance(dtype, (pl.List, pl.Array)):
                    # Arrays/lists are serialised to a string and treated as Text.
                    str_val = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
                    text_idx = _get_text_idx(str_val)
                    node.boolean_values.append(0.0)
                    node.number_values.append(0.0)
                    node.text_values.append(text_idx)
                    node.datetime_values.append(0.0)
                    node.sem_types.append(SemType.Text)
                    node.col_name_idxs.append(col_name_idx)
                    node.class_value_idx.append(text_idx)

                else:
                    raise TypeError(
                        f"Unsupported dtype {dtype} in "
                        f"{tbl.table_name}.{col_name_str} (val={val!r}). "
                        "Cast the column to a supported type in your SQL query."
                    )

    print(f"done in {time.time() - tic:.2f}s.  (cells processed: {cells_done:,})")

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    text_vec: list[str] = [""] * len(text_to_idx)
    for k, v in text_to_idx.items():
        text_vec[v] = k

    print("writing text.json / text_map.json / column_index.json ...")
    tic = time.time()
    with open(pre_path / "text.json", "w") as f:
        json.dump(text_vec, f)
    with open(pre_path / "text_map.json", "w") as f:
        json.dump(text_to_idx, f)
    with open(pre_path / "column_index.json", "w") as f:
        json.dump({name: idx for name, idx in column_name_to_idx}, f)
    print(f"done in {time.time() - tic:.2f}s.")

    print("writing table_info.json ...")
    tic = time.time()
    table_info: dict[str, dict] = {}
    for (tname, ttype), tbl in table_map.items():
        table_info[f"{tname}:{ttype.to_label()}"] = {
            "node_idx_offset": tbl.node_idx_offset,
            "num_nodes": tbl.df.height,
        }
    with open(pre_path / "table_info.json", "w") as f:
        json.dump(table_info, f)
    print(f"done in {time.time() - tic:.2f}s.")

    print(f"writing nodes.pkl ({len(node_vec):,} nodes) ...")
    tic = time.time()
    with open(pre_path / "nodes.pkl", "wb") as f:
        pickle.dump(node_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done in {time.time() - tic:.2f}s.")

    print(f"writing p2f_adj.pkl ({len(p2f_adj):,} entries) ...")
    tic = time.time()
    with open(pre_path / "p2f_adj.pkl", "wb") as f:
        pickle.dump(p2f_adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done in {time.time() - tic:.2f}s.")

    print("preprocessing complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="pysqlrustler — preprocess Postgres tables into the "
                    "relational-transformer node graph format."
    )
    parser.add_argument(
        "--dsn",
        required=True,
        help='Postgres DSN, e.g. "postgresql://user:pass@host:5432/mydb"',
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to schema JSON config (see module docstring for format)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: ~/scratch/pre/<db_name>)",
    )
    args = parser.parse_args()
    main(dsn=args.dsn, config_path=args.config, out_dir=args.out_dir)


if __name__ == "__main__":
    _cli()
