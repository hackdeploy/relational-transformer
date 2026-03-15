"""
Preprocessing — reads relbench parquet tables, builds a node graph, and
serialises everything to disk in a Python-native format (pickle + JSON).

Pure-Python port of rustler/src/pre.rs.

Usage:
    python -m pyrustler.pre rel-f1
    python -m pyrustler.pre rel-amazon --skip-db
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .common import Edge, Node, SemType, TableInfo, TableType


def make_column_boolean(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """
    Binarise a column: set to True where the value matches the first row,
    False otherwise.  Mirrors the Rust ``make_column_boolean``.
    """
    col_str = df[col_name].cast(pl.Utf8)
    first = col_str[0]
    mask = col_str == first
    return df.with_columns(mask.alias(col_name))


def _cast_col_to_bool(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return df.with_columns(pl.col(col_name).cast(pl.Boolean))


# ---------------------------------------------------------------------------
# per-dataset fixups — direct port of the Rust match arms in pre.rs
# ---------------------------------------------------------------------------

def _apply_dataset_fixups_pre(
    df: pl.DataFrame,
    db_name: str,
    table_name: str,
) -> pl.DataFrame:
    """Mutate *before* building nodes (binarise special columns)."""

    if db_name == "rel-stack" and table_name == "postLinks":
        df = make_column_boolean(df, "LinkTypeId")

    if db_name == "rel-trial" and table_name == "studies":
        df = make_column_boolean(df, "has_dmc")

    if db_name == "rel-trial" and table_name == "eligibilities":
        for col in ("adult", "child", "older_adult"):
            df = make_column_boolean(df, col)

    return df


def _apply_dataset_fixups_post(
    df: pl.DataFrame,
    db_name: str,
    table_name: str,
) -> pl.DataFrame:
    """Mutate *after* binarisation (cast / drop / reshape)."""

    # amazon
    if db_name == "rel-amazon":
        if table_name in ("user-churn", "item-churn"):
            df = _cast_col_to_bool(df, "churn")
        if table_name == "product":
            # take only the first element of the list column 'category'
            df = df.with_columns(
                pl.col("category").list.first().cast(pl.Utf8).alias("category")
            )

    # stack
    if db_name == "rel-stack":
        if table_name == "user-engagement":
            df = _cast_col_to_bool(df, "contribution")
        if table_name == "user-badge":
            df = _cast_col_to_bool(df, "WillGetBadge")
        if table_name == "posts":
            df = df.drop("AcceptedAnswerId")

    # trial
    if db_name == "rel-trial" and table_name == "study-outcome":
        df = _cast_col_to_bool(df, "outcome")

    # f1
    if db_name == "rel-f1":
        if table_name == "driver-dnf":
            df = _cast_col_to_bool(df, "did_not_finish")
        if table_name == "driver-top3":
            df = _cast_col_to_bool(df, "qualifying")

    # hm
    if db_name == "rel-hm" and table_name == "user-churn":
        df = _cast_col_to_bool(df, "churn")

    # event
    if db_name == "rel-event":
        if table_name == "event_attendees":
            df = df.drop_nulls()
        if table_name == "user_friends":
            df = df.drop_nulls()
            df = df.with_row_index("dummy")
        if table_name in ("user-repeat", "user-ignore", "user-attendance"):
            df = df.drop("index")
            df = _cast_col_to_bool(df, "target")

    # avito
    if db_name == "rel-avito":
        if table_name in ("user-visits", "user-clicks"):
            df = _cast_col_to_bool(df, "num_click")

    return df


# ---------------------------------------------------------------------------
# core preprocessing
# ---------------------------------------------------------------------------

def _read_parquet_metadata(path: str) -> dict[str, str]:
    """Read Parquet key-value metadata (written by relbench)."""
    pf = pq.ParquetFile(path)
    raw_meta = pf.schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in raw_meta.items()}


def _parse_pkey_col(meta: dict) -> str | None:
    val = json.loads(meta["pkey_col"])
    if val is None:
        return None
    return str(val)


def _parse_fkey_map(meta: dict, db_name: str) -> dict[str, str]:
    val = json.loads(meta["fkey_col_to_pkey_table"])
    out: dict[str, str] = {}
    for k, v in val.items():
        if db_name == "rel-avito":
            if k == "UserID":
                v = "UserInfo"
            elif k == "AdID":
                v = "AdsInfo"
        out[k] = v
    return out


def _parse_time_col(meta: dict) -> str | None:
    val = json.loads(meta["time_col"])
    if val is None:
        return None
    return str(val)


class _Table:
    __slots__ = (
        "table_name",
        "df",
        "col_stats",
        "pcol_name",
        "fcol_name_to_ptable_name",
        "tcol_name",
        "node_idx_offset",
    )

    def __init__(self):
        self.table_name: str = ""
        self.df: pl.DataFrame = pl.DataFrame()
        self.col_stats: list[tuple[float, float]] = []  # (mean, std)
        self.pcol_name: str | None = None
        self.fcol_name_to_ptable_name: dict[str, str] = {}
        self.tcol_name: str | None = None
        self.node_idx_offset: int = 0


def main(db_name: str = "rel-f1", skip_db: bool = False) -> None:
    home = os.environ.get("USERPROFILE", os.environ.get("HOME", "."))
    dataset_path = Path(home) / "scratch" / "relbench" / db_name
    print(f"dataset_path: {dataset_path}")

    dashes = db_name.count("-")

    # ------------------------------------------------------------------
    # 1. Read tables
    # ------------------------------------------------------------------
    print("reading tables...")
    tic = time.time()
    table_map: dict[tuple[str, TableType], _Table] = {}
    num_rows_sum = 0
    num_cells_sum = 0

    # Gather parquet paths: db tables then task tables
    pq_entries: list[tuple[bool, Path]] = []
    for p in sorted(dataset_path.glob("db/*.parquet")):
        pq_entries.append((True, p))
    for p in sorted(dataset_path.glob("tasks/*/*.parquet")):
        pq_entries.append((False, p))

    for is_db_table, pq_path in pq_entries:
        # skip secondary parquets (containing extra dashes beyond db_name)
        if str(pq_path).count("-") == dashes + 2:
            continue
        print(f"  {pq_path}")

        df = pl.read_parquet(pq_path)
        meta = _read_parquet_metadata(str(pq_path))

        pcol_name = _parse_pkey_col(meta)
        fcol_name_to_ptable_name = _parse_fkey_map(meta, db_name)
        tcol_name = _parse_time_col(meta)

        if is_db_table:
            table_name = pq_path.stem
            table_type = TableType.Db
        else:
            table_name = pq_path.parent.name
            stem = pq_path.stem
            table_type = {"train": TableType.Train, "val": TableType.Val, "test": TableType.Test}[stem]

        table_key = (table_name, table_type)

        # dataset-specific fixups
        df = _apply_dataset_fixups_pre(df, db_name, table_name)
        df = _apply_dataset_fixups_post(df, db_name, table_name)

        num_rows = df.height
        num_cells = num_rows * df.width

        tbl = _Table()
        tbl.table_name = table_name
        tbl.df = df
        tbl.pcol_name = pcol_name
        tbl.fcol_name_to_ptable_name = fcol_name_to_ptable_name
        tbl.tcol_name = tcol_name
        tbl.node_idx_offset = num_rows_sum

        table_map[table_key] = tbl
        num_rows_sum += num_rows
        num_cells_sum += num_cells

    print(f"done in {time.time() - tic:.2f}s.")

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

            elif dtype == pl.Datetime("ns") or (
                isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns"
            ):
                col_f = col.cast(pl.Float64).drop_nulls()
                col_f = col_f.filter(col_f.is_not_nan())
                dt_cnt += len(col_f)
                vals = col_f.to_list()
                dt_sum += sum(vals)
                dt_sum_sq += sum(v * v for v in vals)
                tbl.col_stats.append((0.0, 0.0))  # placeholder, filled below

            else:
                tbl.col_stats.append((0.0, 0.0))

    dt_mean = dt_sum / dt_cnt if dt_cnt else 0.0
    dt_std = math.sqrt(dt_sum_sq / dt_cnt - dt_mean * dt_mean) if dt_cnt else 1.0
    print(f"  dt_cnt={dt_cnt}  dt_mean={dt_mean:.2f}  dt_std={dt_std:.2f}")

    # Copy Train col_stats to Val/Test
    col_stats_by_name: dict[str, list[tuple[float, float]]] = {}
    for (name, tt), tbl in table_map.items():
        if tt == TableType.Train:
            col_stats_by_name[name] = tbl.col_stats
    for (name, tt), tbl in table_map.items():
        if tt in (TableType.Val, TableType.Test):
            if name in col_stats_by_name:
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
        if text in text_to_idx:
            return text_to_idx[text]
        idx = len(text_to_idx)
        text_to_idx[text] = idx
        return idx

    cells_done = 0
    for (_table_name, table_type), tbl in table_map.items():
        if skip_db and table_type == TableType.Db:
            print(f"  skipping table {tbl.table_name} (Db, --skip-db)")
            continue

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

            # Foreign-key column → build edges instead of cell values
            if col_name_str in tbl.fcol_name_to_ptable_name:
                ptable_name = tbl.fcol_name_to_ptable_name[col_name_str]
                ptable = table_map.get((ptable_name, TableType.Db))
                if ptable is None:
                    print(f"    WARNING: parent table '{ptable_name}' (Db) not found, skipping FK column '{col_name_str}'")
                    cells_done += len(col)
                    continue
                ptable_offset = ptable.node_idx_offset
                ptable_name_idx = _get_text_idx(ptable_name)

                # Pre-fetch timestamp columns as raw nanosecond ints
                # (avoids Python datetime which fails on Windows for old dates)
                tc_vals = None
                if tbl.tcol_name:
                    tc_series = tbl.df[tbl.tcol_name]
                    tc_vals = tc_series.cast(pl.Int64).to_list()

                ptc_vals = None
                if ptable.tcol_name:
                    ptc_series = ptable.df[ptable.tcol_name]
                    ptc_vals = ptc_series.cast(pl.Int64).to_list()

                col_list = col.to_list()
                for r, val in enumerate(col_list):
                    cells_done += 1
                    if val is None:
                        continue

                    node_idx = tbl.node_idx_offset + r
                    node = node_vec[node_idx]
                    node.is_task_node = table_type != TableType.Db
                    node.node_idx = node_idx
                    node.table_name_idx = table_name_idx

                    try:
                        pnode_idx = ptable_offset + int(val)
                    except (ValueError, TypeError):
                        continue

                    node.f2p_nbr_idxs.append(pnode_idx)

                    # Timestamp for this node (nanoseconds → seconds)
                    timestamp = None
                    if tc_vals is not None and tc_vals[r] is not None:
                        timestamp = int(tc_vals[r] // 1_000_000_000)
                    node.timestamp = timestamp

                    # Timestamp for the parent node
                    ptimestamp = None
                    if ptc_vals is not None:
                        pt_raw = ptc_vals[int(val)]
                        if pt_raw is not None:
                            ptimestamp = int(pt_raw // 1_000_000_000)

                    f2p_edge = Edge(
                        node_idx=pnode_idx,
                        table_name_idx=ptable_name_idx,
                        table_type=TableType.Db,
                        timestamp=ptimestamp,
                    )
                    node.f2p_edges.append(f2p_edge)

                    p2f_edge = Edge(
                        node_idx=node_idx,
                        table_name_idx=table_name_idx,
                        table_type=table_type,
                        timestamp=timestamp,
                    )
                    p2f_adj[pnode_idx].append(p2f_edge)

                continue  # next column

            # Regular value columns
            dtype = col.dtype
            col_list = col.to_list()

            for r, val in enumerate(col_list):
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
                    val_normed = (val_float - col_stat_mean) / col_stat_std if col_stat_std != 0.0 else 0.0
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
                            f"Infinite value in {tbl.table_name}.{col_name_str} "
                            f"(mean={col_stat_mean}, std={col_stat_std})"
                        )
                    node.boolean_values.append(0.0)
                    node.number_values.append(fval)
                    node.text_values.append(0)
                    node.datetime_values.append(0.0)
                    node.sem_types.append(SemType.Number)
                    node.col_name_idxs.append(col_name_idx)
                    node.class_value_idx.append(-1)

                elif isinstance(dtype, pl.Datetime) or dtype == pl.Datetime("ns"):
                    # datetime → normalised float
                    # Polars .to_list() may give datetime objects or ints;
                    # cast the column to Int64 upfront for safety (done below).
                    # Here val may still be a Python datetime from the raw list,
                    # so handle both cases.
                    if hasattr(val, "timestamp"):
                        try:
                            ns_val = val.timestamp() * 1e9
                        except OSError:
                            # Some old dates fail .timestamp() on Windows
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

                elif dtype == pl.Utf8 or dtype == pl.String:
                    text_idx = _get_text_idx(str(val))
                    node.boolean_values.append(0.0)
                    node.number_values.append(0.0)
                    node.text_values.append(text_idx)
                    node.datetime_values.append(0.0)
                    node.sem_types.append(SemType.Text)
                    node.col_name_idxs.append(col_name_idx)
                    node.class_value_idx.append(text_idx)

                else:
                    raise TypeError(
                        f"Unsupported dtype {dtype} for "
                        f"{tbl.table_name}.{col_name_str} val={val!r}"
                    )

    print(f"done in {time.time() - tic:.2f}s.  (cells processed: {cells_done:,})")

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    pre_path = Path(home) / "scratch" / "pre" / db_name
    pre_path.mkdir(parents=True, exist_ok=True)

    # text.json
    print("writing text.json ...")
    tic = time.time()
    text_vec: list[str] = [""] * len(text_to_idx)
    for k, v in text_to_idx.items():
        text_vec[v] = k
    with open(pre_path / "text.json", "w") as f:
        json.dump(text_vec, f)

    # text_map.json
    with open(pre_path / "text_map.json", "w") as f:
        json.dump(text_to_idx, f)

    # column_index.json
    column_index = {name: idx for name, idx in column_name_to_idx}
    with open(pre_path / "column_index.json", "w") as f:
        json.dump(column_index, f)
    print(f"done in {time.time() - tic:.2f}s.")

    # table_info.json
    print("writing table_info.json ...")
    tic = time.time()
    table_info: dict[str, dict] = {}
    for (tname, ttype), tbl in table_map.items():
        key = f"{tname}:{ttype.to_label()}"
        table_info[key] = {
            "node_idx_offset": tbl.node_idx_offset,
            "num_nodes": tbl.df.height,
        }
    with open(pre_path / "table_info.json", "w") as f:
        json.dump(table_info, f)
    print(f"done in {time.time() - tic:.2f}s.")

    # nodes.pkl
    print(f"writing nodes.pkl ({len(node_vec):,} nodes) ...")
    tic = time.time()
    with open(pre_path / "nodes.pkl", "wb") as f:
        pickle.dump(node_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done in {time.time() - tic:.2f}s.")

    # p2f_adj.pkl
    print(f"writing p2f_adj.pkl ({len(p2f_adj):,} entries) ...")
    tic = time.time()
    with open(pre_path / "p2f_adj.pkl", "wb") as f:
        pickle.dump(p2f_adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done in {time.time() - tic:.2f}s.")

    print("preprocessing complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="pyrustler preprocessing")
    parser.add_argument("db_name", nargs="?", default="rel-f1")
    parser.add_argument("--skip-db", action="store_true", default=False)
    args = parser.parse_args()
    main(db_name=args.db_name, skip_db=args.skip_db)


if __name__ == "__main__":
    _cli()
