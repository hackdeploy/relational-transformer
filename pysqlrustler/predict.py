"""
pysqlrustler/predict.py

Generic inference for any task table preprocessed with pysqlrustler.

The caller provides:
  - which task table + target column to predict
  - a SQL query that generates the candidate rows to score
  - which column to group rankings by

Everything else (preprocessing, model loading, inference loop) is identical
for every task.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import polars as pl
import psycopg2
import torch
from torch.utils.data import DataLoader

from .pre import main as preprocess
from rt.data import RelationalDataset
from rt.model import RelationalTransformer


def predict(
    # connection
    dsn: str,
    ckpt_path: str,
    base_schema_path: str,
    # task definition
    task_table: str,
    target_column: str,
    foreign_keys: dict[str, str],
    # what to score
    prediction_sql: str,
    # output
    group_by: str,
    id_columns: list[str],
    top_k: int = 3,
    out_path: str | None = None,
    task_type: str = "classification",  # "classification" or "regression"
    # model arch — must match checkpoint
    d_text: int = 384,
    num_blocks: int = 12,
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 1024,
    embedding_model: str = "all-MiniLM-L12-v2",
    seq_len: int = 1024,
    batch_size: int = 32,
    device: str | None = None,
) -> list[dict]:
    """
    Score candidate rows, rank by predicted probability, return top-k per group.

    Returns:
        [
          {
            "group_value": <value of group_by column>,
            "predictions": [{"model_id": ..., "score": 0.97}, ...]
          },
          ...
        ]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path   = Path(ckpt_path).expanduser()
    schema_path = Path(base_schema_path)

    with open(schema_path) as f:
        base_schema = json.load(f)

    db_name: str = base_schema["db_name"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # ------------------------------------------------------------------
        # 1. Inject prediction split into a temp schema
        # ------------------------------------------------------------------
        predict_table = {
            "table_name": task_table,
            "table_type": "Test",
            "primary_key": None,
            "foreign_keys": foreign_keys,
            "time_col": None,
            "sql": prediction_sql,
        }
        db_tables = [t for t in base_schema["tables"] if t["table_type"] == "Db"]
        train_tables = [t for t in base_schema["tables"] if t["table_type"] == "Train"]
        temp_schema = {"db_name": db_name, "tables": db_tables + train_tables + [predict_table]}

        temp_schema_path = os.path.join(tmp_dir, "schema_predict.json")
        with open(temp_schema_path, "w") as f:
            json.dump(temp_schema, f)

        pre_dir = os.path.join(tmp_dir, "pre")

        # ------------------------------------------------------------------
        # 2. Preprocess candidates
        # ------------------------------------------------------------------
        print("Preprocessing candidates...")
        preprocess(dsn=dsn, config_path=temp_schema_path, out_dir=pre_dir)

        with open(os.path.join(pre_dir, "table_info.json")) as f:
            table_info = json.load(f)
        with open(os.path.join(pre_dir, "text.json")) as f:
            text_vec = json.load(f)
        # Read col_stats from the persistent training pre dir — the temp
        # preprocessing has no Train split so its col_stats.json is always empty.
        _home = os.environ.get("HOME") or os.environ.get("USERPROFILE", ".")
        persistent_col_stats_path = os.path.join(_home, "scratch", "pre", db_name, "col_stats.json")
        col_stats: dict = json.load(open(persistent_col_stats_path)) if os.path.exists(persistent_col_stats_path) else {}

        ti = table_info[f"{task_table}:Test"]
        offset    = ti["node_idx_offset"]
        num_nodes = ti["num_nodes"]

        # ------------------------------------------------------------------
        # 3. Fetch the candidate rows in the same order as the SQL result
        #    so we can map node_idx → id_columns values
        # ------------------------------------------------------------------
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(prediction_sql)
                col_names = [d[0] for d in cur.description]
                rows = cur.fetchall()

        assert len(rows) == num_nodes, (
            f"Row count mismatch: SQL returned {len(rows)} rows "
            f"but preprocessed {num_nodes} nodes"
        )
        node_to_row = {offset + i: dict(zip(col_names, rows[i]))
                       for i in range(num_nodes)}

        # ------------------------------------------------------------------
        # 4. Embed prediction text tokens
        # Always re-embed: prediction data may contain new strings not in the
        # training vocab, so copying the training text_emb.bin would be wrong.
        # ------------------------------------------------------------------
        import numpy as np
        from ml_dtypes import bfloat16
        from sentence_transformers import SentenceTransformer

        dest_emb = os.path.join(pre_dir, f"text_emb_{embedding_model}.bin")
        print(f"Embedding {len(text_vec):,} tokens for prediction...")
        st = SentenceTransformer(f"sentence-transformers/{embedding_model}")
        emb = st.encode(text_vec, batch_size=512, show_progress_bar=False,
                        convert_to_numpy=True)
        np.stack(emb).astype(bfloat16).tofile(dest_emb)

        # ------------------------------------------------------------------
        # 5. Point HOME at tmp so RelationalDataset finds the pre dir
        # ------------------------------------------------------------------
        fake_home = os.path.join(tmp_dir, "home")
        os.makedirs(os.path.join(fake_home, "scratch", "pre"), exist_ok=True)
        os.symlink(pre_dir, os.path.join(fake_home, "scratch", "pre", db_name))

        orig_home        = os.environ.get("HOME")
        orig_userprofile = os.environ.get("USERPROFILE")
        os.environ["HOME"]        = fake_home
        os.environ["USERPROFILE"] = fake_home

        try:
            # ------------------------------------------------------------------
            # 6. Load model
            # ------------------------------------------------------------------
            print(f"Loading checkpoint: {ckpt_path}")
            net = RelationalTransformer(
                num_blocks=num_blocks, d_model=d_model, d_text=d_text,
                num_heads=num_heads, d_ff=d_ff,
            )
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # torch.compile() adds "_orig_mod." prefix — strip it for inference
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            net = net.to(torch.bfloat16).to(device).eval()

            # ------------------------------------------------------------------
            # 7. Inference
            # ------------------------------------------------------------------
            dataset = RelationalDataset(
                tasks=[(db_name, task_table, target_column, "test", [])],
                batch_size=batch_size,
                seq_len=seq_len,
                rank=0, world_size=1,
                max_bfs_width=256,
                embedding_model=embedding_model,
                d_text=d_text,
                seed=0,
            )
            dataset.sampler.shuffle_py(0)
            loader = DataLoader(dataset, batch_size=None, num_workers=0)

            # Fetch target column normalization stats from the Train split
            _target_mean, _target_std = 0.0, 1.0
            if task_table in col_stats:
                _tbl_stats = col_stats[task_table]
                if target_column in _tbl_stats["columns"]:
                    _ci = _tbl_stats["columns"].index(target_column)
                    _target_mean, _target_std = _tbl_stats["stats"][_ci]

            node_idx_to_score: dict[int, float] = {}
            with torch.inference_mode():
                for batch in loader:
                    true_bs = batch.pop("true_batch_size")
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)
                    batch["masks"][true_bs:, :]      = False
                    batch["is_targets"][true_bs:, :] = False
                    batch["is_padding"][true_bs:, :] = True

                    _, yhat = net(batch)
                    is_tgt  = batch["is_targets"]
                    nidxs   = batch["node_idxs"][is_tgt][:true_bs].cpu().tolist()
                    key     = "number" if task_type == "regression" else "boolean"
                    scores  = yhat[key][is_tgt][:true_bs].flatten().float().cpu().tolist()
                    for nid, sc in zip(nidxs, scores):
                        # Un-normalize regression scores using Train col stats
                        if task_type == "regression":
                            sc = sc * _target_std + _target_mean
                        node_idx_to_score[nid] = sc

        finally:
            if orig_home is not None:
                os.environ["HOME"] = orig_home
            if orig_userprofile is not None:
                os.environ["USERPROFILE"] = orig_userprofile

    # ------------------------------------------------------------------
    # 8. Group by group_by column, rank by probability, return top-k
    # ------------------------------------------------------------------
    groups: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for node_idx, raw_score in node_idx_to_score.items():
        row   = node_to_row[node_idx]
        score = raw_score if task_type == "regression" else float(torch.sigmoid(torch.tensor(raw_score)))
        groups[row[group_by]].append((row, score))

    results = []
    for group_val, candidates in groups.items():
        ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        results.append({
            "group_value": group_val,
            "predictions": [
                {col: row[col] for col in id_columns} | {"score": round(prob, 4)}
                for row, prob in ranked
            ],
        })

    # ------------------------------------------------------------------
    # 9. Print + optional CSV
    # ------------------------------------------------------------------
    for r in results:
        print(f"\n{group_by}={r['group_value']}")
        for i, p in enumerate(r["predictions"]):
            print(f"  #{i+1}  {p}")

    if out_path:
        flat = [
            {"rank": i + 1, "group_value": r["group_value"]} | p
            for r in results
            for i, p in enumerate(r["predictions"])
        ]
        pl.DataFrame(flat).write_csv(out_path)
        print(f"\nSaved → {out_path}")

    return results
