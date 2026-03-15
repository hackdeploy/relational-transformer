"""
Run predictions with a pretrained RelationalTransformer checkpoint against a
relbench task database. The task split is defined by a relbench parquet table
(entity IDs + timestamps + optional labels). Predictions are merged back onto
those entity rows and written to CSV.

Usage (defaults match the pretrain_rel-amazon_item-churn checkpoint):

    python scripts/predict.py

Override any parameter via CLI (powered by strictfire):

    python scripts/predict.py \
        --ckpt_path ~/relational-transformer-checkpoints/pretrain_rel-amazon_item-churn.pt \
        --db_name rel-amazon \
        --task_name item-churn \
        --target_column churn \
        --split test \
        --out_path predictions.csv
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from relbench.datasets import get_dataset  # noqa: F401  (triggers cache setup)
from relbench.tasks import get_task
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import RelationalDataset
from rt.model import RelationalTransformer
from rt.tasks import forecast_reg_tasks

# Regression table names — everything else is treated as classification.
REG_TABLE_NAMES = {t[1] for t in forecast_reg_tasks}

HOME = os.environ.get("HOME", os.path.expanduser("~"))


def _load_table_info(db_name: str) -> dict:
    path = Path(HOME) / "scratch" / "pre" / db_name / "table_info.json"
    with open(path) as f:
        return json.load(f)


def predict(
    # checkpoint
    ckpt_path: str = os.path.join(
        HOME,
        "relational-transformer-checkpoints",
        "pretrain_rel-amazon_item-churn.pt",
    ),
    # task
    db_name: str = "rel-amazon",
    task_name: str = "item-churn",      # relbench task name (also the table_name)
    target_column: str = "churn",
    columns_to_drop: list[str] = [],
    split: str = "test",                # "train" | "val" | "test"
    # output
    out_path: str = "predictions.csv",
    # data
    batch_size: int = 32,
    seq_len: int = 1024,
    max_bfs_width: int = 256,
    embedding_model: str = "all-MiniLM-L12-v2",
    max_steps: int = -1,                # -1 = full split; e.g. 10 for a quick test
    num_workers: int = 0,               # 0 is safest on Windows; increase on Linux
    # model architecture — must match the checkpoint
    d_text: int = 384,
    num_blocks: int = 12,
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 1024,
    # runtime
    device: str = "cpu",               # "cuda" if a GPU is available
    dtype: str = "bfloat16",           # "float32" on older hardware without bf16
):
    ckpt_path = Path(ckpt_path).expanduser()
    out_path = Path(out_path).expanduser()
    torch_dtype = getattr(torch, dtype)
    task_type = "reg" if task_name in REG_TABLE_NAMES else "clf"

    # ------------------------------------------------------------------
    # Load relbench task parquet → entity DataFrame
    # ------------------------------------------------------------------
    print(f"Loading relbench task: {db_name} / {task_name} / {split}")
    rb_task = get_task(db_name, task_name, download=False)
    task_df: pd.DataFrame = rb_task.get_table(split).df.reset_index(drop=True)
    print(f"  {len(task_df):,} rows  columns: {list(task_df.columns)}")

    # ------------------------------------------------------------------
    # Map each task-table row → global node_idx
    # node_idx = node_idx_offset + row_position  (rows are ordered identically
    # in the relbench parquet and in the preprocessed graph split)
    # ------------------------------------------------------------------
    table_info = _load_table_info(db_name)

    # Try split-specific key first, fall back to the Db-level key
    split_cap = split.capitalize()          # "test" → "Test"
    ti_key = f"{task_name}:{split_cap}"
    if ti_key not in table_info:
        ti_key = f"{task_name}:Db"
    if ti_key not in table_info:
        raise KeyError(
            f"Could not find table_info entry for '{task_name}:{split_cap}' "
            f"or '{task_name}:Db' in {db_name}/table_info.json. "
            f"Available keys: {list(table_info.keys())}"
        )

    node_idx_offset: int = table_info[ti_key]["node_idx_offset"]
    print(f"  node_idx_offset={node_idx_offset}  (key='{ti_key}')")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {ckpt_path}")
    net = RelationalTransformer(
        num_blocks=num_blocks,
        d_model=d_model,
        d_text=d_text,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net = net.to(torch_dtype).to(device)
    net.eval()

    param_count = sum(p.numel() for p in net.parameters())
    print(f"param_count={param_count:_}")
    print(f"task_type={task_type}  device={device}  dtype={dtype}\n")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = RelationalDataset(
        tasks=[(db_name, task_name, target_column, split, columns_to_drop)],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=0,
        world_size=1,
        max_bfs_width=max_bfs_width,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=0,
    )

    dataset.sampler.shuffle_py(0)  # initialise sampler ordering (matches main.py eval)

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=(device != "cpu"),
    )

    n_steps = min(max_steps, len(loader)) if max_steps > 0 else len(loader)
    print(f"Running {n_steps} batch(es) [{db_name}/{task_name}/{split}]…\n")

    # ------------------------------------------------------------------
    # Inference — collect node_idx → prediction
    # ------------------------------------------------------------------
    node_idx_to_pred: dict[int, float] = {}
    batch_times: list[float] = []

    with torch.inference_mode():
        pbar = tqdm(total=n_steps, desc=f"{db_name}/{task_name}/{split}")

        for batch_idx, batch in enumerate(loader):
            tic = time.time()

            true_batch_size = batch.pop("true_batch_size")

            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            # Mask out padding rows beyond true_batch_size
            batch["masks"][true_batch_size:, :] = False
            batch["is_targets"][true_batch_size:, :] = False
            batch["is_padding"][true_batch_size:, :] = True

            _loss, yhat_dict = net(batch)

            toc = time.time()
            batch_times.append(toc - tic)

            is_targets = batch["is_targets"]  # [B, L] bool

            # One node_idx per sample — the task-node for each sequence
            node_idxs_batch = (
                batch["node_idxs"][is_targets][:true_batch_size].cpu().tolist()
            )

            if task_type == "clf":
                preds = yhat_dict["boolean"][is_targets][:true_batch_size]
            else:
                preds = yhat_dict["number"][is_targets][:true_batch_size]

            preds = preds.flatten().float().cpu().tolist()

            for nid, pred in zip(node_idxs_batch, preds):
                node_idx_to_pred[nid] = pred

            pbar.update(1)
            if max_steps > 0 and (batch_idx + 1) >= max_steps:
                break

        pbar.close()

    # ------------------------------------------------------------------
    # Merge predictions back onto the task DataFrame
    # Each parquet row i → node_idx = node_idx_offset + i
    # ------------------------------------------------------------------
    task_df["node_idx"] = node_idx_offset + task_df.index
    task_df["prediction"] = task_df["node_idx"].map(node_idx_to_pred)

    # For classification add a probability column (sigmoid of log-odds)
    if task_type == "clf":
        task_df["probability"] = torch.sigmoid(
            torch.tensor(
                task_df["prediction"].fillna(float("nan")).values,
                dtype=torch.float32,
            )
        ).numpy()

    # For regression, rescale back to original units.
    # Rustler z-scores each numeric column using the TRAIN split's mean/std
    # (ddof=1, NaNs dropped) and never saves those stats to disk.
    # We recompute them here from the same source used by Rustler.
    if task_type == "reg":
        train_col = rb_task.get_table("train").df[target_column].dropna()
        mu = float(train_col.mean())
        sigma = float(train_col.std(ddof=1))  # matches Polars std(1)
        sigma = sigma if sigma != 0.0 else 1.0
        task_df["prediction_rescaled"] = task_df["prediction"] * sigma + mu
        print(
            f"Rescaled predictions to original units "
            f"(train μ={mu:.4f}, σ={sigma:.4f})"
        )

    n_missing = int(task_df["prediction"].isna().sum())
    if n_missing:
        print(
            f"\nWarning: {n_missing:,} rows have no prediction "
            "(may occur when max_steps < full dataset)."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_df.to_csv(out_path, index=False)

    avg_ms = float(np.mean(batch_times)) * 1000
    print(f"\nWrote {len(task_df):,} rows → {out_path}")
    print(f"avg_batch_ms={avg_ms:.1f}  total_batches={len(batch_times)}")
    extra = "prediction (log-odds), probability (sigmoid)" if task_type == "clf" else "prediction (z-scored), prediction_rescaled (original units)"
    print(f"Columns: entity id(s), timestamp, label, node_idx, {extra}")

    return out_path


if __name__ == "__main__":
    try:
        import strictfire
        strictfire.StrictFire(predict)
    except ImportError:
        predict()
