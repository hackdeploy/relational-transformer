"""
Predict the best matching model_id for new listings.

Usage (Colab / CLI):
    python scripts/predict_motolote.py \
        --dsn  "postgresql://..." \
        --ckpt "$HOME/scratch/ckpts/motolote_best.pt" \
        --listing-ids "uuid-1,uuid-2,uuid-3" \
        --top-k 3 \
        --out  predictions.csv

Or call predict() directly from Python.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path

import polars as pl
import psycopg2
import torch

from pysqlrustler.pre import main as preprocess
from pysqlrustler.common import TableType
from rt.data import RelationalDataset
from rt.model import RelationalTransformer
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fetch_model_ids(dsn: str) -> list[str]:
    """Return all model IDs available in the models table."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM models ORDER BY id")
            return [row[0] for row in cur.fetchall()]


def _build_temp_schema(
    dsn: str,
    listing_ids: list[str],
    base_schema_path: str,
    tmp_dir: str,
) -> str:
    """
    Clone the base schema.json, replacing Train/Val/Test splits with a
    single Test split that cross-joins the requested listing_ids × all models.
    Returns path to the temporary schema file.
    """
    with open(base_schema_path) as f:
        base = json.load(f)

    ids_sql = ", ".join(f"'{lid}'" for lid in listing_ids)

    predict_table = {
        "table_name": "listing_model_matches",
        "table_type": "Test",
        "primary_key": None,
        "foreign_keys": {"listing_id": "listings", "model_id": "models"},
        "time_col": None,
        "sql": (
            f"SELECT l.id AS listing_id, m.id AS model_id, "
            f"false::boolean AS label "
            f"FROM listings l CROSS JOIN models m "
            f"WHERE l.id IN ({ids_sql})"
        ),
    }

    # Keep only Db tables + add prediction split
    tables = [t for t in base["tables"] if t["table_type"] == "Db"]
    tables.append(predict_table)

    schema = {"db_name": base["db_name"], "tables": tables}
    schema_path = os.path.join(tmp_dir, "schema_predict.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f)
    return schema_path


# ---------------------------------------------------------------------------
# main predict function
# ---------------------------------------------------------------------------

def predict(
    dsn: str,
    ckpt_path: str,
    listing_ids: list[str],
    base_schema_path: str = "pysqlrustler/schema.json",
    top_k: int = 3,
    out_path: str | None = "predictions.csv",
    # model architecture — must match the checkpoint
    d_text: int = 384,
    num_blocks: int = 12,
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 1024,
    embedding_model: str = "all-MiniLM-L12-v2",
    seq_len: int = 1024,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict]:
    """
    Returns a list of dicts:
        [{"listing_id": ..., "predictions": [{"model_id": ..., "score": ...}, ...]}, ...]
    """
    ckpt_path = Path(ckpt_path).expanduser()
    base_schema_path = Path(base_schema_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # ------------------------------------------------------------------
        # 1. Build a temporary schema + preprocess
        # ------------------------------------------------------------------
        print(f"Preprocessing {len(listing_ids)} listing(s) × all models...")
        schema_path = _build_temp_schema(dsn, listing_ids, str(base_schema_path), tmp_dir)
        pre_dir = os.path.join(tmp_dir, "pre")

        preprocess(dsn=dsn, config_path=schema_path, out_dir=pre_dir)

        # ------------------------------------------------------------------
        # 2. Load preprocessed artefacts
        # ------------------------------------------------------------------
        with open(os.path.join(pre_dir, "table_info.json")) as f:
            table_info = json.load(f)

        with open(os.path.join(pre_dir, "text.json")) as f:
            text_vec = json.load(f)

        with open(os.path.join(pre_dir, "nodes.pkl"), "rb") as f:
            nodes = pickle.load(f)

        # Map global node_idx → (listing_id, model_id)
        ti = table_info["listing_model_matches:Test"]
        offset = ti["node_idx_offset"]
        num_nodes = ti["num_nodes"]

        # Reconstruct the listing_id / model_id for each Test node
        # (same order as the SQL result)
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                ids_sql = ", ".join(f"'{lid}'" for lid in listing_ids)
                cur.execute(
                    f"SELECT l.id, m.id FROM listings l CROSS JOIN models m "
                    f"WHERE l.id IN ({ids_sql}) ORDER BY l.id, m.id"
                )
                pairs = cur.fetchall()  # [(listing_id, model_id), ...]

        assert len(pairs) == num_nodes, (
            f"Pair count mismatch: {len(pairs)} SQL vs {num_nodes} nodes"
        )
        node_to_pair = {offset + i: pairs[i] for i in range(num_nodes)}

        # ------------------------------------------------------------------
        # 3. Copy the text embedding from the training pre-dir so the
        #    Sampler can find it (it expects the .bin next to nodes.pkl)
        # ------------------------------------------------------------------
        home = os.environ.get("USERPROFILE", os.environ.get("HOME", "."))
        train_emb = (
            Path(home) / "scratch" / "pre" / "motolote"
            / f"text_emb_{embedding_model}.bin"
        )
        import shutil
        if train_emb.exists():
            shutil.copy(train_emb, os.path.join(pre_dir, f"text_emb_{embedding_model}.bin"))
        else:
            print(f"WARNING: {train_emb} not found — re-embedding text tokens...")
            import numpy as np
            from ml_dtypes import bfloat16
            from sentence_transformers import SentenceTransformer
            model_st = SentenceTransformer(f"sentence-transformers/{embedding_model}")
            emb = model_st.encode(text_vec, batch_size=512, show_progress_bar=True,
                                  convert_to_numpy=True)
            np.stack(emb).astype(bfloat16).tofile(
                os.path.join(pre_dir, f"text_emb_{embedding_model}.bin")
            )

        # Override HOME/USERPROFILE so RelationalDataset finds the tmp pre_dir
        orig_home = os.environ.get("HOME")
        orig_userprofile = os.environ.get("USERPROFILE")
        # We point HOME to tmp_dir so scratch/pre/motolote resolves correctly
        fake_home = tmp_dir
        os.makedirs(os.path.join(fake_home, "scratch", "pre"), exist_ok=True)
        os.symlink(pre_dir, os.path.join(fake_home, "scratch", "pre", "motolote"))
        os.environ["HOME"] = fake_home
        os.environ["USERPROFILE"] = fake_home

        try:
            # ------------------------------------------------------------------
            # 4. Load model
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
            net = net.to(device)
            net.eval()
            print(f"param_count={sum(p.numel() for p in net.parameters()):_}")

            # ------------------------------------------------------------------
            # 5. Dataset + inference
            # ------------------------------------------------------------------
            with open(os.path.join(pre_dir, "column_index.json")) as f:
                col_idx = json.load(f)
            label_col_key = "label of listing_model_matches"
            if label_col_key not in col_idx:
                raise KeyError(f"'{label_col_key}' not found in column_index.json")

            dataset = RelationalDataset(
                tasks=[("motolote", "listing_model_matches", "label", "test", [])],
                batch_size=batch_size,
                seq_len=seq_len,
                rank=0,
                world_size=1,
                max_bfs_width=256,
                embedding_model=embedding_model,
                d_text=d_text,
                seed=0,
            )
            dataset.sampler.shuffle_py(0)

            loader = DataLoader(dataset, batch_size=None, num_workers=0)

            node_idx_to_score: dict[int, float] = {}
            with torch.inference_mode():
                for batch in loader:
                    true_bs = batch.pop("true_batch_size")
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)
                    batch["masks"][true_bs:, :] = False
                    batch["is_targets"][true_bs:, :] = False
                    batch["is_padding"][true_bs:, :] = True

                    _, yhat = net(batch)
                    is_targets = batch["is_targets"]
                    node_idxs = batch["node_idxs"][is_targets][:true_bs].cpu().tolist()
                    scores = yhat["boolean"][is_targets][:true_bs].flatten().float().cpu().tolist()
                    for nid, score in zip(node_idxs, scores):
                        node_idx_to_score[nid] = score

        finally:
            # Restore env vars
            if orig_home is not None:
                os.environ["HOME"] = orig_home
            if orig_userprofile is not None:
                os.environ["USERPROFILE"] = orig_userprofile

    # ------------------------------------------------------------------
    # 6. Group by listing_id, rank by score, return top-k
    # ------------------------------------------------------------------
    from collections import defaultdict
    listing_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for node_idx, score in node_idx_to_score.items():
        listing_id, model_id = node_to_pair[node_idx]
        prob = float(torch.sigmoid(torch.tensor(score)))
        listing_scores[listing_id].append((model_id, prob))

    results = []
    for listing_id in listing_ids:
        ranked = sorted(listing_scores.get(listing_id, []), key=lambda x: x[1], reverse=True)
        results.append({
            "listing_id": listing_id,
            "predictions": [
                {"model_id": mid, "score": round(sc, 4)}
                for mid, sc in ranked[:top_k]
            ],
        })

    # ------------------------------------------------------------------
    # 7. Optionally save to CSV
    # ------------------------------------------------------------------
    if out_path:
        rows = [
            {"listing_id": r["listing_id"], "rank": i + 1,
             "model_id": p["model_id"], "score": p["score"]}
            for r in results
            for i, p in enumerate(r["predictions"])
        ]
        pl.DataFrame(rows).write_csv(out_path)
        print(f"\nSaved {len(rows)} rows → {out_path}")

    for r in results:
        print(f"\n{r['listing_id']}")
        for i, p in enumerate(r["predictions"]):
            print(f"  #{i+1}  model={p['model_id']}  score={p['score']:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Predict model_id for new listings")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--listing-ids", required=True,
                        help="Comma-separated listing UUIDs")
    parser.add_argument("--schema", default="pysqlrustler/schema.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    listing_ids = [x.strip() for x in args.listing_ids.split(",")]
    predict(
        dsn=args.dsn,
        ckpt_path=args.ckpt,
        listing_ids=listing_ids,
        base_schema_path=args.schema,
        top_k=args.top_k,
        out_path=args.out,
    )


if __name__ == "__main__":
    _cli()
