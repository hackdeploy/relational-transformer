"""
Predict price and deal score for listings using a trained price checkpoint.

Usage:
    python scripts/predict_price_motolote.py \
        --dsn  "postgresql://..." \
        --ckpt "$HOME/scratch/ckpts/motolote_price_best.pt" \
        --listing-ids "uuid-1,uuid-2"
"""
import argparse, math, os
from pathlib import Path

import psycopg2

from pysqlrustler.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent

DSN    = "postgresql://postgres.zetsjztchfmihqfzobok:M%40satepe123@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
CKPT   = f"{os.environ.get('HOME', '~')}/scratch/ckpts/motolote_price_best.pt"
SCHEMA = str(REPO_ROOT / "pysqlrustler" / "schema_price.json")

DEAL_THRESHOLDS = {
    "great":      0.80,  # >20% below expected
    "good":       0.95,  # 5-20% below expected
    "fair":       1.10,  # within 10% of expected
    "overpriced": float("inf"),
}


def _deal_category(actual: float, predicted: float) -> tuple[str, float]:
    ratio = actual / predicted
    pct   = round((1 - ratio) * 100, 1)
    for label, threshold in DEAL_THRESHOLDS.items():
        if ratio < threshold:
            return label, pct
    return "overpriced", pct


def _fetch_actual_prices(dsn: str, listing_ids: list[str]) -> dict[str, float | None]:
    if not listing_ids:
        return {}
    ids_sql = ", ".join(f"'{lid}'" for lid in listing_ids)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, price FROM listings WHERE id IN ({ids_sql})")
            return {row[0]: float(row[1]) if row[1] is not None else None for row in cur.fetchall()}


def predict_price_for_listing_query(
    listing_query: str,
    out_path: str | None = None,
    ckpt_path: str | None = None,
) -> list[dict]:
    """
    listing_query: SQL returning listing IDs, e.g.
        'SELECT id FROM listings WHERE model_id IS NULL LIMIT 100'

    Returns one result per listing with predicted_price, actual_price,
    deal_score, and deal_category.
    """
    results = predict(
        dsn=DSN,
        ckpt_path=ckpt_path or CKPT,
        base_schema_path=SCHEMA,
        task_table="listing_prices",
        target_column="log_price",
        foreign_keys={"listing_id": "listings"},
        prediction_sql=(
            f"SELECT id AS listing_id, 0.0::float AS log_price "
            f"FROM listings "
            f"WHERE id IN ({listing_query})"
        ),
        group_by="listing_id",
        id_columns=["listing_id"],
        top_k=1,
        task_type="regression",
        out_path=None,
    )

    listing_ids  = [r["group_value"] for r in results]
    actual_prices = _fetch_actual_prices(DSN, listing_ids)

    enriched = []
    for r in results:
        lid            = r["group_value"]
        log_price_pred = r["predictions"][0]["score"]
        predicted      = math.exp(log_price_pred)
        actual         = actual_prices.get(lid)

        row: dict = {
            "listing_id":      lid,
            "predicted_price": round(predicted, 2),
            "actual_price":    round(actual, 2) if actual else None,
        }

        if actual:
            category, pct_below = _deal_category(actual, predicted)
            row["deal_category"] = category
            row["pct_below_market"] = pct_below

        enriched.append(row)
        actual_str = f"  actual=${actual:,.0f}  {row.get('deal_category','?')} ({row.get('pct_below_market','?')}% below)" if actual else ""
        print(f"{lid}  predicted=${predicted:,.0f}{actual_str}")

    if out_path:
        import polars as pl
        pl.DataFrame(enriched).write_csv(out_path)
        print(f"\nSaved → {out_path}")

    return enriched


def predict_price_for_listings(
    listing_ids: list[str],
    out_path: str | None = None,
    ckpt_path: str | None = None,
) -> list[dict]:
    ids_sql = ", ".join(f"'{lid}'" for lid in listing_ids)
    return predict_price_for_listing_query(
        listing_query=f"SELECT id FROM listings WHERE id IN ({ids_sql})",
        out_path=out_path,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn",         default=DSN)
    parser.add_argument("--ckpt",        default=CKPT)
    parser.add_argument("--listing-ids", required=True)
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    predict_price_for_listings(
        listing_ids=[x.strip() for x in args.listing_ids.split(",")],
        out_path=args.out,
        ckpt_path=args.ckpt,
    )
