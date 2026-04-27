"""
Predict best model_id for new listings using the trained motolote checkpoint.

Usage:
    python scripts/predict_motolote.py \
        --dsn  "postgresql://..." \
        --ckpt "$HOME/scratch/ckpts/motolote_best.pt" \
        --listing-ids "uuid-1,uuid-2"
"""
import argparse, os
from pysqlrustler.predict import predict

DSN          = "postgresql://postgres.zetsjztchfmihqfzobok:M%40satepe123@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
CKPT         = f"{os.environ.get('HOME', '~')}/scratch/ckpts/motolote_best.pt"
SCHEMA       = "pysqlrustler/schema.json"


def predict_for_listings(listing_ids: list[str], top_k: int = 3, out_path=None):
    ids_sql = ", ".join(f"'{lid}'" for lid in listing_ids)
    return predict(
        dsn=DSN,
        ckpt_path=CKPT,
        base_schema_path=SCHEMA,
        task_table="listing_model_matches",
        target_column="label",
        foreign_keys={"listing_id": "listings", "model_id": "models"},
        prediction_sql=(
            f"SELECT l.id AS listing_id, m.id AS model_id, "
            f"false::boolean AS label "
            f"FROM listings l CROSS JOIN models m "
            f"WHERE l.id IN ({ids_sql})"
        ),
        group_by="listing_id",
        id_columns=["listing_id", "model_id"],
        top_k=top_k,
        out_path=out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn",         default=DSN)
    parser.add_argument("--ckpt",        default=CKPT)
    parser.add_argument("--listing-ids", required=True)
    parser.add_argument("--top-k",       type=int, default=3)
    parser.add_argument("--out",         default="predictions.csv")
    args = parser.parse_args()

    predict_for_listings(
        listing_ids=[x.strip() for x in args.listing_ids.split(",")],
        top_k=args.top_k,
        out_path=args.out,
    )
