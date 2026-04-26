"""
pysqlrustler — PostgreSQL-backed port of pyrustler preprocessing.

Reads relational data directly from Postgres tables (instead of relbench
parquet files) and produces the same pickle + JSON artefacts consumed by
the rest of the relational-transformer pipeline.

Sub-modules
-----------
- ``pysqlrustler.pre`` — preprocessing entry point (``python -m pysqlrustler.pre``)
"""
