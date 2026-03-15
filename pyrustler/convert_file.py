"""
Convert preprocessed pickle files to JSON for inspection/debugging.

Pure-Python port of rustler/src/convert-file.rs.

Usage:
    python -m pyrustler.convert_file  ~/scratch/pre/rel-f1
    python -m pyrustler.convert_file  ~/scratch/pre/rel-f1/nodes.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

from .common import Edge, Node, SemType, TableType


class _Encoder(json.JSONEncoder):
    """JSON encoder that handles our dataclasses and enums."""

    def default(self, o):
        if isinstance(o, (SemType, TableType)):
            return o.name
        if isinstance(o, (Edge, Node)):
            return asdict(o)
        return super().default(o)


def _convert_nodes(path: Path):
    print(f"Converting {path} …")
    with open(path, "rb") as f:
        nodes = pickle.load(f)

    out_path = path.parent / f"out_{path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(nodes, f, indent=2, cls=_Encoder)
    print(f"Saved {len(nodes):,} nodes → {out_path}")


def _convert_p2f_adj(path: Path):
    print(f"Converting {path} …")
    with open(path, "rb") as f:
        adj = pickle.load(f)

    out_path = path.parent / f"out_{path.stem}.json"
    # adj is list[list[Edge]]
    serialisable = [
        [asdict(e) for e in edges] for edges in adj
    ]
    with open(out_path, "w") as f:
        json.dump({"adj": serialisable}, f, indent=2, cls=_Encoder)
    print(f"Saved adjacency ({len(adj):,} entries) → {out_path}")


def process_file(path: Path):
    """Convert a single .pkl file to JSON."""
    name = path.name
    if name == "nodes.pkl":
        _convert_nodes(path)
    elif name == "p2f_adj.pkl":
        _convert_p2f_adj(path)
    else:
        print(f"Skipping unknown file: {name}")


def main(input_path: str):
    p = Path(input_path).expanduser()

    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.suffix == ".pkl":
                process_file(child)
    elif p.is_file():
        process_file(p)
    else:
        print(f"Invalid path: {p}", file=sys.stderr)
        sys.exit(1)


def _cli():
    parser = argparse.ArgumentParser(
        description="Convert pyrustler pickle files to JSON"
    )
    parser.add_argument("input_path", help="Folder or .pkl file to convert")
    args = parser.parse_args()
    main(args.input_path)


if __name__ == "__main__":
    _cli()
