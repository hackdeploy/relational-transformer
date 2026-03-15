"""
Debug column index lookup issues
"""
import json
from pathlib import Path
import os

DATA_DIR = Path(os.environ.get('USERPROFILE', os.path.expanduser('~'))) / "scratch" / "pre" / "rel-f1"

# Load data
print("Loading data...")
with open(DATA_DIR / "out_nodes.json", encoding='utf-8') as f:
    nodes = json.load(f)

with open(DATA_DIR / "text_map.json", encoding='utf-8') as f:
    text_map = json.load(f)

with open(DATA_DIR / "column_index.json", encoding='utf-8') as f:
    column_index = json.load(f)

print(f"Loaded {len(nodes)} nodes, {len(text_map)} text entries, {len(column_index)} columns")

# Get node 123649
node = nodes[123649]
print(f"\nNode 123649:")
print(f"  table_name_idx: {node['table_name_idx']}")
print(f"  col_name_idxs: {node['col_name_idxs']}")

# Reverse mappings
idx_to_text = {int(idx): text for text, idx in text_map.items()}
idx_to_column = {int(idx): col for col, idx in column_index.items()}

# Check table name
table_name_idx = node['table_name_idx']
table_name = idx_to_text.get(table_name_idx, f"NOT_FOUND[{table_name_idx}]")
print(f"\nTable name (idx={table_name_idx}): {table_name}")

# Check each column
print(f"\nColumn indices to look up: {node['col_name_idxs']}")
for col_idx in node['col_name_idxs']:
    if col_idx in idx_to_column:
        print(f"  ✓ {col_idx}: {idx_to_column[col_idx]}")
    else:
        print(f"  ✗ {col_idx}: NOT FOUND in column_index")

# Search for similar indices
print(f"\nSearching for indices near 2317...")
nearby = {idx: col for idx, col in idx_to_column.items() if 2310 <= idx <= 2325}
for idx in sorted(nearby.keys()):
    print(f"  {idx}: {nearby[idx]}")

# Check if text_map has these indices
print(f"\nSearching text_map for indices 2317, 2319...")
for idx in [2317, 2319]:
    if idx in idx_to_text:
        print(f"  ✓ {idx} in text_map: {idx_to_text[idx]}")
    else:
        print(f"  ✗ {idx} NOT in text_map")

# Show range of indices in each mapping
print(f"\n=== Index Ranges ===")
text_indices = [int(idx) for idx in text_map.values()]
col_indices = [int(idx) for idx in column_index.values()]
print(f"text_map indices: {min(text_indices)} to {max(text_indices)}")
print(f"column_index indices: {min(col_indices)} to {max(col_indices)}")

# Check what's the difference
print(f"\n=== Checking for overlap ===")
text_set = set(text_indices)
col_set = set(col_indices)
print(f"Are column indices a subset of text indices? {col_set.issubset(text_set)}")
print(f"Indices in text_map but not in column_index: {len(text_set - col_set)}")
