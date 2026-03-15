"""
Check what each node field references
"""
import json
from pathlib import Path
import os

DATA_DIR = Path(os.environ.get('USERPROFILE', os.path.expanduser('~'))) / "scratch" / "pre" / "rel-f1"

# Load data
with open(DATA_DIR / "out_nodes.json", encoding='utf-8') as f:
    nodes = json.load(f)

with open(DATA_DIR / "text_map.json", encoding='utf-8') as f:
    text_map = json.load(f)

with open(DATA_DIR / "column_index.json", encoding='utf-8') as f:
    column_index = json.load(f)

idx_to_text = {int(idx): text for text, idx in text_map.items()}
idx_to_column = {int(idx): col for col, idx in column_index.items()}

# Sample a regular table node
node = nodes[100000]  # In driver-dnf:Train range

print("Sample node structure:")
print(f"  node_idx: {node['node_idx']}")
print(f"  is_task_node: {node['is_task_node']}")
print(f"  table_name_idx: {node['table_name_idx']} → text_map[{node['table_name_idx']}] = '{idx_to_text[node['table_name_idx']]}'")
print(f"  col_name_idxs: {node['col_name_idxs']}")

print("\nDecoding col_name_idxs:")
for idx in node['col_name_idxs']:
    text = idx_to_text.get(idx, "NOT FOUND")
    in_column_index = idx in idx_to_column
    print(f"  {idx} → text_map: '{text}' | in column_index? {in_column_index}")
    if in_column_index:
        print(f"       → column_index: '{idx_to_column[idx]}'")

print(f"\nDecoding text_values (if any non-zero):")
for i, idx in enumerate(node['text_values'][:5]):  # Just first 5
    if idx != 0:
        text = idx_to_text.get(idx, "NOT FOUND")
        in_column_index = idx in idx_to_column
        print(f"  [{i}] {idx} → text_map: '{text}' | in column_index? {in_column_index}")

print(f"\nDecoding class_value_idx:")
for idx in node['class_value_idx']:
    if idx != -1:
        text = idx_to_text.get(idx, "NOT FOUND")
        in_column_index = idx in idx_to_column
        print(f"  {idx} → text_map: '{text}' | in column_index? {in_column_index}")

print("\n" + "="*60)
print("Summary: How column_index.json is used")
print("="*60)

print("""
column_index.json maps: "column_name of table_name" → text_map index

Usage:
  1. Configuration: Specify target columns for training
     Example: target_idx = column_index["did_not_finish of driver-dnf"]
  
  2. Configuration: Specify columns to drop
     Example: drop_idx = column_index["raceId of results"]
  
  3. NOT used for decoding node data!

All node fields use text_map indices directly:
  - node['table_name_idx'] → text_map
  - node['col_name_idxs'] → text_map
  - node['text_values'] → text_map
  - node['class_value_idx'] → text_map
""")
