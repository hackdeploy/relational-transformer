"""
Check what table_name_idx values actually mean
"""
import json
from pathlib import Path
import os
from collections import Counter

DATA_DIR = Path(os.environ.get('USERPROFILE', os.path.expanduser('~'))) / "scratch" / "pre" / "rel-f1"

# Load data
with open(DATA_DIR / "out_nodes.json", encoding='utf-8') as f:
    nodes = json.load(f)

with open(DATA_DIR / "text_map.json", encoding='utf-8') as f:
    text_map = json.load(f)

with open(DATA_DIR / "table_info.json", encoding='utf-8') as f:
    table_info = json.load(f)

idx_to_text = {int(idx): text for text, idx in text_map.items()}

# Check what table_name_idx values appear in nodes
print("Sampling table_name_idx values from different node ranges...\n")

# Sample first 10 nodes
print("First 10 nodes:")
for i in range(10):
    node = nodes[i]
    table_name = idx_to_text.get(node['table_name_idx'], f"Unknown[{node['table_name_idx']}]")
    print(f"  Node {node['node_idx']}: table_name_idx={node['table_name_idx']} → '{table_name}'")

# Look at node 123649 specifically
print(f"\nNode 123649:")
node = nodes[123649]
table_name = idx_to_text[node['table_name_idx']]
print(f"  table_name_idx={node['table_name_idx']} → '{table_name}'")
print(f"  is_task_node={node['is_task_node']}")
print(f"  timestamp={node['timestamp']}")

# Get frequency of table_name_idx values
print(f"\nTop 20 most common table_name_idx values:")
table_idx_counter = Counter(n['table_name_idx'] for n in nodes)
for table_idx, count in table_idx_counter.most_common(20):
    table_name = idx_to_text.get(table_idx, f"Unknown[{table_idx}]")
    print(f"  {table_idx:5d}: {table_name:30s} ({count:6d} nodes)")

# Check if "ferrari" could be a foreign key table reference
print(f"\nChecking actual table boundaries from table_info...")
for table_key, info in sorted(table_info.items()):
    offset = info['node_idx_offset']
    num_nodes = info['num_nodes']
    end = offset + num_nodes - 1
    
    # Check if node 123649 falls in this range
    if offset <= 123649 <= end:
        print(f"  ✓ Node 123649 is in table '{table_key}'")
        print(f"    Range: [{offset}, {end}] ({num_nodes} nodes)")
        
        # Sample a node from this table
        sample_node = nodes[offset]
        sample_table_name = idx_to_text[sample_node['table_name_idx']]
        print(f"    First node's table_name_idx → '{sample_table_name}'")
