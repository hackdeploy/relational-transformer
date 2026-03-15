"""
Verify what's actually at node_idx 123649
"""
import json
from pathlib import Path
import os

DATA_DIR = Path(os.environ.get('USERPROFILE', os.path.expanduser('~'))) / "scratch" / "pre" / "rel-f1"

# Load nodes
with open(DATA_DIR / "out_nodes.json") as f:
    nodes = json.load(f)

print(DATA_DIR)
print(nodes[123649])

print(f"Total nodes loaded: {len(nodes)}")
print(f"Checking position 123649...")

if 123649 < len(nodes):
    node = nodes[123649]
    print(f"\nNode at array position 123649:")
    print(f"  node_idx field: {node['node_idx']}")
    print(f"  is_task_node: {node['is_task_node']}")
    print(f"  table_name_idx: {node['table_name_idx']}")
    print(f"  Full node: {node}")
    
    # Check if node_idx matches position
    if node['node_idx'] == 123649:
        print(f"\n✓ node_idx field matches array position")
    else:
        print(f"\n✗ MISMATCH! node_idx={node['node_idx']} but position=123649")
else:
    print(f"\n✗ Position 123649 is OUT OF BOUNDS! Array only has {len(nodes)} nodes")

# Also check what file size we have
json_file = DATA_DIR / "out_nodes.json"
file_size_mb = json_file.stat().st_size / (1024 * 1024)
print(f"\nFile size: {file_size_mb:.2f} MB")
