"""
Quick check to see the range of valid node indices.
"""
import json
from pathlib import Path
import os

DATA_DIR = Path(os.environ.get('USERPROFILE', os.path.expanduser('~'))) / "scratch" / "pre" / "rel-f1"

# Load nodes
with open(DATA_DIR / "out_nodes.json") as f:
    nodes = json.load(f)

# Get node_idx range
node_indices = [n['node_idx'] for n in nodes]
print(f"Total nodes: {len(nodes)}")
print(f"Min node_idx: {min(node_indices)}")
print(f"Max node_idx: {max(node_indices)}")
print(f"\nFirst 10 node_idxs: {node_indices[:10]}")
print(f"Last 10 node_idxs: {node_indices[-10:]}")

# Check if node_idx matches array position
matches_position = all(n['node_idx'] == i for i, n in enumerate(nodes))
print(f"\nnode_idx == array_position for all nodes: {matches_position}")

# Check for 123649
has_123649 = 123649 in node_indices
print(f"\n123649 exists in data: {has_123649}")
if has_123649:
    pos = node_indices.index(123649)
    print(f"  Position: {pos}")
else:
    print(f"  123649 is BEYOND the valid range!")
    print(f"  This is likely a batch padding or invalid sentinel value")
