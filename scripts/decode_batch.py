"""
Decode batch indices to actual table values using JSON files.
Shows how to map from node_idxs to actual row records.
"""
import json
import os
from pathlib import Path

# Configuration
DATA_DIR = Path(os.environ.get('USERPROFILE', os.path.expanduser('~'))) / "scratch" / "pre" / "rel-f1"
# Or use Google Drive path:
# DATA_DIR = Path("G:/My Drive/Colab_Data/scratch/pre/rel-f1")

def load_metadata(data_dir):
    """Load all metadata JSON files."""
    print(f"Loading from: {data_dir}")
    
    with open(data_dir / "out_nodes.json", encoding='utf-8') as f:
        nodes = json.load(f)
    
    with open(data_dir / "text_map.json", encoding='utf-8') as f:
        text_map = json.load(f)
    
    with open(data_dir / "column_index.json", encoding='utf-8') as f:
        column_index = json.load(f)
    
    with open(data_dir / "table_info.json", encoding='utf-8') as f:
        table_info = json.load(f)
    
    # Reverse mappings for easier lookup
    idx_to_text = {int(idx): text for text, idx in text_map.items()}
    idx_to_column = {int(idx): col_name for col_name, idx in column_index.items()}
    
    print(f"Loaded {len(nodes)} nodes")
    
    # Check if node_idx matches array position (usually true for this dataset)
    matches = all(node['node_idx'] == i for i, node in enumerate(nodes[:100]))
    
    if matches:
        print("✓ node_idx == array position (can use direct indexing)")
        node_idx_to_position = None  # Direct indexing works
    else:
        print("Building node_idx lookup map...")
        node_idx_to_position = {node['node_idx']: i for i, node in enumerate(nodes)}
    
    return nodes, idx_to_text, idx_to_column, table_info, node_idx_to_position


def decode_node(node, idx_to_text, idx_to_column):
    """Decode a single node to a human-readable record."""
    record = {
        'node_idx': node['node_idx'],
        'table_name': idx_to_text.get(node['table_name_idx'], f"Unknown[{node['table_name_idx']}]"),
        'is_task_node': node['is_task_node'],
        'columns': {}
    }
    
    # Decode each column value
    for i, col_name_idx in enumerate(node['col_name_idxs']):
        # col_name_idx is an index into text_map, not column_index!
        col_name = idx_to_text.get(col_name_idx, f"Unknown[{col_name_idx}]")
        sem_type = node['sem_types'][i]
        
        # Get value based on semantic type
        if sem_type == 'Number':
            value = node['number_values'][i]
        elif sem_type == 'Text':
            text_idx = node['text_values'][i]
            value = idx_to_text.get(text_idx, f"TextIdx[{text_idx}]")
        elif sem_type == 'DateTime':
            value = node['datetime_values'][i]
        elif sem_type == 'Boolean':
            value = bool(node['boolean_values'][i])
        else:
            value = f"Unknown type: {sem_type}"
        
        record['columns'][col_name] = {
            'value': value,
            'type': sem_type
        }
    
    # Add class/target value if present
    if node['class_value_idx']:
        class_idx = node['class_value_idx'][0]
        record['class_value'] = idx_to_text.get(class_idx, f"ClassIdx[{class_idx}]")
    
    return record


def decode_batch_to_records(node_idxs, data_dir=DATA_DIR):
    """
    Convert a list of node_idxs from a batch to actual table records.
    
    Args:
        node_idxs: List of node indices from batch (e.g., batch_dict['node_idxs'])
        data_dir: Path to directory containing JSON files
        
    Returns:
        List of decoded records with actual values
    """
    # Load metadata
    nodes, idx_to_text, idx_to_column, table_info, node_idx_to_position = load_metadata(data_dir)
    
    # Decode each node
    records = []
    for node_idx in node_idxs:
        # Try direct indexing first (works when node_idx == array position)
        if node_idx_to_position is None:
            # Direct indexing
            if 0 <= node_idx < len(nodes):
                node = nodes[node_idx]
                record = decode_node(node, idx_to_text, idx_to_column)
                records.append(record)
            else:
                records.append({'error': f'node_idx {node_idx} out of range [0, {len(nodes)-1}]'})
        else:
            # Use lookup map
            array_position = node_idx_to_position.get(node_idx)
            if array_position is not None:
                node = nodes[array_position]
                record = decode_node(node, idx_to_text, idx_to_column)
                records.append(record)
            else:
                records.append({'error': f'node_idx {node_idx} not found in data'})
    
    return records


def print_record(record, indent=0):
    """Pretty print a decoded record."""
    prefix = "  " * indent
    
    if 'error' in record:
        print(f"{prefix}ERROR: {record['error']}")
        return
    
    print(f"{prefix}Node {record['node_idx']}: {record['table_name']}")
    if record['is_task_node']:
        print(f"{prefix}  [TASK NODE]")
    
    if 'class_value' in record:
        print(f"{prefix}  Target: {record['class_value']}")
    
    print(f"{prefix}  Columns:")
    for col_name, col_info in record['columns'].items():
        print(f"{prefix}    {col_name}: {col_info['value']} ({col_info['type']})")


if __name__ == "__main__":
    # Example: Load a saved batch and decode it
    import pickle
    
    # Check if we have a sample batch
    batch_file = Path(__file__).parent.parent / "sample_batch.pkl"
    if not batch_file.exists():
        print(f"No sample batch found at {batch_file}")
        print("Run save_sample_batch.py first to create one.")
        exit(1)
    
    # Load the batch
    with open(batch_file, 'rb') as f:
        batch_dict = pickle.load(f)
    
    print("Batch structure:")
    for key in batch_dict.keys():
        print(f"  {key}")
    
    # Get node indices (flatten if it's a 2D array)
    node_idxs = batch_dict['node_idxs']
    if hasattr(node_idxs, 'flatten'):
        node_idxs = node_idxs.flatten()
    
    # Take first 5 non-padding nodes
    is_padding = batch_dict['is_padding']
    if hasattr(is_padding, 'flatten'):
        is_padding = is_padding.flatten()
    
    valid_indices = [idx for idx, is_pad in zip(node_idxs, is_padding) if not is_pad][:5]
    
    print(f"\n{'='*60}")
    print(f"Decoding first {len(valid_indices)} non-padding nodes...")
    print(f"{'='*60}\n")
    
    # Decode to records
    records = decode_batch_to_records(valid_indices, DATA_DIR)
    
    # Print results
    for i, record in enumerate(records):
        print(f"\nRecord {i+1}:")
        print("-" * 60)
        print_record(record)
