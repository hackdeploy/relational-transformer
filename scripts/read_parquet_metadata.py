import pyarrow.parquet as pq
import json
import sys

def read_relational_metadata(parquet_path):
    """
    Extract primary key, foreign key, and time column metadata from a parquet file.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        dict with keys: pkey_col, fkey_col_to_pkey_table, time_col
    """
    # Read parquet file metadata
    parquet_file = pq.ParquetFile(parquet_path)
    metadata = parquet_file.schema_arrow.metadata
    
    if metadata is None:
        print("No metadata found in parquet file")
        return None
    
    # Decode metadata (it's stored as bytes)
    metadata_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
    
    # Extract and parse the relational metadata
    result = {}
    
    # Primary key column
    if 'pkey_col' in metadata_dict:
        result['pkey_col'] = json.loads(metadata_dict['pkey_col'])
    else:
        result['pkey_col'] = None
        
    # Foreign key columns to primary key tables mapping
    if 'fkey_col_to_pkey_table' in metadata_dict:
        result['fkey_col_to_pkey_table'] = json.loads(metadata_dict['fkey_col_to_pkey_table'])
    else:
        result['fkey_col_to_pkey_table'] = {}
        
    # Time column
    if 'time_col' in metadata_dict:
        result['time_col'] = json.loads(metadata_dict['time_col'])
    else:
        result['time_col'] = None
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_parquet_metadata.py <parquet_file_path>")
        print("\nExample:")
        print("  python read_parquet_metadata.py ~/scratch/relbench/rel-f1/db/circuits.parquet")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    
    print(f"Reading metadata from: {parquet_path}\n")
    
    metadata = read_relational_metadata(parquet_path)
    
    if metadata:
        print("Primary Key Column:")
        print(f"  {metadata['pkey_col']}")
        print()
        
        print("Foreign Key Columns → Primary Key Tables:")
        if metadata['fkey_col_to_pkey_table']:
            for fk_col, pk_table in metadata['fkey_col_to_pkey_table'].items():
                print(f"  {fk_col} → {pk_table}")
        else:
            print("  (none)")
        print()
        
        print("Time Column:")
        print(f"  {metadata['time_col']}")
        print()
        
        # Also print raw metadata for debugging
        print("\nAll metadata keys:")
        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.schema_arrow.metadata:
            for k in parquet_file.schema_arrow.metadata.keys():
                print(f"  - {k.decode('utf-8')}")
