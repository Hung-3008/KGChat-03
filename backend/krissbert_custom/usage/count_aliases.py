import duckdb
import os
import sys

# Path to UMLS file
json_path = "/media/hung/data1/codes/projects/FHC/backend/krissbert_custom/umls_full.json"

print(f"Counting aliases in {json_path}...")

try:
    con = duckdb.connect()
    
    # Count raw concepts
    count_concepts = con.execute(f"SELECT count(*) FROM read_json_auto('{json_path}')").fetchone()[0]
    print(f"Total concepts (rows): {count_concepts}")
    
    # Count unnested aliases
    query = f"SELECT count(*) FROM (SELECT unnest(aliases) FROM read_json_auto('{json_path}'))"
    count_aliases = con.execute(query).fetchone()[0]
    print(f"Total aliases (unnested): {count_aliases}")
    
except Exception as e:
    print(f"Error: {e}")
