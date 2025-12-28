import duckdb

def check():
    con = duckdb.connect()
    # Count total nodes
    total = con.execute("SELECT count(*) FROM read_csv('output/nodes.csv', header=True, delim=',', quote='\"', escape='\"', ignore_errors=True)").fetchone()[0]
    
    # Count nodes with embeddings (not empty and not '[]')
    # SQL filter: embedding IS NOT NULL AND length(embedding) > 2
    # Note: '[]' has length 2.
    with_embedding = con.execute("SELECT count(*) FROM read_csv('output/nodes.csv', header=True, delim=',', quote='\"', escape='\"', ignore_errors=True) WHERE embedding IS NOT NULL AND length(embedding) > 2").fetchone()[0]
    
    print(f"Total rows: {total}")
    print(f"Rows with embeddings: {with_embedding}")

if __name__ == "__main__":
    check()
