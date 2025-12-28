import os
import sys
import yaml
import uuid
import json
import logging
import duckdb
import ast
from pathlib import Path
from typing import List, Dict, Set
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils.neo4j_helper import Neo4jHelper
from backend.utils.qdrant_helper import QdrantHelper
from backend.utils.time_logger import setup_logger
from backend.encoders.transformer_encoder import TransformerEncoder

# Load environment variables
load_dotenv()

logger = setup_logger("insert_graph")

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backend/configs/configs.yml", help="Path to config file")
    args = parser.parse_args()
    
    configs = load_config(args.config)
    insert_config = configs.get("Insert", {})
    batch_size = insert_config.get("Batch_size", 1000)
    limit = insert_config.get("Limit", None)
    resume = insert_config.get("Resume", False)
    
    output_dir = Path("output")
    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"
    
    if not nodes_path.exists():
        logger.error(f"Nodes file not found: {nodes_path}")
        return

    # Initialize Helpers
    try:
        neo4j = Neo4jHelper()
        qdrant = QdrantHelper()
        
        # Initialize Encoder
        encoder_config = configs.get("Encoder", {})
        model_name = encoder_config.get("model_name", "dmis-lab/biobert-v1.1")
        device = encoder_config.get("device", "cuda")
        logger.info(f"Initializing Encoder: {model_name} on {device}...")
        encoder = TransformerEncoder(model_name=model_name, device=device)
        # Ensure model is loaded (optional, embed does it too but good for fail-fast)
        encoder._ensure_model_loaded()
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    collection_name = "kg_lv1_nodes"
    
    # Setup/Clear Data
    if not resume:
        logger.info("Starting fresh (Resume=False). Clearing existing data...")
        neo4j.clear_database()
        neo4j.drop_constraints()
        qdrant.clear_collection(collection_name)
        qdrant.create_collection(collection_name)
        neo4j.create_index()
    else:
        logger.info("Resuming insertion...")
        qdrant.create_collection(collection_name) # Ensure exists
        neo4j.create_index() # Ensure exists

    NAMESPACE_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "fhc.project")
    seen_nodes: Set[str] = set()
    
    # Connect to DuckDB
    # We use an in-memory connection since we are just reading CSVs
    con = duckdb.connect(database=':memory:')

    # 1. Process Nodes
    logger.info(f"Reading nodes from {nodes_path}...")
    
    # Get total count (for tqdm) if possible, or just estimate/skip
    # Counting might be expensive on huge files, but DuckDB is fast.
    # Counting might be expensive on huge files, but DuckDB is fast.
    try:
        # Use read_csv with explicit parameters for robustness
        total_nodes_est = con.execute(f"SELECT count(*) FROM read_csv('{nodes_path}', header=True, delim=',', quote='\"', escape='\"', ignore_errors=True)").fetchone()[0]
    except:
        total_nodes_est = None

    # Prepare Node Cursor
    # Using read_csv with header=True and ignore_errors=True to handle malformed rows
    node_cursor = con.execute(f"SELECT * FROM read_csv('{nodes_path}', header=True, delim=',', quote='\"', escape='\"', ignore_errors=True)")
    
    # Get column names to map rows to dicts
    columns = [desc[0] for desc in node_cursor.description]
    
    logger.info("Inserting nodes...")
    total_neo4j_inserted = 0
    total_qdrant_inserted = 0
    
    # Iterate in batches
    pbar = tqdm(total=total_nodes_est, desc="Nodes") if total_nodes_est else tqdm(desc="Nodes")
    
    while True:
        rows = node_cursor.fetchmany(batch_size)
        if not rows:
            break
            
        # Prepare batches
        neo4j_batch = []
        qdrant_batch = []
        
        # 1. Collect names for embedding generation
        # We need to process names first to batch embed
        current_batch_rows = []
        names_to_embed = []
        
        for row in rows:
            n = dict(zip(columns, row))
            name = n.get("name", "")
            if name: 
                name = str(name).strip()
            if not name:
                continue
            
            # Skip if already seen (global check or check here?)
            # Logic below checks seen_nodes. We need to respect that.
            if name in seen_nodes:
                continue
            
            # Note: We don't add to seen_nodes yet because we might skip if embedding fails? 
            # Actually embedding unlikely to fail.
            # But duplicate names in SAME batch need handling.
            # Let's simple filter unique names within batch?
            # Or just let duplicates be processed and embedded redundantly? 
            # DuckDB fetchmany might return duplicates across batches? Yes.
            
            current_batch_rows.append(n)
            names_to_embed.append(name) # Only valid names

        if not names_to_embed:
             pbar.update(len(rows)) 
             continue

        # 2. Generate Embeddings
        try:
             # embed_to_numpy takes a list of strings
             # Note: duplicates in names_to_embed will be embedded multiple times. 
             # Optimization: unique names? But we need to map back to rows.
             embeddings = encoder.embed_to_numpy(names_to_embed, batch_size=32)
             # embeddings is a numpy array (N, dim)
             embeddings_list = embeddings.tolist()
        except Exception as e:
             logger.error(f"Embedding generation failed for batch: {e}")
             # Skip this batch or continue without embeddings? 
             # Goal is ensure vector exists. So we must skip or fail.
             pbar.update(len(rows))
             continue
             
        # Map Name -> Embedding (taking care of potential duplicates in batch)
        # Actually easier to zip row and embedding if we filter rows first.
        # But we did filtering: current_batch_rows contains rows corresponding to names_to_embed?
        # WAIT: 'current_batch_rows' loop above has check `if name in seen_nodes`.
        # So `current_batch_rows` and `names_to_embed` are 1-to-1 aligned.
        
        # 3. Process Rows with Embeddings
        # We loop through aligned (row, embedding)
        for i, n in enumerate(current_batch_rows):
            pbar.update(1)
            name = names_to_embed[i]
            
            # Re-check seen_nodes just in case of internal batch dupes?
            if name in seen_nodes:
                 continue
            seen_nodes.add(name)
            
            # Deterministic UUID based on name
            node_id = str(uuid.uuid5(NAMESPACE_UUID, name))
            
            # Use GENERATED embedding
            embedding = embeddings_list[i]
            
            level = n.get("level")
            if not level:
                level = "Level 1"
            
            # Handle semantic_types list
            raw_semantic_types = n.get("semantic_types", "")
            semantic_types_list = []
            if raw_semantic_types:
                try:
                    semantic_types_list = ast.literal_eval(raw_semantic_types)
                except:
                    pass
            
            semantic_type = n.get("semantic_type", "")
            if not semantic_type and isinstance(semantic_types_list, list) and len(semantic_types_list) > 0:
                semantic_type = semantic_types_list[0]
                
            # Prepare Neo4j Props
            node_props = {
                "id": node_id,
                "name": name,
                "semantic_type": semantic_type,
                "level": level
            }
            
            # Add Level 2 specific / optional fields
            for key in ["icd", "definition", "cui"]:
                if key in n and n[key] is not None:
                     val = n[key]
                     if isinstance(val, str) and val.strip():
                        node_props[key] = val.strip()
            
            if semantic_types_list:
                node_props["semantic_types"] = semantic_types_list

            neo4j_batch.append(node_props)
            
            # Prepare Qdrant Payload
            # Always insert to Qdrant now
            payload = {
                    "name": name, 
                    "semantic_type": semantic_type,
                    "level": level
            }
            for key in ["icd", "definition", "cui"]:
                if key in node_props:
                    payload[key] = node_props[key]
            if "semantic_types" in node_props:
                payload["semantic_types"] = node_props["semantic_types"] 
                    
            qdrant_batch.append({
                "id": node_id,
                "vector": embedding,
                "payload": payload
            })

            # Check Limit
            if limit and len(seen_nodes) >= limit:
                 break
        
        # Insert Batch to Neo4j
        if neo4j_batch:
            try:
                inserted_count = neo4j.insert_nodes(neo4j_batch)
                total_neo4j_inserted += (inserted_count or 0)
            except Exception as e:
                logger.error(f"Failed to insert Neo4j node batch: {e}")
        
        # Insert Batch to Qdrant
        if qdrant_batch:
            try:
                qdrant.insert_points(collection_name, qdrant_batch)
                total_qdrant_inserted += len(qdrant_batch)
            except Exception as e:
                logger.error(f"Failed to insert Qdrant node batch: {e}")
                
        if limit and len(seen_nodes) >= limit:
            logger.info(f"Reached limit of {limit} nodes. Stopping node ingestion.")
            break

    pbar.close()
    logger.info(f"Node insertion complete: {total_neo4j_inserted} nodes to Neo4j, {total_qdrant_inserted} vectors to Qdrant")

    # 2. Process Edges
    if edges_path.exists():
        logger.info(f"Reading edges from {edges_path}...")
        
        try:
            total_edges_est = con.execute(f"SELECT count(*) FROM read_csv('{edges_path}', header=True, delim=',', quote='\"', escape='\"', ignore_errors=True)").fetchone()[0]
        except:
            total_edges_est = None
            
        edge_cursor = con.execute(f"SELECT * FROM read_csv('{edges_path}', header=True, delim=',', quote='\"', escape='\"', ignore_errors=True)")
        edge_cols = [desc[0] for desc in edge_cursor.description]
        
        logger.info("Inserting edges...")
        total_edges_inserted = 0
        
        pbar = tqdm(total=total_edges_est, desc="Edges") if total_edges_est else tqdm(desc="Edges")
        
        edges_processed = 0
        
        while True:
            rows = edge_cursor.fetchmany(batch_size)
            if not rows:
                break
                
            neo4j_edges = []
            
            for row in rows:
                e = dict(zip(edge_cols, row))
                pbar.update(1)
                
                source = e.get("source", "")
                if source: source = str(source).strip()
                
                target = e.get("target", "")
                if target: target = str(target).strip()
                
                relation = e.get("relation", "")
                if relation: relation = str(relation).strip()
                
                if source in seen_nodes and target in seen_nodes and relation:
                    # Resolve IDs locally without DB lookup (optimization)
                    source_id = str(uuid.uuid5(NAMESPACE_UUID, source))
                    target_id = str(uuid.uuid5(NAMESPACE_UUID, target))
                    
                    sanitized_relation = relation.upper().replace(" ", "_")
                    neo4j_edges.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation": sanitized_relation
                    })
            
            if neo4j_edges:
                try:
                    inserted_count = neo4j.insert_edges(neo4j_edges)
                    total_edges_inserted += (inserted_count or 0)
                except Exception as e:
                    logger.error(f"Failed to insert Neo4j edge batch: {e}")
            
            
            edges_processed += len(rows)
            # Re-check limit logic? 
            # Actually edges limit effectively limits processed edges, not necessarily inserted?
            # User wants limit on inserted? Code says `edges_processed >= limit`.
            # Let's keep it but make it stricter if needed. Since edges are batched, 
            # strict limit inside loop is cleaner but might be overkill.
            # But let's fix it to match nodes logic if possible.
            # However, edges don't have unique ID tracking like nodes.
            # Let's just adjust the break condition to be consistent.
            if limit and total_edges_inserted >= limit:
                 logger.info(f"Reached limit of {limit} inserted edges. Stopping edge ingestion.")
                 break
                
        pbar.close()
        logger.info(f"Edge insertion complete: {total_edges_inserted} relationships created in Neo4j")
    else:
        logger.info("No edges file found.")

    # Final Verification
    logger.info("Verifying insertion...")
    try:
        neo4j_count = neo4j.query("MATCH (n:Level1) RETURN count(n) as count")[0]['count']
        neo4j_rels = neo4j.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        logger.info(f"✅ Neo4j verification: {neo4j_count} nodes, {neo4j_rels} relationships")
    except Exception as e:
        logger.info(f"Neo4j verification check failed: {e}")

    try:
        qdrant_info = qdrant.client.get_collection(collection_name)
        logger.info(f"✅ Qdrant verification: {qdrant_info.points_count} vectors in '{collection_name}'")
    except Exception as e:
        logger.warning(f"Could not verify Qdrant: {e}")

    neo4j.close()
    con.close()
    logger.info("Graph insertion complete.")

if __name__ == "__main__":
    main()
