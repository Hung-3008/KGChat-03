import os
import sys
import yaml
import csv
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils.neo4j_helper import Neo4jHelper
from backend.utils.qdrant_helper import QdrantHelper
from backend.utils.time_logger import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger("insert_graph")

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def read_csv(filepath: Path) -> List[Dict]:
    if not filepath.exists():
        return []
    with filepath.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

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

    # 1. Process Nodes
    logger.info("Reading nodes...")
    raw_nodes = read_csv(nodes_path)
    logger.info(f"Found {len(raw_nodes)} raw nodes.")
    
    unique_nodes = {}
    NAMESPACE_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "fhc.project")
    
    for n in raw_nodes:
        name = n.get("name", "").strip()
        if not name:
            continue
        if name not in unique_nodes:
            # Deterministic UUID based on name
            node_id = str(uuid.uuid5(NAMESPACE_UUID, name))
            
            # Parse embedding
            embedding_str = n.get("embedding", "[]")
            try:
                embedding = json.loads(embedding_str)
            except:
                embedding = []
                
            unique_nodes[name] = {
                "id": node_id,
                "name": name,
                "semantic_type": n.get("semantic_type", ""),
                "vector": embedding
            }
            
    logger.info(f"Identified {len(unique_nodes)} unique nodes.")
    
    # Convert to list for batching
    node_list = list(unique_nodes.values())
    
    if limit:
        logger.info(f"Limiting to {limit} nodes/edges for testing.")
        node_list = node_list[:limit]
    
    # Insert Nodes to Neo4j and Qdrant
    logger.info("Inserting nodes...")
    for i in tqdm(range(0, len(node_list), batch_size), desc="Nodes"):
        batch = node_list[i : i + batch_size]
        
        # Neo4j Batch
        neo4j_nodes = [{"id": n["id"], "name": n["name"], "semantic_type": n["semantic_type"]} for n in batch]
        neo4j.insert_nodes(neo4j_nodes)
        
        # Qdrant Batch
        qdrant_points = []
        for n in batch:
            if n["vector"] and len(n["vector"]) > 0:
                qdrant_points.append({
                    "id": n["id"],
                    "vector": n["vector"],
                    "payload": {"name": n["name"], "semantic_type": n["semantic_type"]}
                })
        if qdrant_points:
            qdrant.insert_points(collection_name, qdrant_points)

    # 2. Process Edges
    if edges_path.exists():
        logger.info("Reading edges...")
        raw_edges = read_csv(edges_path)
        logger.info(f"Found {len(raw_edges)} edges.")
        
        valid_edges = []
        for e in raw_edges:
            source = e.get("source", "").strip()
            target = e.get("target", "").strip()
            relation = e.get("relation", "").strip()
            
            if source in unique_nodes and target in unique_nodes and relation:
                # Sanitize relation: UPPER_SNAKE_CASE
                sanitized_relation = relation.strip().upper().replace(" ", "_")
                valid_edges.append({
                    "source_id": unique_nodes[source]["id"],
                    "target_id": unique_nodes[target]["id"],
                    "relation": sanitized_relation
                })
        
        logger.info(f"Identified {len(valid_edges)} valid edges (both nodes exist).")
        
        if limit:
             valid_edges = valid_edges[:limit]
             logger.info(f"Limiting edges to {limit}...")
        
        # Insert Edges to Neo4j
        logger.info("Inserting edges...")
        for i in tqdm(range(0, len(valid_edges), batch_size), desc="Edges"):
            batch = valid_edges[i : i + batch_size]
            neo4j.insert_edges(batch)
    else:
        logger.info("No edges file found.")

    neo4j.close()
    logger.info("Graph insertion complete.")

if __name__ == "__main__":
    main()
