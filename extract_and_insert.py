"""
extract_and_insert.py
Reads chunks_export.json, extracts entities + edges using NodeExtractor/EdgeExtractor,
resolves CUI via Krissbert, then inserts Entity nodes and edges into Neo4j and Qdrant.
Links each Entity to its source Chunk via MENTIONED_IN relationship.
"""

import os
import sys
import yaml
import json
import uuid
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.neo4j_helper import Neo4jHelper
from backend.utils.qdrant_helper import QdrantHelper
from backend.utils.time_logger import TimeLogger, setup_logger, Timer

# Load environment variables
load_dotenv()

logger = setup_logger("extract_and_insert")

NAMESPACE_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "fhc.project")


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_chunks(chunks_path: str) -> List[Dict]:
    """Load chunks from chunks_export.json"""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_cui(entity_linker, nodes: List[Dict]) -> List[Dict]:
    """
    Use Krissbert to resolve CUI for each entity node.
    Adds 'cui' field to each node dict.
    """
    if not entity_linker or not nodes:
        for n in nodes:
            n['cui'] = ""
        return nodes

    # Prepare Krissbert input
    krissbert_input = []
    for node in nodes:
        krissbert_input.append({
            "mention": node.get("name", ""),
            "context_left": node.get("context_left", ""),
            "context_right": node.get("context_right", ""),
        })

    try:
        results = entity_linker.predict(krissbert_input, top_k=1)

        for i, res in enumerate(results):
            if i >= len(nodes):
                break
            candidates = res.get("candidates", [])
            # Take top candidate with score >= 0.85
            top = next((c for c in candidates if c.get("score", 0) >= 0.85), None)
            nodes[i]['cui'] = top.get("cui", "") if top else ""

        # Fill remaining nodes without results
        for i in range(len(results), len(nodes)):
            nodes[i]['cui'] = ""

    except Exception as e:
        logger.warning(f"Krissbert CUI resolution failed: {e}")
        for n in nodes:
            if 'cui' not in n:
                n['cui'] = ""

    return nodes


def main():
    parser = argparse.ArgumentParser(description="Extract entities/edges from chunks and insert into Neo4j + Qdrant")
    parser.add_argument("--config", default="/home/nguyenthang/Tài liệu/import_qdrant/KGChat-03/backend/configs/configs.yml", help="Path to config file")
    parser.add_argument("--chunks", default="/home/nguyenthang/Tài liệu/import_qdrant/KGChat-03/chunks_export.json", help="Path to chunks_export.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process")
    parser.add_argument("--resume", action="store_true", help="Resume from previously processed chunks")
    parser.add_argument("--clear", action="store_true", help="Clear existing Entity data before inserting")
    args = parser.parse_args()

    config_path = args.config
    configs = load_config(config_path)

    insert_config = configs.get("Insert", {})
    batch_size = insert_config.get("Batch_size", 1000)

    # ── Load Chunks ──────────────────────────────────────────────────────
    logger.info(f"Loading chunks from {args.chunks}...")
    all_chunks = load_chunks(args.chunks)
    logger.info(f"Loaded {len(all_chunks)} chunks")

    # ── Output Directory ─────────────────────────────────────────────────
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    log_path = output_dir / "processed_chunks.txt"
    nodes_csv_path = output_dir / "nodes.csv"
    edges_csv_path = output_dir / "edges.csv"

    # ── Resume Logic ─────────────────────────────────────────────────────
    processed_chunks = set()
    if args.resume and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            processed_chunks = set(line.strip() for line in f if line.strip())
        logger.info(f"Resuming: {len(processed_chunks)} chunks already processed")
    elif not args.resume:
        for p in [log_path, nodes_csv_path, edges_csv_path]:
            if p.exists():
                p.unlink()

    # Filter out already-processed chunks
    chunks_to_process = [c for c in all_chunks if c.get("chunk_id", "") not in processed_chunks]

    if args.limit:
        chunks_to_process = chunks_to_process[:args.limit]

    total = len(chunks_to_process)
    if total == 0:
        logger.info("No chunks to process.")
        return

    logger.info(f"Will process {total} chunks")

    # ── Initialize Components ────────────────────────────────────────────
    time_logger = TimeLogger(output_dir / "time_log.csv")
    extractor = GraphExtractor(config_path=config_path, time_logger=time_logger)

    # Get Krissbert entity linker from EdgeExtractor
    entity_linker = extractor.edge_extractor.entity_linker

    try:
        neo4j = Neo4jHelper()
        qdrant = QdrantHelper()
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j/Qdrant: {e}")
        return

    collection_name = "kg_entities"

    # ── Setup ────────────────────────────────────────────────────────────
    if args.clear and not args.resume:
        logger.info("Clearing existing Entity data...")
        neo4j.query("MATCH (e:Entity) DETACH DELETE e")
        qdrant.clear_collection(collection_name)

    qdrant.create_collection(collection_name)
    neo4j.create_entity_index()

    # ── Process Each Chunk ───────────────────────────────────────────────
    total_nodes_inserted = 0
    total_edges_inserted = 0
    total_qdrant_inserted = 0

    for idx, chunk in enumerate(tqdm(chunks_to_process, desc="Extracting & Inserting")):
        chunk_id = chunk.get("chunk_id", f"unknown_{idx}")
        content = chunk.get("content", "")

        if not content.strip():
            logger.warning(f"Skipping empty chunk: {chunk_id}")
            _log_processed(log_path, chunk_id)
            continue

        logger.info(f"[{idx+1}/{total}] Processing chunk: {chunk_id}")

        try:
            # ── Step 1: Extract Nodes ────────────────────────────────
            with Timer(time_logger, chunk_id, "Node Extraction"):
                nodes = extractor.node_extractor.extract(content, file_name=chunk_id)

            logger.info(f"  → Extracted {len(nodes)} entities")

            # ── Step 2: Resolve CUI via Krissbert ────────────────────
            if nodes:
                with Timer(time_logger, chunk_id, "CUI Resolution"):
                    nodes = resolve_cui(entity_linker, nodes)

            # ── Step 3: Extract Edges (LLM only, skip Level 2) ───────
            with Timer(time_logger, chunk_id, "Edge Extraction"):
                edges_result, _ = extractor.edge_extractor.extract(
                    text=content, nodes=nodes, file_name=chunk_id
                )

            all_edges = []
            if edges_result and edges_result.edges:
                for edge in edges_result.edges:
                    edge_dict = edge.dict() if hasattr(edge, 'dict') else edge.model_dump()
                    # Skip REF_TO edges (Level 2 remnants)
                    if edge_dict.get('relation') == 'REF_TO':
                        continue
                    all_edges.append(edge_dict)

            logger.info(f"  → Extracted {len(all_edges)} edges")

            # ── Step 4: Save CSV backup ──────────────────────────────
            if nodes:
                extractor.save_nodes(nodes, nodes_csv_path, append=True)
            if all_edges:
                extractor.save_edges(all_edges, edges_csv_path, append=True)

            # ── Step 5: Insert into Neo4j + Qdrant ───────────────────
            entity_ids = []
            if nodes:
                n_neo4j, n_qdrant, entity_ids = _insert_entities(
                    neo4j, qdrant, collection_name, nodes, batch_size
                )
                total_nodes_inserted += n_neo4j
                total_qdrant_inserted += n_qdrant

            if all_edges:
                n_edges = _insert_edges(neo4j, nodes, all_edges, batch_size)
                total_edges_inserted += n_edges

            # ── Step 6: Link Entities → Chunk ────────────────────────
            if entity_ids:
                neo4j.link_entities_to_chunk(entity_ids, chunk_id)

            # ── Log processed chunk ──────────────────────────────────
            _log_processed(log_path, chunk_id)

        except Exception as e:
            import traceback
            logger.error(f"Failed to process chunk {chunk_id}: {e}\n{traceback.format_exc()}")
            continue

    # ── Final Summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Extraction & Insertion Complete!")
    logger.info(f"  Entities → Neo4j: {total_nodes_inserted}")
    logger.info(f"  Vectors → Qdrant: {total_qdrant_inserted}")
    logger.info(f"  Edges → Neo4j: {total_edges_inserted}")

    # ── Verification ─────────────────────────────────────────────────────
    try:
        entity_count = neo4j.query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']
        rel_count = neo4j.query("MATCH (:Entity)-[r]->() RETURN count(r) as count")[0]['count']
        mentioned_count = neo4j.query("MATCH (:Entity)-[r:MENTIONED_IN]->(:Chunk) RETURN count(r) as count")[0]['count']
        logger.info(f"  ✅ Neo4j: {entity_count} entities, {rel_count} relationships, {mentioned_count} MENTIONED_IN links")
    except Exception as e:
        logger.warning(f"  Could not verify Neo4j: {e}")

    try:
        qdrant_info = qdrant.client.get_collection(collection_name)
        logger.info(f"  ✅ Qdrant: {qdrant_info.points_count} vectors in '{collection_name}'")
    except Exception as e:
        logger.warning(f"  Could not verify Qdrant: {e}")

    neo4j.close()
    logger.info("Done.")


def _insert_entities(neo4j: Neo4jHelper, qdrant: QdrantHelper, collection_name: str,
                     nodes: List[Dict], batch_size: int) -> tuple:
    """Insert Entity nodes into Neo4j and Qdrant. Returns (neo4j_count, qdrant_count, entity_ids)."""

    # Deduplicate by name
    unique_nodes = {}
    for n in nodes:
        name = n.get("name", "").strip()
        if not name or name in unique_nodes:
            continue

        node_id = str(uuid.uuid5(NAMESPACE_UUID, name))
        unique_nodes[name] = {
            "id": node_id,
            "name": name,
            "semantic_type": n.get("semantic_type", ""),
            "cui": n.get("cui", ""),
            "vector": n.get("embedding", []),
        }

    node_list = list(unique_nodes.values())
    entity_ids = [n["id"] for n in node_list]
    neo4j_count = 0
    qdrant_count = 0

    for i in range(0, len(node_list), batch_size):
        batch = node_list[i:i + batch_size]

        # Neo4j
        neo4j_nodes = [{"id": n["id"], "name": n["name"], "semantic_type": n["semantic_type"], "cui": n["cui"]} for n in batch]
        try:
            inserted = neo4j.insert_entity_nodes(neo4j_nodes)
            neo4j_count += (inserted or 0)
        except Exception as e:
            logger.error(f"Failed Neo4j entity batch: {e}")

        # Qdrant
        qdrant_points = []
        for n in batch:
            if n["vector"] and len(n["vector"]) > 0:
                qdrant_points.append({
                    "id": n["id"],
                    "vector": n["vector"],
                    "payload": {
                        "name": n["name"],
                        "semantic_type": n["semantic_type"],
                        "cui": n["cui"],
                    },
                })

        if qdrant_points:
            try:
                qdrant.insert_points(collection_name, qdrant_points)
                qdrant_count += len(qdrant_points)
            except Exception as e:
                logger.error(f"Failed Qdrant batch: {e}")

    return neo4j_count, qdrant_count, entity_ids


def _insert_edges(neo4j: Neo4jHelper, nodes: List[Dict], edges: List[Dict], batch_size: int) -> int:
    """Insert edges between Entity nodes in Neo4j."""

    # Build name → id map
    name_to_id = {}
    for n in nodes:
        name = n.get("name", "").strip()
        if name:
            name_to_id[name] = str(uuid.uuid5(NAMESPACE_UUID, name))

    valid_edges = []
    for e in edges:
        source = e.get("source", "").strip()
        target = e.get("target", "").strip()
        relation = e.get("relation", "").strip()

        source_id = name_to_id.get(source)
        target_id = name_to_id.get(target)

        if source_id and target_id and relation:
            sanitized = relation.strip().upper().replace(" ", "_")
            valid_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "relation": sanitized,
            })

    if not valid_edges:
        return 0

    total = 0
    for i in range(0, len(valid_edges), batch_size):
        batch = valid_edges[i:i + batch_size]
        try:
            inserted = neo4j.insert_entity_edges(batch)
            total += (inserted or 0)
        except Exception as e:
            logger.error(f"Failed Neo4j edge batch: {e}")

    return total


def _log_processed(log_path: Path, chunk_id: str):
    """Append chunk_id to the processed log."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{chunk_id}\n")


if __name__ == "__main__":
    main()
