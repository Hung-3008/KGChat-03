"""
creat_graph_v1.py
Reads parsed_structure_final.json, flattens all chunks from the hierarchical structure,
then extracts entities + edges using NodeExtractor/EdgeExtractor and saves to CSV.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.time_logger import TimeLogger, setup_logger, Timer

logger = setup_logger("create_graph_v1")


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_chunks(data: dict) -> List[Dict]:
    """
    Recursively walk the parsed_structure_final.json tree and collect
    all chunks that have non-empty content.
    Returns a list of {"chunk_id": ..., "content": ..., "title": ...}
    """
    chunks = []

    def _collect(node, parent_title=""):
        # Collect chunk at current node
        chunk = node.get("chunk", {})
        chunk_name = chunk.get("name", "")
        content = chunk.get("content", "").strip()

        if content:
            chunks.append({
                "chunk_id": chunk_name,
                "content": content,
                "title": node.get("title", parent_title),
            })

        # Recurse into sections
        for section in node.get("sections", []):
            _collect(section, node.get("title", ""))

        # Recurse into subsections
        for subsection in node.get("subsections", []):
            _collect(subsection, node.get("title", ""))

    # Top-level document chunk
    _collect(data, data.get("title", ""))

    # Iterate through chapters
    for chapter in data.get("chapters", []):
        _collect(chapter, chapter.get("title", ""))

    return chunks


def main():
    config_path = "/home/nguyenthang/Tài liệu/import_qdrant/KGChat-03/backend/configs/configs.yml"
    input_path = "/home/nguyenthang/Tài liệu/import_qdrant/KGChat-03/parsed_structure_final.json"

    configs = load_config(config_path)

    # Load parsed structure
    logger.info(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten all chunks from hierarchical structure
    all_chunks = flatten_chunks(data)
    logger.info(f"Found {len(all_chunks)} chunks with content")

    if not all_chunks:
        logger.error("No chunks found!")
        return

    # Output setup
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"

    # Clear previous output
    for p in [nodes_path, edges_path]:
        if p.exists():
            p.unlink()

    # Initialize
    time_logger = TimeLogger(output_dir / "time_log.csv")
    extractor = GraphExtractor(config_path=config_path, time_logger=time_logger)

    total = len(all_chunks)
    total_nodes = 0
    total_edges = 0

    for idx, chunk in enumerate(all_chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        logger.info(f"[{idx+1}/{total}] Processing: {chunk_id} ({chunk['title'][:50]}...)")

        try:
            with Timer(time_logger, chunk_id, "Total Chunk Processing"):
                # Extract nodes (Level 1 entities only)
                nodes = extractor.node_extractor.extract(content, file_name=chunk_id)

                for node in nodes:
                    node["chunk_id"] = chunk_id
                    node["source_file"] = "parsed_structure_final.json"

                # Extract edges (skip REF_TO and Level 2 nodes)
                edges_result, _ = extractor.edge_extractor.extract(
                    text=content, nodes=nodes, file_name=chunk_id
                )

                all_edges = []
                if edges_result and edges_result.edges:
                    for edge in edges_result.edges:
                        edge_dict = edge.dict() if hasattr(edge, 'dict') else edge.model_dump()
                        if edge_dict.get("relation") == "REF_TO":
                            continue
                        edge_dict["chunk_id"] = chunk_id
                        edge_dict["source_file"] = "parsed_structure_final.json"
                        all_edges.append(edge_dict)

            # Save incrementally
            if nodes:
                extractor.save_nodes(nodes, nodes_path, append=True)
                total_nodes += len(nodes)

            if all_edges:
                extractor.save_edges(all_edges, edges_path, append=True)
                total_edges += len(all_edges)

            logger.info(f"  -> {len(nodes)} entities, {len(all_edges)} edges")

        except Exception as e:
            import traceback
            logger.error(f"Failed chunk {chunk_id}: {e}\n{traceback.format_exc()}")
            continue

    # Summary
    logger.info("=" * 60)
    logger.info(f"Done! Total: {total_nodes} nodes, {total_edges} edges")
    logger.info(f"Output: {nodes_path}, {edges_path}")


if __name__ == "__main__":
    main()