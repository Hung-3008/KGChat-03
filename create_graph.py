import os
import sys
import yaml
from pathlib import Path
from typing import List
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.time_logger import TimeLogger, setup_logger, Timer

logger = setup_logger("create_graph")

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
    
    config_path = args.config
    configs = load_config(config_path)
    
    batch_size = configs.get("Batch_size", 10)
    limit = configs.get("Limit")
    resume = configs.get("Resume", False)
    
    data_dir = Path("data/500_samples_pmc")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Get all JSON files
    all_files = sorted(list(data_dir.glob("*.json")))
    total_files = len(all_files)
    logger.info(f"Found {total_files} files in {data_dir}")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize TimeLogger
    time_logger = TimeLogger(output_dir / "time_log.csv")
    
    # Pass time_logger to GraphExtractor
    extractor = GraphExtractor(config_path=config_path, time_logger=time_logger)
    
    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"
    log_path = output_dir / "processed_files.txt"
    
    processed_files = set()
    
    if resume:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as f:
                processed_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Resuming... Found {len(processed_files)} processed files in log.")
        else:
            logger.info("Resume requested but no log file found. Starting from scratch.")
    else:
        logger.info("Starting fresh (Resume=False). Clearing existing output.")
        if nodes_path.exists():
            nodes_path.unlink()
        if edges_path.exists():
            edges_path.unlink()
        if log_path.exists():
            log_path.unlink()
            
    # Filter files if resuming
    files_to_process = [f for f in all_files if f.name not in processed_files]
    
    if limit is not None:
        if isinstance(limit, int):
            remaining_limit = limit - len(processed_files)
            if remaining_limit <= 0:
                logger.info(f"Limit ({limit}) reached or exceeded by already processed files ({len(processed_files)}). Nothing to do.")
                files_to_process = []
            else:
                files_to_process = files_to_process[:remaining_limit]
                logger.info(f"Limit applied: {limit}. Already processed: {len(processed_files)}. Processing next {len(files_to_process)} files.")
        else:
             logger.info(f"Limit is not an integer ({limit}), processing all remaining files.")
    
    logger.info(f"Total files to process in this run: {len(files_to_process)}")
    
    if not files_to_process:
        return

    # Process in batches
    for i in range(0, len(files_to_process), batch_size):
        batch_files = files_to_process[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} (Files {i+1} to {min(i + batch_size, len(files_to_process))})...")
        
        batch_nodes = []
        batch_edges = []
        successful_files = []
        
        for file_path in batch_files:
            try:
                with Timer(time_logger, file_path.name, "Total File Processing"):
                    nodes, edges = extractor.extract_from_file(str(file_path))
                    batch_nodes.extend(nodes)
                    batch_edges.extend(edges)
                successful_files.append(file_path.name)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Save batch results
        if batch_nodes or batch_edges:
            logger.info(f"Saving batch results: {len(batch_nodes)} nodes, {len(batch_edges)} edges...")
            extractor.save_nodes(batch_nodes, nodes_path, append=True)
            extractor.save_edges(batch_edges, edges_path, append=True)
        
        # Update log
        if successful_files:
            with log_path.open("a", encoding="utf-8") as f:
                for fname in successful_files:
                    f.write(f"{fname}\n")
        
        # Clear RAM (variables)
        del batch_nodes
        del batch_edges
        
    logger.info("Graph creation complete.")

if __name__ == "__main__":
    main()
