import os
import sys
import yaml
from pathlib import Path
from typing import List
import logging
import concurrent.futures
import queue
import time
import threading

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.time_logger import TimeLogger, setup_logger, Timer
from backend.encoders.transformer_encoder import TransformerEncoder

logger = setup_logger("create_graph")

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def process_single_file(file_path: Path, extractor_queue: queue.Queue, time_logger: TimeLogger, output_dir: Path):
    """
    Process a single file using an available extractor from the queue.
    """
    extractor = None
    try:
        # Get an available extractor (blocks until one is free)
        extractor = extractor_queue.get()
        port = extractor.llm_config.get('base_url', '11434').split(':')[-1]
        logger.info(f"Processing file: {file_path.name} on port {port}")
        
        # Create output directory for this file
        file_stem = file_path.stem
        file_output_dir = output_dir / "graph" / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        nodes_path = file_output_dir / "nodes.csv"
        edges_path = file_output_dir / "edges.csv"
        
        with Timer(time_logger, file_path.name, "Total File Processing"):
            nodes, edges = extractor.extract_from_file(str(file_path))
        
        # Save results
        if nodes:
            extractor.save_nodes(nodes, nodes_path, append=False)
        if edges:
            extractor.save_edges(edges, edges_path, append=False)
            
        time_logger.finalize_file(file_path.name)
        
        logger.info(f"✓ Completed {file_path.name} on port {port}: {len(nodes)} nodes, {len(edges)} edges")
        return file_path.name, True
        
    except Exception as e:
        logger.error(f"✗ Failed {file_path.name}: {e}")
        time_logger.finalize_file(file_path.name)
        return file_path.name, False
    finally:
        # Return extractor to queue
        if extractor:
            extractor_queue.put(extractor)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backend/configs/configs.yml", help="Path to config file")
    args = parser.parse_args()
    
    config_path = args.config
    configs = load_config(config_path)
    
    create_config = configs.get("Create", {})
    limit = create_config.get("Limit")
    resume = create_config.get("Resume", False)
    # Use max_parallel_files from config, which determines how many threads run
    # Should ideally be >= number of ports to utilize all ports
    max_parallel_files = create_config.get("max_parallel_files", 5) 
    
    data_dir = Path("data/PMC_Part1")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Get all JSON files
    all_files = sorted(list(data_dir.glob("*.json")))
    
    logger.info(f"Found {len(all_files)} files in {data_dir}")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize TimeLogger
    time_logger = TimeLogger(output_dir / "time_log.csv")
    
    # --- Initialize Resources ---
    
    # 1. Initialize Shared Encoder
    logger.info("Initializing Shared Encoder...")
    encoder_config = configs.get("Encoder", {})
    embedding_model = encoder_config.get("model_name", "intfloat/multilingual-e5-base")
    device = encoder_config.get("device", "cpu")
    shared_encoder = TransformerEncoder(model_name=embedding_model, device=device)
    
    # 2. Initialize GraphExtractors Pool
    # Ports mapping to docker instances
    ollama_config = configs.get("Ollama", {})
    OLLAMA_PORTS = ollama_config.get("ports", [11434, 11435, 11436, 11437, 11438])
    extractor_queue = queue.Queue()
    
    logger.info(f"Initializing {len(OLLAMA_PORTS)} GraphExtractors for ports {OLLAMA_PORTS}...")
    for port in OLLAMA_PORTS:
        base_url = f"http://localhost:{port}"
        # Create extractor sharing the encoder
        ex = GraphExtractor(
            config_path=config_path, 
            time_logger=time_logger,
            encoder=shared_encoder, 
            llm_base_url=base_url
        )
        extractor_queue.put(ex)
        
    log_path = output_dir / "processed_files.txt"
    processed_files = set()
    
    if resume and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f if line.strip())
        logger.info(f"Resuming from {len(processed_files)} processed files")
    elif not resume and log_path.exists():
        log_path.unlink()
            
    files_to_process = [f for f in all_files if f.name not in processed_files]
    
    if limit is not None and isinstance(limit, int) and limit > 0:
        files_to_process = files_to_process[:limit]
    
    total_to_process = len(files_to_process)
    logger.info(f"Files to process: {total_to_process}")
    
    if not files_to_process:
        return

    # Process in Parallel
    logger.info(f"Starting parallel processing with {max_parallel_files} threads and {len(OLLAMA_PORTS)} LLM backends")
    
    completed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_files) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, f, extractor_queue, time_logger, output_dir): f 
            for f in files_to_process
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                fname, success = future.result()
                if success:
                    completed_count += 1
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{fname}\n")
            except Exception as e:
                logger.error(f"Critical error in future for {file_path}: {e}")
                
    logger.info(f"Completed {completed_count}/{total_to_process} files")

if __name__ == "__main__":
    main()
