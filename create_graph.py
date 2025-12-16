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

def process_single_file(args):
    """Process a single file and return nodes, edges, and filename.
    This must be a module-level function for ProcessPoolExecutor pickling.
    """
    file_path, config_path, output_dir = args
    
    try:
        # Each process needs its own extractor
        time_logger = TimeLogger(output_dir / "time_log.csv")
        extractor = GraphExtractor(config_path=config_path, time_logger=time_logger)
        
        logger.info(f"Processing: {file_path.name}")
        
        with Timer(time_logger, file_path.name, "Total File Processing"):
            nodes, edges = extractor.extract_from_file(str(file_path))
        
        logger.info(f"✓ Completed {file_path.name}: {len(nodes)} nodes, {len(edges)} edges")
        return {
            'filename': file_path.name,
            'nodes': nodes,
            'edges': edges,
            'success': True,
            'error': None
        }
    except Exception as e:
        logger.error(f"✗ Failed {file_path.name}: {e}")
        import traceback
        return {
            'filename': file_path.name,
            'nodes': [],
            'edges': [],
            'success': False,
            'error': str(e) + "\n" + traceback.format_exc()
        }

def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import threading
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backend/configs/configs.yml", help="Path to config file")
    args = parser.parse_args()
    
    config_path = args.config
    configs = load_config(config_path)
    
    create_config = configs.get("Create", {})
    batch_size = create_config.get("Batch_size", 10)
    limit = create_config.get("Limit")
    resume = create_config.get("Resume", False)
    max_parallel_files = create_config.get("max_parallel_files", 1)
    
    data_dir = Path("data/PMC_Part1")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Get all JSON files
    all_files = sorted(list(data_dir.glob("*.json")))
    total_files = len(all_files)
    
    logger.info(f"Found {total_files} files in {data_dir}")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"
    log_path = output_dir / "processed_files.txt"
    
    # Thread lock for file writing
    file_lock = threading.Lock()
    
    processed_files = set()
    
    if resume:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as f:
                processed_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Resuming from {len(processed_files)} processed files")
        else:
            logger.info("No previous log found, starting fresh")
    else:
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
                logger.info(f"Limit reached ({limit} files). Nothing to do.")
                files_to_process = []
            else:
                files_to_process = files_to_process[:remaining_limit]
    
    total_to_process = len(files_to_process)
    if not files_to_process:
        return

    logger.info(f"Processing {total_to_process} files with {max_parallel_files} workers")
    
    # Process files in batches
    completed_count = 0
    
    # Use ProcessPoolExecutor for true parallelism
    # Note: Using processes instead of threads to avoid GIL and share GPU properly
    with ProcessPoolExecutor(max_workers=max_parallel_files) as executor:
        # Submit all files with arguments
        future_to_file = {
            executor.submit(process_single_file, (file_path, config_path, output_dir)): file_path 
            for file_path in files_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed_count += 1
            
            try:
                result = future.result()
                
                if result['success']:
                    # Thread-safe file writing
                    with file_lock:
                        if result['nodes'] or result['edges']:
                            # Create a temporary extractor just for saving
                            temp_extractor = GraphExtractor(config_path=config_path)
                            temp_extractor.save_nodes(result['nodes'], nodes_path, append=True)
                            temp_extractor.save_edges(result['edges'], edges_path, append=True)
                        
                        # Update log
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(f"{result['filename']}\n")
                    
                    logger.info(f"Progress: {completed_count}/{total_to_process} files completed")
                else:
                    logger.error(f"Failed to process {result['filename']}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Exception while processing {file_path.name}: {e}")
    
    logger.info(f"✓ Completed all {total_to_process} files")

if __name__ == "__main__":
    main()
