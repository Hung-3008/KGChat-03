import os
import sys
import yaml
from pathlib import Path
from typing import List
import logging
import concurrent.futures

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

def process_single_file(file_path: Path, extractor: GraphExtractor, time_logger: TimeLogger, output_dir: Path):
    """
    Process a single file and save results to its own directory
    """
    try:
        # Create output directory for this file
        file_stem = file_path.stem
        file_output_dir = output_dir / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        nodes_path = file_output_dir / "nodes.csv"
        edges_path = file_output_dir / "edges.csv"
        
        # Skip if already done (check nodes/edges existence) - Optional, but keeping logic relying on central log for now
        # Actually logic is handled by 'processed_files' set in main, but let's double check if we want to overwrite
        
        logger.info(f"Processing file: {file_path.name}")
        
        with Timer(time_logger, file_path.name, "Total File Processing"):
            nodes, edges = extractor.extract_from_file(str(file_path))
        
        # Save results immediately to file-specific CSVs
        if nodes:
            extractor.save_nodes(nodes, nodes_path, append=False)
        if edges:
            extractor.save_edges(edges, edges_path, append=False)
            
        # Finalize time log for this file
        time_logger.finalize_file(file_path.name)
        
        logger.info(f"✓ Completed {file_path.name}: {len(nodes)} nodes, {len(edges)} edges")
        return file_path.name, True
        
    except Exception as e:
        logger.error(f"✗ Failed {file_path.name}: {e}")
        # Still finalize time log for partial timings
        time_logger.finalize_file(file_path.name)
        return file_path.name, False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backend/configs/configs.yml", help="Path to config file")
    args = parser.parse_args()
    
    config_path = args.config
    configs = load_config(config_path)
    
    create_config = configs.get("Create", {})
    # batch_size = create_config.get("Batch_size", 10) # Not relevant for parallel file processing strategy
    limit = create_config.get("Limit")
    resume = create_config.get("Resume", False)
    max_parallel_files = create_config.get("max_parallel_files", 3)
    
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
    
    # Pass time_logger to GraphExtractor
    # Note: creating one extractor instance. Assuming internal components (LLMClient, TransformerEncoder) are thread-safe or stateless
    # TransformerEncoder usually loads model on GPU. Sharing it across threads is fine for inference if handled correctly.
    # LLMClient (Ollama/VLLM) is http based so thread safe.
    extractor = GraphExtractor(config_path=config_path, time_logger=time_logger)
    
    log_path = output_dir / "processed_files.txt"
    
    processed_files = set()
    
    if resume:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as f:
                processed_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Resuming from {len(processed_files)} processed files")
        else:
            logger.info("No previous log found, starting fresh")
    else:
        # If not resuming, we should ideally clear output dir but that's dangerous.
        # Just clear log file
        if log_path.exists():
            log_path.unlink()
            
    # Filter files
    files_to_process = [f for f in all_files if f.name not in processed_files]
    
    if limit is not None:
        if isinstance(limit, int):
            remaining_limit = limit - len(processed_files) # Correct logic?? Or limit applied to total run?
            # Usually limit means "process X files in this run" or "stop after X total".
            # Let's assume limit is "max files to process in this run"
            if limit > 0:
                files_to_process = files_to_process[:limit]
            else:
                files_to_process = []
    
    total_to_process = len(files_to_process)
    logger.info(f"Files to process: {total_to_process}")
    
    if not files_to_process:
        return

    # Process in Parallel
    logger.info(f"Starting parallel processing with {max_parallel_files} workers")
    
    completed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_files) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, f, extractor, time_logger, output_dir): f 
            for f in files_to_process
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                fname, success = future.result()
                if success:
                    completed_count += 1
                    # Append to processed log safely
                    # Although 'a' is atomic, let's keep it simple. Main thread writing is safe.
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{fname}\n")
            except Exception as e:
                logger.error(f"Critical error in future for {file_path}: {e}")
                
    logger.info(f"✓ Completed {completed_count}/{total_to_process} files")

if __name__ == "__main__":
    main()
