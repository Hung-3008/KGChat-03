import csv
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Optional
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.time_logger import TimeLogger, setup_logger, Timer

logger = setup_logger("create_graph", log_file=Path("output/create_graph.log"))

_EXTRACTOR: Optional[GraphExtractor] = None
_EXTRACTOR_CONFIG: Optional[str] = None


def _get_extractor(config_path: str, time_logger: TimeLogger) -> GraphExtractor:
    """Reuse a single GraphExtractor per process and swap time_logger cheaply."""
    global _EXTRACTOR, _EXTRACTOR_CONFIG
    if _EXTRACTOR is None or _EXTRACTOR_CONFIG != config_path:
        _EXTRACTOR = GraphExtractor(config_path=config_path, time_logger=time_logger)
        _EXTRACTOR_CONFIG = config_path
    else:
        _EXTRACTOR.set_time_logger(time_logger)
    return _EXTRACTOR


def append_csv(temp_path: Path, dest_handle, has_header: bool) -> bool:
    """Append a CSV file to an open destination handle, writing header only once."""
    if not temp_path or not temp_path.exists():
        return has_header

    with temp_path.open("r", encoding="utf-8") as src:
        header = src.readline()
        if not header:
            temp_path.unlink(missing_ok=True)
            return has_header

        if not has_header:
            dest_handle.write(header)
            has_header = True
        # Skip header if dest already has one
        for line in src:
            dest_handle.write(line)

    dest_handle.flush()
    temp_path.unlink(missing_ok=True)
    return has_header

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
        # Each process needs its own logger instance, but shouldn't write to file directly
        # to avoid race conditions. We'll return the stats to the main process instead.
        time_logger = TimeLogger(output_dir / "time_log.csv", write_to_file=False)
        time_logger.start_file(file_path.name)

        extractor = _get_extractor(config_path=config_path, time_logger=time_logger)

        logger.info(f"Processing: {file_path.name}")

        start_time = time.time()
        nodes, edges = extractor.extract_from_file(str(file_path))
        total_duration = time.time() - start_time

        temp_dir = output_dir / "tmp_results"
        temp_dir.mkdir(parents=True, exist_ok=True)

        nodes_path = None
        edges_path = None

        if nodes:
            nodes_path = temp_dir / f"{file_path.stem}__nodes.csv"
            extractor.save_nodes(nodes, nodes_path, append=False)

        if edges:
            edges_path = temp_dir / f"{file_path.stem}__edges.csv"
            extractor.save_edges(edges, edges_path, append=False)

        # Get stats but don't write to file here
        timing_stats = time_logger.get_file_stats(file_path.name, total_duration)

        logger.info(f"✓ Completed {file_path.name}: {len(nodes)} nodes, {len(edges)} edges")
        return {
            'filename': file_path.name,
            'nodes_path': nodes_path,
            'edges_path': edges_path,
            'timing_stats': timing_stats,
            'nodes_count': len(nodes),
            'edges_count': len(edges),
            'success': True,
            'error': None
        }
    except Exception as e:
        logger.error(f"✗ Failed {file_path.name}: {e}")
        import traceback
        return {
            'filename': file_path.name,
            'nodes_path': None,
            'edges_path': None,
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

    temp_dir = output_dir / "tmp_results"
    temp_dir.mkdir(exist_ok=True)

    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"
    time_log_path = output_dir / "time_log.csv"
    log_path = output_dir / "processed_files.txt"

    # Initialize TimeLogger in main process to get fieldnames without touching disk
    main_time_logger = TimeLogger(time_log_path, write_to_file=False)

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
        if time_log_path.exists():
            time_log_path.unlink()
        if temp_dir.exists():
            for f in temp_dir.glob("*"):
                f.unlink()
            
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

    # Prepare shared file handles
    nodes_handle = nodes_path.open("a", encoding="utf-8")
    edges_handle = edges_path.open("a", encoding="utf-8")
    processed_handle = log_path.open("a", encoding="utf-8")
    time_log_handle = time_log_path.open("a", newline="", encoding="utf-8")
    time_log_writer = csv.DictWriter(time_log_handle, fieldnames=main_time_logger.fieldnames)

    nodes_has_header = nodes_path.exists() and nodes_path.stat().st_size > 0
    edges_has_header = edges_path.exists() and edges_path.stat().st_size > 0
    if not (time_log_path.exists() and time_log_path.stat().st_size > 0):
        time_log_writer.writeheader()
        time_log_handle.flush()

    # Process files
    completed_count = 0

    try:
        if max_parallel_files <= 1:
            logger.info("Single-worker mode: running sequentially (no process pool) for easier interrupt.")
            try:
                for file_path in files_to_process:
                    result = process_single_file((file_path, config_path, output_dir))

                    if result['success']:
                        with file_lock:
                            nodes_has_header = append_csv(result.get('nodes_path'), nodes_handle, nodes_has_header)
                            edges_has_header = append_csv(result.get('edges_path'), edges_handle, edges_has_header)

                            if result.get('timing_stats'):
                                time_log_writer.writerow(result['timing_stats'])
                                time_log_handle.flush()

                            processed_handle.write(f"{result['filename']}\n")
                            processed_handle.flush()

                        completed_count += 1
                        logger.info(f"Progress: {completed_count}/{total_to_process} files completed")
                    else:
                        logger.error(f"Failed to process {result['filename']}: {result['error']}")
            except KeyboardInterrupt:
                logger.warning("Interrupted by user; stopping sequential run and killing process group.")
                import signal
                try:
                    os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
                except Exception as e:
                    logger.error(f"Failed to kill process group: {e}")
                return
        else:
            # Use ProcessPoolExecutor for true parallelism
            # Note: Using processes instead of threads to avoid GIL and share GPU properly
            with ProcessPoolExecutor(max_workers=max_parallel_files) as executor:
                future_to_file = {
                    executor.submit(process_single_file, (file_path, config_path, output_dir)): file_path 
                    for file_path in files_to_process
                }
                try:
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        completed_count += 1
                        
                        result = future.result()
                        
                        if result['success']:
                            with file_lock:
                                nodes_has_header = append_csv(result.get('nodes_path'), nodes_handle, nodes_has_header)
                                edges_has_header = append_csv(result.get('edges_path'), edges_handle, edges_has_header)

                                if result.get('timing_stats'):
                                    time_log_writer.writerow(result['timing_stats'])
                                    time_log_handle.flush()

                                processed_handle.write(f"{result['filename']}\n")
                                processed_handle.flush()

                            logger.info(f"Progress: {completed_count}/{total_to_process} files completed")
                        else:
                            logger.error(f"Failed to process {result['filename']}: {result['error']}")
                except KeyboardInterrupt:
                    logger.warning("Interrupted by user; cancelling pending tasks and killing process group.")
                    import signal
                    try:
                        # Kill the entire process group to ensure all children are terminated
                        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
                    except Exception as e:
                        logger.error(f"Failed to kill process group: {e}")
                        # Fallback to executor shutdown
                        for future in future_to_file:
                            future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                    return
                except Exception as e:
                    logger.error(f"Exception while processing {file_path.name}: {e}")
    finally:
        nodes_handle.close()
        edges_handle.close()
        processed_handle.close()
        time_log_handle.close()

    logger.info(f"✓ Completed {completed_count}/{total_to_process} files")

if __name__ == "__main__":
    main()
