import csv
import time
import threading
from pathlib import Path
from collections import defaultdict
from typing import Optional
import logging

# Setup standard logging
def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

class TimeLogger:
    def __init__(self, output_path: Path, write_to_file: bool = True):
        self.output_path = output_path
        self.write_to_file = write_to_file
        # Define optimized columns
        self.fieldnames = [
            'file_name', 'start_time', 'total_duration', 'num_chunks', 
            'chunking', 
            'node_stage1_avg', 'node_stage2_avg', 'node_stage3_avg', 'node_stage4_avg', 'node_total_avg',
            'edge_krissbert_avg', 'edge_llm_avg', 'edge_total_avg'
        ]
        self._lock = threading.RLock()
        
        # Store active metrics: {file_name: {'start_time': ..., 'metrics': defaultdict(list)}}
        self.active_files = {}
        
        if self.write_to_file:
            self._init_file()

    def _init_file(self):
        with self._lock:
            if not self.output_path.exists():
                with self.output_path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

    def start_file(self, file_name: str):
        """Initialize tracking for a new file."""
        with self._lock:
            self.active_files[file_name] = {
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': defaultdict(list)
            }

    def log_metric(self, file_name: str, metric_name: str, duration: float):
        """Store a duration metric for later aggregation."""
        with self._lock:
            if file_name in self.active_files:
                self.active_files[file_name]['metrics'][metric_name].append(duration)

    def get_file_stats(self, file_name: str, total_duration: float) -> Optional[dict]:
        """Calculate and return stats for a file without writing."""
        with self._lock:
            if file_name not in self.active_files:
                return None

            data = self.active_files.pop(file_name)
            metrics = data['metrics']
            
            # Helper to calculate average
            def get_avg(key):
                vals = metrics.get(key, [])
                return sum(vals) / len(vals) if vals else 0.0

            # Determine number of chunks (based on node_total entries)
            num_chunks = len(metrics.get('node_total', []))
            if num_chunks == 0 and len(metrics.get('chunking', [])) > 0:
                num_chunks = 1 # Fallback

            chunking_time = sum(metrics.get('chunking', []))

            row = {
                'file_name': file_name,
                'start_time': data['start_time'],
                'total_duration': f"{total_duration:.4f}",
                'num_chunks': num_chunks,
                'chunking': f"{chunking_time:.4f}",
                
                'node_stage1_avg': f"{get_avg('node_stage1'):.4f}",
                'node_stage2_avg': f"{get_avg('node_stage2'):.4f}",
                'node_stage3_avg': f"{get_avg('node_stage3'):.4f}",
                'node_stage4_avg': f"{get_avg('node_stage4'):.4f}",
                'node_total_avg': f"{get_avg('node_total'):.4f}",
                
                'edge_krissbert_avg': f"{get_avg('edge_krissbert'):.4f}",
                'edge_llm_avg': f"{get_avg('edge_llm'):.4f}",
                'edge_total_avg': f"{get_avg('edge_total'):.4f}",
            }
            return row

    def write_row(self, row: dict):
        """Manually write a row to the CSV."""
        if not self.write_to_file:
            return
            
        with self._lock:
            # Ensure header exists if file was deleted or not created
            if not self.output_path.exists():
                with self.output_path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

            with self.output_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)

    def finalize_file(self, file_name: str, total_duration: float):
        """Aggregate and write file stats to CSV."""
        row = self.get_file_stats(file_name, total_duration)
        if row and self.write_to_file:
            self.write_row(row)

class Timer:
    def __init__(self, time_logger: Optional[TimeLogger], file_name: str, metric_name: str):
        self.time_logger = time_logger
        self.file_name = file_name
        self.metric_name = metric_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.time_logger:
            duration = time.time() - self.start_time
            self.time_logger.log_metric(self.file_name, self.metric_name, duration)
