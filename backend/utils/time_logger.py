import csv
import time
from pathlib import Path
from typing import Optional, Dict
import logging
import threading

# Setup standard logging with HTTP suppression
def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO):
    """Setup logger with clean formatting and HTTP log suppression"""
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
    
    # Suppress HTTP logs from httpx and qdrant
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger


class TimeLogger:
    """
    Thread-safe time logger that stores one row per file with stage timings as columns.
    
    Output format:
    file_name, total_time, chunking, node_stage1, node_stage2, node_stage3, node_stage4, edge_krissbert, edge_llm, ...
    """
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._lock = threading.RLock()
        # Store timings per file in memory before writing
        self._file_timings: Dict[str, Dict[str, float]] = {}
        self._init_file()

    def _init_file(self):
        """Initialize CSV file with headers"""
        with self._lock:
            if not self.output_path.exists():
                # Will write header when first file is completed
                pass

    def log_time(self, file_name: str, step_name: str, duration: float):
        """Log timing for a specific step of a file"""
        with self._lock:
            if file_name not in self._file_timings:
                self._file_timings[file_name] = {}
            
            # Normalize step name for column naming
            column_name = step_name.replace(" ", "_").replace(":", "_").lower()
            
            # Accumulate time instead of overwriting (fixing parallel logging issue)
            current_time = self._file_timings[file_name].get(column_name, 0.0)
            self._file_timings[file_name][column_name] = current_time + duration

    def finalize_file(self, file_name: str):
        """
        Write accumulated timings for a file to CSV as a single row.
        Should be called when file processing is complete.
        """
        with self._lock:
            if file_name not in self._file_timings:
                return
            
            timings = self._file_timings[file_name]
            
            # Calculate total time
            total_time = sum(timings.values())
            
            # Prepare row data
            row_data = {
                'file_name': file_name,
                'total_time': f"{total_time:.2f}",
            }
            
            # Add all stage timings
            for stage, duration in sorted(timings.items()):
                row_data[stage] = f"{duration:.2f}"
            
            # Determine all columns (union of all stages seen so far)
            all_columns = {'file_name', 'total_time'}
            
            # Read existing file to get all column names
            if self.output_path.exists():
                with self.output_path.open('r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        all_columns.update(reader.fieldnames)
            
            # Add new columns from current file
            all_columns.update(timings.keys())
            
            # Sort columns: file_name, total_time, then alphabetically
            fieldnames = ['file_name', 'total_time'] + sorted([c for c in all_columns if c not in ['file_name', 'total_time']])
            
            # Write/append to file
            write_header = not self.output_path.exists()
            mode = 'w' if write_header else 'a'
            
            # If file exists but we have new columns, need to rewrite entire file
            if not write_header and self.output_path.exists():
                # Read all existing rows
                existing_rows = []
                with self.output_path.open('r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_fieldnames = reader.fieldnames or []
                    for row in reader:
                        existing_rows.append(row)
                
                # Check if we need to add new columns
                new_columns = set(fieldnames) - set(existing_fieldnames)
                if new_columns:
                    # Rewrite file with new columns
                    with self.output_path.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        # Write existing rows (missing columns will be empty)
                        for row in existing_rows:
                            writer.writerow(row)
                        # Write new row
                        writer.writerow(row_data)
                else:
                    # Just append
                    with self.output_path.open('a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(row_data)
            else:
                # New file, write header and row
                with self.output_path.open(mode, newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row_data)
            
            # Clear from memory
            del self._file_timings[file_name]


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, time_logger: TimeLogger, file_name: str, step_name: str):
        self.time_logger = time_logger
        self.file_name = file_name
        self.step_name = step_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        self.time_logger.log_time(self.file_name, self.step_name, duration)
