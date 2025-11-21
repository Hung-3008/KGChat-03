import csv
import time
from pathlib import Path
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
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.fieldnames = ['file_name', 'step_name', 'duration_seconds', 'timestamp']
        self._init_file()

    def _init_file(self):
        if not self.output_path.exists():
            with self.output_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_time(self, file_name: str, step_name: str, duration: float):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with self.output_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'file_name': file_name,
                'step_name': step_name,
                'duration_seconds': f"{duration:.4f}",
                'timestamp': timestamp
            })

class Timer:
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
