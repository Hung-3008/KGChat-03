import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def get_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Creates and configures a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    
    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)

    if log_file:
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        f_handler = logging.FileHandler(log_file)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

def save_results(results: dict, output_dir: str = "reports"):
    """
    Saves the experiment results to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"results_{timestamp}.json")
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Results saved to {file_path}")
