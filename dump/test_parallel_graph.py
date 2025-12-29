import sys
import os
import argparse
import time
import logging
from pathlib import Path

# Add project root to path
current_dir = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(current_dir) == 'scripts':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.time_logger import setup_logger

logger = setup_logger("test_parallel")

def test_parallel_execution(input_file):
    print(f"Testing parallel execution with file: {input_file}")
    
    # Ensure config exists or uses default
    config_path = os.path.join(project_root, "backend/configs/configs.yml")
    if not os.path.exists(config_path):
        print(f"Config path {config_path} not found!")
    
    extractor = GraphExtractor(config_path=config_path)
    
    # Run extracion
    start_time = time.time()
    try:
        nodes, edges = extractor.extract_from_file(input_file)
        end_time = time.time()
        
        print(f"Extraction completed in {end_time - start_time:.2f} seconds")
        print(f"Extracted {len(nodes)} nodes and {len(edges)} edges")
        
        if len(nodes) > 0:
            print("Success: Nodes extracted.")
            # Verify if chunk_ids are present and seemingly processed
            chunk_ids = set(n.get('chunk_id') for n in nodes)
            print(f"Processed chunks: {chunk_ids}")
        else:
            print("Warning: No nodes extracted.")
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Allow passing file via args
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input file")
    args = parser.parse_args()
    
    test_file = args.file
    
    if not test_file:
        test_file = os.path.join(project_root, "data/PMC_Part1/PMC10029850_result.json")
        
    if not os.path.exists(test_file):
        print(f"File {test_file} not found.")
        # fallback to finding one
        import glob
        files = glob.glob(os.path.join(project_root, "data/PMC_Part1/*.json"))
        if files:
            test_file = files[0]
            print(f"Using {test_file} instead.")
        else:
            print("No files found in data/PMC_Part1")
            sys.exit(1)
            
    test_parallel_execution(test_file)
