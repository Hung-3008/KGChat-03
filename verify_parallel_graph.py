
import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor
from backend.utils.time_logger import TimeLogger

def test_parallel_extraction():
    # Use an existing file or create a dummy one
    data_dir = Path("data/PMC_Part1")
    input_file = next(data_dir.glob("*.json"), None)
    
    if not input_file:
        print("No input file found in data/PMC_Part1")
        return

    print(f"Testing with file: {input_file}")
    
    # Initialize Extractor
    extractor = GraphExtractor(config_path="backend/configs/configs.yml")
    
    start_time = time.time()
    try:
        nodes, edges = extractor.extract_from_file(str(input_file))
        duration = time.time() - start_time
        print(f"Extraction completed in {duration:.2f}s")
        print(f"Extracted {len(nodes)} nodes and {len(edges)} edges")
        
        # Verify chunks were processed
        chunk_ids = set(n.get('chunk_id') for n in nodes)
        print(f"Processed chunks: {chunk_ids}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parallel_extraction()
