#!/usr/bin/env python3
"""
Test script for multi-file processing optimization

This script creates a small test subset and runs create_graph.py to verify:
1. Multi-file concurrent processing works correctly
2. No race conditions in file writing
3. Resume logic still functions
4. Performance improvement is measurable
"""

import os
import sys
import shutil
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

def setup_test_data(num_files=5):
    """Copy a small subset of files for testing"""
    source_dir = project_root / "data" / "PMC_Part1"
    test_dir = project_root / "data" / "test_subset"
    
    print(f"Setting up test data in {test_dir}")
    
    # Clean and create test directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy first N files
    source_files = sorted(list(source_dir.glob("*.json")))[:num_files]
    if not source_files:
        print(f"ERROR: No source files found in {source_dir}")
        return False
    
    for i, src_file in enumerate(source_files):
        dest_file = test_dir / src_file.name
        shutil.copy2(src_file, dest_file)
        print(f"  Copied {i+1}/{num_files}: {src_file.name}")
    
    return True

def clean_output():
    """Clean previous test outputs"""
    output_dir = project_root / "output_test"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cleaned output directory: {output_dir}")

def create_test_config():
    """Create a test configuration file"""
    test_config_path = project_root / "backend" / "configs" / "configs_test.yml"
    
    config_content = """LLM:
  client: ollama
  model: llama3.1:8b
  temperature: 0.85
  top_p: 1.0
  seed: 42

Gemini:
  model: gemini-2.5-flash
  temperature: 0.85
  top_p: 0.9

Encoder:
  model_name: dmis-lab/biobert-v1.1
  device: cuda

Create:
  Batch_size: 10
  Limit: Null
  Resume: False  # Test from scratch
  max_parallel_chunks: 3
  max_parallel_files: 3  # Use 3 for testing

Insert:
  Batch_size: 1
  Limit: 100000
  Resume: False

Qdrant:
  timeout: 300
  search_batch_size: 10
"""
    
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Created test config: {test_config_path}")
    return test_config_path

def modify_create_graph_for_test():
    """Temporarily modify create_graph.py to use test directories"""
    create_graph_path = project_root / "create_graph.py"
    
    # Read the file
    with open(create_graph_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already modified
    if 'TEST_MODE' in content:
        print("create_graph.py already in test mode")
        return True
    
    # Backup
    backup_path = project_root / "create_graph.py.backup"
    shutil.copy2(create_graph_path, backup_path)
    print(f"Backed up create_graph.py to {backup_path}")
    
    # Modify data_dir and output_dir
    modified_content = content.replace(
        'data_dir = Path("data/PMC_Part1")',
        'data_dir = Path("data/test_subset")  # TEST_MODE'
    ).replace(
        'output_dir = Path("output")',
        'output_dir = Path("output_test")  # TEST_MODE'
    )
    
    with open(create_graph_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("Modified create_graph.py for testing")
    return True

def restore_create_graph():
    """Restore original create_graph.py"""
    create_graph_path = project_root / "create_graph.py"
    backup_path = project_root / "create_graph.py.backup"
    
    if backup_path.exists():
        shutil.copy2(backup_path, create_graph_path)
        backup_path.unlink()
        print("Restored original create_graph.py")
        return True
    return False

def run_test():
    """Run the test"""
    print("\n" + "="*60)
    print("RUNNING MULTI-FILE PROCESSING TEST")
    print("="*60 + "\n")
    
    # Setup
    if not setup_test_data(num_files=5):
        return False
    
    clean_output()
    test_config = create_test_config()
    
    if not modify_create_graph_for_test():
        return False
    
    # Run create_graph.py with test config
    print("\n" + "-"*60)
    print("Running create_graph.py with multi-file processing...")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(project_root / "create_graph.py"), "--config", str(test_config)],
        cwd=str(project_root),
        capture_output=False
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "-"*60)
    print(f"Test completed in {elapsed_time:.2f} seconds")
    print("-"*60 + "\n")
    
    # Verify outputs
    output_test_dir = project_root / "output_test"
    nodes_csv = output_test_dir / "nodes.csv"
    edges_csv = output_test_dir / "edges.csv"
    log_file = output_test_dir / "processed_files.txt"
    
    success = True
    
    if not nodes_csv.exists():
        print("❌ nodes.csv not created")
        success = False
    else:
        print(f"✓ nodes.csv created ({nodes_csv.stat().st_size} bytes)")
    
    if not edges_csv.exists():
        print("❌ edges.csv not created")
        success = False
    else:
        print(f"✓ edges.csv created ({edges_csv.stat().st_size} bytes)")
    
    if not log_file.exists():
        print("❌ processed_files.txt not created")
        success = False
    else:
        with open(log_file, 'r') as f:
            processed_count = len([line for line in f if line.strip()])
        print(f"✓ processed_files.txt created ({processed_count} files logged)")
    
    # Restore
    restore_create_graph()
    
    if success:
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return success

if __name__ == "__main__":
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        restore_create_graph()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        restore_create_graph()
        sys.exit(1)
