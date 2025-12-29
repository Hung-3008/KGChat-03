import sys
import os
import json
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from backend.krissbert_custom.usage.run_entity_linking import EntityLinker

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    # Config
    model_path = "/media/hung/data1/codes/projects/FHC/backend/krissbert_custom"
    # Check if model exists, if not fallback to bert-base-uncased for testing logic
    if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        logger.warning(f"Model not found at {model_path}, falling back to 'bert-base-uncased'")
        model_path = "bert-base-uncased"
        
    input_file = "/media/hung/data1/codes/projects/FHC/scripts/sample_input.jsonl"
    
    # Initialize EntityLinker
    logger.info("Initializing EntityLinker...")
    linker = EntityLinker(
        model_name_or_path=model_path,
        device="cuda" # or "cpu"
    )
    
    # Load data
    logger.info(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    # Predict
    logger.info("Running prediction...")
    results = linker.predict(data, top_k=5)
    
    # Print results
    print("\n" + "="*50)
    print("ENTITY LINKING RESULTS")
    print("="*50)
    for res in results:
        print(f"\nMention: {res['mention']}")
        print("-" * 20)
        for i, cand in enumerate(res['candidates']):
            print(f"{i+1}. {cand['name']} (CUI: {cand['cui']})")
            print(f"   Score: {cand['score']:.4f}")
            if cand['definition']:
                print(f"   Def: {cand['definition'][:100]}...")
            if cand['icd']:
                print(f"   ICD: {cand['icd']}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
