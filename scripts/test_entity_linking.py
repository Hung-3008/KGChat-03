import os
import sys
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def run_test():
    project_root = "/home/hung/KGChat-03"
    
    # Add usage dir to sys.path to allow imports of utils and run_entity_linking
    usage_dir = os.path.join(project_root, "backend/krissbert_custom/usage")
    if usage_dir not in sys.path:
        sys.path.append(usage_dir)
        
    try:
        from run_entity_linking import EntityLinker
    except ImportError as e:
        print(f"Error importing EntityLinker: {e}")
        return

    # Paths
    sample_input_path = os.path.join(project_root, "scripts/sample_input.jsonl")
    model_path = os.path.join(project_root, "backend/krissbert_custom")
    encoded_files = [os.path.join(project_root, "backend/krissbert_custom/prototypes/embeddings.pkl")]
    entity_list_names = os.path.join(project_root, "backend/krissbert_custom/prototypes/name_cuis")
    
    # Load sample data
    data = []
    with open(sample_input_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples.")
    
    # Initialize Linker
    print("Initializing EntityLinker...")
    # Use /workspace for index cache to avoid disk space issues
    index_cache_path = "/workspace/krissbert_index_cache"
    
    linker = EntityLinker(
        model_name_or_path=model_path,
        encoded_files=encoded_files,
        entity_list_names=entity_list_names,
        index_path=index_cache_path,
        device="cuda" # or "cpu" if no gpu
    )
    
    # Predict
    # Load CUI to Name mapping
    print("Loading CUI to Name mapping...")
    cui_to_name = {}
    with open(entity_list_names, 'r', encoding='utf-8') as f:
        for line in f:
            if '||' in line:
                cuis_str, name = line.strip().split('||')
                for cui in cuis_str.split('|'):
                    # Store the first name encountered for each CUI, or overwrite? 
                    # Let's store the first one we see if not present, to keep it stable.
                    # Or maybe store all and pick the longest? 
                    # For now, just keeping the first one encountered is fine.
                    if cui not in cui_to_name:
                        cui_to_name[cui] = name

    # Predict
    print("Running prediction...")
    results = linker.predict(data, top_k=3)
    
    print("\n=== ENTITY LINKING RESULTS ===\n")
    for i, res in enumerate(results):
        print(f"Sample {i+1}:")
        print(f" MENTION: {res['mention']}")
        
        if res['candidates']:
            print(f" TOP {len(res['candidates'])} CANDIDATES:")
            for j, cand in enumerate(res['candidates']):
                cui = cand['cui']
                entity_name = cui_to_name.get(cui, "Unknown Name")
                print(f"  {j+1}. CUI: {cui} ({entity_name}) (Score: {cand['score']:.4f})")
        else:
            print(" PRED CUI: None")
        print("-" * 50)

if __name__ == "__main__":
    run_test()
