import sys
import os
import logging
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from backend.krissbert_custom.usage.generate_prototypes import run_processing
from backend.krissbert_custom.usage.utils_umls import PreprocessedUMLS

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define config manually for testing
    base_cfg = OmegaConf.create({
        "model_name_or_path": "bert-base-uncased", # Use standard BERT for testing pipeline since local weights are missing
        "train_data": {
            "_target_": "backend.krissbert_custom.usage.utils_umls.PreprocessedUMLS",
            "UMLS_path": "/media/hung/data1/codes/projects/FHC/backend/krissbert_custom/umls_full.json"
        },
        "batch_size": 10, # Small batch for testing
        "max_length": 64,
        "output_prototypes": "backend/krissbert_custom/prototypes/embeddings_test.pkl", # Won't be used but needed for path check
        "output_name_cuis": "backend/krissbert_custom/prototypes/name_cuis_test"
    })
    
    # Merge with CLI arguments
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    logger.info("Loading dataset...")
    # Instantiate dataset manually
    dataset = PreprocessedUMLS(cfg.train_data.UMLS_path)
    
    # Slice dataset to 1000 samples
    original_len = len(dataset.data)
    dataset.data = dataset.data[:1000]
    logger.info(f"Sliced dataset from {original_len} to {len(dataset.data)} samples.")

    logger.info("Running processing...")
    run_processing(cfg, dataset=dataset)
    
    logger.info("Test completed.")

if __name__ == "__main__":
    main()
