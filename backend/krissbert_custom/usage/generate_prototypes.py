# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large set of entity mentions
 based on the pretrained mention encoder.
"""
import logging
import os
import pathlib
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer, AutoModel

from utils_umls import generate_vectors


# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


@hydra.main(config_path="conf", config_name="generate_prototypes", version_base=None)
def main(cfg: DictConfig):
    logger.info("Configuration:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
    )
    encoder = AutoModel.from_pretrained(
        cfg.model_name_or_path,
        config=config
    )
    encoder.cuda()
    encoder.eval()

    ds = hydra.utils.instantiate(cfg.train_data)
    
    # Check output path
    output_dir = os.path.dirname(cfg.output_prototypes)
    if output_dir:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    iterator = generate_vectors(encoder, tokenizer, ds, cfg.batch_size, cfg.max_length, is_prototype=True)
    
    # Collect all records in RAM
    all_records = []
    
    logger.info("Loading all records into memory...")
    
    # Open name_cuis file once and write
    with open(cfg.output_name_cuis, 'w') as f_names:
        for batch in iterator:
            all_records.extend(batch)
            
            # Write names for this batch
            for metadata, vector in batch:
                cui = metadata['cui']
                alias = metadata['alias']
                f_names.write(f"{cui}||{alias}\n")
            
            # Log progress every 10000 records
            if len(all_records) % 10000 == 0:
                logger.info(f"Loaded {len(all_records)} records so far...")
    
    logger.info(f"Total records loaded: {len(all_records)}")
    
    # Save all records to a single pickle file
    logger.info(f"Saving all records to {cfg.output_prototypes}...")
    with open(cfg.output_prototypes, mode="wb") as f:
        pickle.dump(all_records, f)
    
    logger.info(f"Successfully saved {len(all_records)} records to {cfg.output_prototypes}")


if __name__ == "__main__":
    main()
