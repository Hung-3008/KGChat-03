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
    
    # Prepare output name pattern
    # If cfg.output_prototypes is "prototypes/embeddings.pkl", we want "prototypes/embeddings_part_0.pkl"
    base_name, ext = os.path.splitext(cfg.output_prototypes)
    
    iterator = generate_vectors(encoder, tokenizer, ds, cfg.batch_size, cfg.max_length, is_prototype=True)
    
    buffer = []
    chunk_id = 0
    CHUNK_SIZE = 100000  # Adjust as needed
    
    total_saved = 0
    
    # Open name_cuis file once and append
    with open(cfg.output_name_cuis, 'w') as f_names:
        for batch in iterator:
            buffer.extend(batch)
            
            # Write names immediately to avoid storing them in memory if possible, 
            # but we need them in the buffer for pickle? 
            # The buffer contains (metadata, vector).
            # We can write names from the batch.
            for metadata, vector in batch:
                cui = metadata['cui']
                alias = metadata['alias']
                f_names.write(f"{cui}||{alias}\n")
            
            if len(buffer) >= CHUNK_SIZE:
                chunk_path = f"{base_name}_part_{chunk_id}{ext}"
                logger.info(f"Saving chunk {chunk_id} to {chunk_path} ({len(buffer)} items)")
                with open(chunk_path, mode="wb") as f:
                    pickle.dump(buffer, f)
                total_saved += len(buffer)
                buffer = []
                chunk_id += 1
        
        # Save remaining
        if buffer:
            chunk_path = f"{base_name}_part_{chunk_id}{ext}"
            logger.info(f"Saving chunk {chunk_id} to {chunk_path} ({len(buffer)} items)")
            with open(chunk_path, mode="wb") as f:
                pickle.dump(buffer, f)
            total_saved += len(buffer)
            
    logger.info("Total data processed %d. Written chunks to %s*", total_saved, base_name)


if __name__ == "__main__":
    main()
