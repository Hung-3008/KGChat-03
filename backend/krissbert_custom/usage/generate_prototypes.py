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

import sys
import uuid
import duckdb
from typing import Dict, Optional
from tqdm import tqdm

# Add project root to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.utils.qdrant_helper import QdrantHelper
from backend.krissbert_custom.usage.utils_umls import generate_vectors


# Setup logger
logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress httpx logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Simplified formatter without thread ID
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


def run_processing(cfg: DictConfig, dataset=None):
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

    if dataset is None:
        ds = hydra.utils.instantiate(cfg.train_data)
    else:
        ds = dataset
    
    # Check output path
    output_dir = os.path.dirname(cfg.output_prototypes)
    if output_dir:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Qdrant
    qdrant = QdrantHelper()
    collection_name = "kg_lv2_nodes"
    
    # Check if clear is requested
    if cfg.get("clear", False):
        logger.info(f"Clearing collection '{collection_name}' as requested...")
        qdrant.clear_collection(collection_name)
        
    qdrant.create_collection(collection_name, vector_size=768)

    # Initialize DuckDB
    duck_con = duckdb.connect('/media/hung/data1/codes/projects/FHC/data/umls.duckdb', read_only=True)

    iterator = generate_vectors(encoder, tokenizer, ds, cfg.batch_size, cfg.max_length, is_prototype=True)
    
    # Open name_cuis file once and write
    logger.info(f"Streaming embeddings to Qdrant collection '{collection_name}'...")
    
    total_processed = 0
    
    # Use tqdm for progress tracking
    # Note: generate_vectors is a generator, so we might not know total length easily unless we calculate it.
    # But we can wrap it in tqdm without total or estimate it.
    # Since we know batch_size and total items (roughly), we can try to pass total if available.
    # For now, just a simple tqdm wrapper.
    
    with open(cfg.output_name_cuis, 'w') as f_names:
        for batch in tqdm(iterator, desc="Processing batches"):
            points = []
            
            # Process batch
            for metadata, vector in batch:
                cui = metadata['cui']
                alias = metadata['alias']
                
                # Write to name_cuis file
                f_names.write(f"{cui}||{alias}\n")
                
                # Fetch info from DuckDB
                # 1. Definition (Prioritize MSH, NCI, then any)
                # MRDEF: CUI, SAB, DEF
                def_query = """
                    SELECT DEF FROM mrdef 
                    WHERE CUI = ? 
                    ORDER BY CASE WHEN SAB IN ('MSH', 'NCI') THEN 0 ELSE 1 END 
                    LIMIT 1
                """
                res_def = duck_con.execute(def_query, [cui]).fetchone()
                definition = res_def[0] if res_def else None
                
                # 2. ICD Code
                # MRCONSO: CUI, SAB, CODE
                icd_query = """
                    SELECT CODE FROM mrconso 
                    WHERE CUI = ? AND SAB LIKE 'ICD%' 
                    LIMIT 1
                """
                res_icd = duck_con.execute(icd_query, [cui]).fetchone()
                icd_code = res_icd[0] if res_icd else None
                
                # 3. Semantic Types
                # MRSTY: CUI, STY
                sty_query = "SELECT STY FROM mrsty WHERE CUI = ?"
                res_sty = duck_con.execute(sty_query, [cui]).fetchall()
                semantic_types = [r[0] for r in res_sty] if res_sty else []
                
                # Create Qdrant point
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{cui}_{alias}"))
                
                payload = {
                    "cui": cui,
                    "name": alias,
                    "definition": definition,
                    "icd": icd_code,
                    "semantic_types": semantic_types
                }
                
                points.append({
                    "id": point_id,
                    "vector": vector.tolist(),
                    "payload": payload
                })
            
            # Insert batch to Qdrant
            if points:
                qdrant.insert_points(collection_name, points)
                total_processed += len(points)
            
            # Log progress - removed custom logging in favor of tqdm
            # if total_processed % 1000 == 0:
            #     logger.info(f"Processed and inserted {total_processed} records...")

    logger.info(f"Successfully processed and inserted {total_processed} records into Qdrant.")
    duck_con.close()


@hydra.main(config_path="conf", config_name="generate_prototypes", version_base=None)
def main(cfg: DictConfig):
    run_processing(cfg)
    
    # No longer saving to pickle
    # logger.info(f"Saving all records to {cfg.output_prototypes}...")
    # with open(cfg.output_prototypes, mode="wb") as f:
    #     pickle.dump(all_records, f)
    
    # logger.info(f"Successfully saved {len(all_records)} records to {cfg.output_prototypes}")


if __name__ == "__main__":
    main()
