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

    # ds instantiation removed to use DuckDB streaming
    json_path = cfg.train_data.UMLS_path
    
    # Initialize DuckDB connections
    # One for streaming (read_json_auto) - Use in-memory DB to avoid lock contention
    duck_con_stream = duckdb.connect() 
    # One for lookups (mrdef, mrconso, etc)
    duck_con_lookup = duckdb.connect('/media/hung/data1/codes/projects/FHC/data/umls.duckdb', read_only=True)
    
    # Check output path
    output_dir = os.path.dirname(cfg.output_prototypes)
    if output_dir:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Qdrant
    qdrant = QdrantHelper()
    collection_name = "kg_lv2_nodes"
    
    # Determine start index and file mode
    start_index = 0
    file_mode = 'w'
    
    if cfg.get("resume", False):
        logger.info("Resume requested. Checking Qdrant collection count...")
        start_index = qdrant.get_collection_count(collection_name)
        if start_index > 0:
            file_mode = 'a'
            logger.info(f"Found {start_index} existing records in Qdrant. Resuming...")
        else:
            logger.info("Collection is empty or does not exist. Starting from scratch...")
            qdrant.create_collection(collection_name, vector_size=768)
    else:
        logger.info("Starting from scratch...")
        if cfg.get("clear", False):
             logger.info(f"Clearing collection '{collection_name}' as requested...")
             qdrant.clear_collection(collection_name)
        
        # Ensure collection exists
        qdrant.create_collection(collection_name, vector_size=768)

    # Construct DuckDB query
    duck_query = f"SELECT cui, stn, type, unnest(aliases) as alias FROM read_json_auto('{json_path}') OFFSET {start_index}"
    
    iterator = generate_vectors(
        encoder, tokenizer, dataset=None, batch_size=cfg.batch_size, 
        max_length=cfg.max_length, is_prototype=True, start_index=start_index,
        duck_con=duck_con_stream, duck_query=duck_query
    )
    
    # Open name_cuis file once and write
    logger.info(f"Streaming embeddings to Qdrant collection '{collection_name}'...")
    
    total_processed = 0
    
    # Thread pool for async Qdrant insertion
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)
    futures = []

    def insert_batch_to_qdrant(collection, points_batch):
        try:
            qdrant.insert_points(collection, points_batch)
            return len(points_batch)
        except Exception as e:
            logger.error(f"Error inserting batch to Qdrant: {e}")
            return 0

    with open(cfg.output_name_cuis, file_mode) as f_names:
        for batch in tqdm(iterator, desc="Processing batches"):
            points = []
            cuis = [m['cui'] for m, _ in batch]
            unique_cuis = list(set(cuis))
            
            if not unique_cuis:
                continue

            # Batch DuckDB Lookups
            # Prepare placeholders for IN clause
            placeholders = ','.join(['?'] * len(unique_cuis))
            
            # 1. Definition (Prioritize MSH, NCI)
            def_query = f"""
                SELECT CUI, DEF 
                FROM (
                    SELECT CUI, DEF, 
                           ROW_NUMBER() OVER (PARTITION BY CUI ORDER BY CASE WHEN SAB IN ('MSH', 'NCI') THEN 0 ELSE 1 END) as rn
                    FROM mrdef
                    WHERE CUI IN ({placeholders})
                ) WHERE rn = 1
            """
            res_def = duck_con_lookup.execute(def_query, unique_cuis).fetchall()
            def_map = {r[0]: r[1] for r in res_def}

            # 2. ICD Code
            icd_query = f"""
                SELECT CUI, CODE 
                FROM (
                    SELECT CUI, CODE,
                           ROW_NUMBER() OVER (PARTITION BY CUI ORDER BY CODE) as rn
                    FROM mrconso
                    WHERE CUI IN ({placeholders}) AND SAB LIKE 'ICD%'
                ) WHERE rn = 1
            """
            res_icd = duck_con_lookup.execute(icd_query, unique_cuis).fetchall()
            icd_map = {r[0]: r[1] for r in res_icd}

            # 3. Semantic Types
            sty_query = f"SELECT CUI, STY FROM mrsty WHERE CUI IN ({placeholders})"
            res_sty = duck_con_lookup.execute(sty_query, unique_cuis).fetchall()
            sty_map = {}
            for r in res_sty:
                if r[0] not in sty_map:
                    sty_map[r[0]] = []
                sty_map[r[0]].append(r[1])

            # Process batch
            for metadata, vector in batch:
                cui = metadata['cui']
                alias = metadata['alias']
                
                # Write to name_cuis file
                f_names.write(f"{cui}||{alias}\n")
                
                definition = def_map.get(cui)
                icd_code = icd_map.get(cui)
                semantic_types = sty_map.get(cui, [])
                
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
            
            # Insert batch to Qdrant asynchronously
            if points:
                future = executor.submit(insert_batch_to_qdrant, collection_name, points)
                futures.append(future)
                # Clean up finished futures to avoid memory leak
                futures = [f for f in futures if not f.done()]
                
                total_processed += len(points)
            
    # Wait for all insertions to complete
    for f in futures:
        f.result()

    logger.info(f"Successfully processed and inserted {total_processed} records into Qdrant.")
    executor.shutdown()
    duck_con_stream.close()
    duck_con_lookup.close()


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
