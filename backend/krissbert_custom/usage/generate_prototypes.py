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
import queue
import threading

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer, AutoModel

import sys
import uuid
import duckdb
from typing import Dict, Optional
from tqdm import tqdm
import time

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
    # Added ORDER BY to ensure deterministic ordering for OFFSET
    duck_query = f"""
        SELECT cui, stn, type, unnest(aliases) as alias 
        FROM read_json_auto('{json_path}') 
        ORDER BY cui, alias
        OFFSET {start_index}
    """
    logger.info("Executing DuckDB query (this may take a while due to sorting)...")
    
    iterator = generate_vectors(
        encoder, tokenizer, dataset=None, batch_size=cfg.batch_size, 
        max_length=cfg.max_length, is_prototype=True, start_index=start_index,
        duck_con=duck_con_stream, duck_query=duck_query
    )
    
    # Open name_cuis file once and write
    logger.info(f"Streaming embeddings to Qdrant collection '{collection_name}'...")
    
    # Queue for passing (metadata, vector) tuples to consumer
    # Limit size to prevent memory overflow if consumer is slow
    data_queue = queue.Queue(maxsize=50000) 
    
    # Event to signal consumer to stop
    stop_event = threading.Event()
    
    def consumer_worker(data_queue, stop_event, output_name_cuis, file_mode, collection_name):
        # Consumer needs its own DuckDB connection for lookups
        # This ensures thread safety and avoids blocking the stream connection
        local_duck_con = duckdb.connect('/media/hung/data1/codes/projects/FHC/data/umls.duckdb', read_only=True)
        local_qdrant = QdrantHelper() # New instance might be safer, or reuse if thread-safe
        
        item_buffer = []
        BUFFER_SIZE = 2048 # Increased buffer size for better batching
        
        total_inserted = 0
        
        with open(output_name_cuis, file_mode) as f_names:
            while not stop_event.is_set() or not data_queue.empty():
                try:
                    # Wait for item with timeout to check stop_event periodically
                    item = data_queue.get(timeout=1.0)
                    item_buffer.append(item)
                    data_queue.task_done()
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                
                if len(item_buffer) >= BUFFER_SIZE:
                    total_inserted += process_buffer(item_buffer, local_duck_con, local_qdrant, f_names, collection_name)
                    item_buffer = []
            
            # Process remaining items
            if item_buffer:
                total_inserted += process_buffer(item_buffer, local_duck_con, local_qdrant, f_names, collection_name)
        
        local_duck_con.close()
        logger.info(f"Consumer finished. Total inserted: {total_inserted}")

    def process_buffer(buffer, duck_con, qdrant_client, f_names, collection_name):
        if not buffer:
            return 0
            
        # Extract unique CUIs for lookup
        cuis = [m['cui'] for m, _ in buffer]
        unique_cuis = list(set(cuis))
        
        if not unique_cuis:
            return 0

        # Batch DuckDB Lookups
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
        res_def = duck_con.execute(def_query, unique_cuis).fetchall()
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
        res_icd = duck_con.execute(icd_query, unique_cuis).fetchall()
        icd_map = {r[0]: r[1] for r in res_icd}

        # 3. Semantic Types
        sty_query = f"SELECT CUI, STY FROM mrsty WHERE CUI IN ({placeholders})"
        res_sty = duck_con.execute(sty_query, unique_cuis).fetchall()
        sty_map = {}
        for r in res_sty:
            if r[0] not in sty_map:
                sty_map[r[0]] = []
            sty_map[r[0]].append(r[1])

        # Process buffer items
        points = []
        for metadata, vector in buffer:
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
        
        # Insert batch to Qdrant
        if points:
            try:
                qdrant_client.insert_points(collection_name, points)
                return len(points)
            except Exception as e:
                logger.error(f"Error inserting batch to Qdrant: {e}")
                return 0
        return 0

    # Start consumer thread
    consumer_thread = threading.Thread(
        target=consumer_worker, 
        args=(data_queue, stop_event, cfg.output_name_cuis, file_mode, collection_name)
    )
    consumer_thread.start()
    
    try:
        for batch in tqdm(iterator, desc="Processing batches"):
            # Accumulate items from current batch
            for metadata, vector in batch:
                data_queue.put((metadata, vector))
                
    except KeyboardInterrupt:
        logger.info("Interrupted! Stopping...")
    finally:
        logger.info("Producer finished. Waiting for consumer to empty queue...")
        stop_event.set()
        consumer_thread.join()
        
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
