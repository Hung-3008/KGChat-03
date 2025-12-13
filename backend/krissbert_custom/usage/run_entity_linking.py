# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Run entity linking
"""

import os
import glob
import logging
import pathlib
import pickle
import time
import math
import multiprocessing
from typing import List, Tuple, Dict, Set
from functools import partial
from multiprocessing.dummy import Pool

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

import sys
# Add project root to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from backend.utils.qdrant_helper import QdrantHelper
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import httpx

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
)
from backend.krissbert_custom.usage.utils import generate_vectors, ContextualMention


# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


class DenseRetriever:
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def generate_mention_vectors(self, ds: torch.utils.data.Dataset) -> T:
        self.encoder.eval()
        return generate_vectors(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dataset=ds,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )


class EntityLinker:
    def __init__(
        self,
        model_name_or_path: str,
        entity_list_names: str = None, # Optional now
        batch_size: int = 256,
        max_length: int = 64,
        device: str = "cuda",
        search_batch_size: int = 5 # Default to 10 to avoid Qdrant timeouts
    ):
        self.model_name_or_path = model_name_or_path
        self.entity_list_names = entity_list_names
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.search_batch_size = search_batch_size

        logger.info("Loading model from %s", model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.encoder.cuda()
        self.encoder.eval()
        
        # Init Qdrant
        self.qdrant = QdrantHelper()
        self.collection_name = "kg_lv2_nodes"
        
        # Load candidate names mapping
        self.lut = {}
        if self.entity_list_names and os.path.exists(self.entity_list_names):
            logger.info("Loading entity list names from %s", self.entity_list_names)
            with open(self.entity_list_names, encoding='utf-8') as f:
                for ln in f:
                    if '||' not in ln: continue
                    cuis, name = ln.strip().split('||')
                    cuis = cuis.split('|')
                    self.lut[name] = cuis

        # Init retriever helper
        self.retriever = DenseRetriever(
            self.encoder, self.tokenizer, self.batch_size, self.max_length
        )

    def _query_qdrant(self, vec: List[float], top_k: int) -> list:
        """Helper to run a single Qdrant search with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                groups = self.qdrant.client.query_points_groups(
                    collection_name=self.collection_name,
                    query=vec,
                    group_by="cui",
                    limit=top_k, 
                    group_size=1, 
                    with_payload=["cui", "name", "definition", "icd", "semantic_types"],
                    search_params=models.SearchParams(
                        hnsw_ef=128,
                        exact=False
                    )
                )
                return groups.groups
            except (ResponseHandlingException, httpx.ReadTimeout, UnexpectedResponse) as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Search failed after {max_retries} attempts: {e}")
                    return []
            except Exception as e:
                logger.error(f"Unexpected error in search: {e}")
                return []
        return []

    def predict(self, data: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Predict entities for a list of mentions.
        data: List of dicts with keys 'mention', 'context_left', 'context_right'
        """
        # Create a temporary dataset
        class TempDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                d = self.data[idx]
                return ContextualMention(
                    ctx_l=d.get('context_left', ''),
                    ctx_r=d.get('context_right', ''),
                    mention=d['mention'],
                    cuis=d.get('cuis', [])
                )

        ds = TempDataset(data)
        
        # Generate mention vectors
        logger.info("Generating mention vectors...")
        mentions_tensor = self.retriever.generate_mention_vectors(ds)
        
        results = []
        logger.info("Retrieving top hits from Qdrant using Search Groups...")
        
        vectors = mentions_tensor.numpy().tolist()
        chunk_size = self.search_batch_size
        
        # Using ThreadPoolExecutor specifically for I/O bound Qdrant queries
        # We process in chunks to control concurrency depth and memory usage
        from concurrent.futures import ThreadPoolExecutor
        
        for i in range(0, len(vectors), chunk_size):
            chunk_vectors = vectors[i : i + chunk_size]
            
            chunk_group_results = []
            
            # Execute queries concurrently for this chunk
            with ThreadPoolExecutor(max_workers=chunk_size) as executor:
                # Prepare all tasks
                futures = [
                    executor.submit(self._query_qdrant, vec, top_k) 
                    for vec in chunk_vectors
                ]
                
                # Collect results in order
                for future in futures:
                    chunk_group_results.append(future.result())
            
            # Process results for this chunk
            for j, groups in enumerate(chunk_group_results):
                if not groups: continue
                
                final_candidates = []
                for group in groups:
                    if not group.hits: continue
                    hit = group.hits[0]
                    payload = hit.payload or {}
                    final_candidates.append({
                        'cui': payload.get('cui'),
                        'name': payload.get('name'),
                        'definition': payload.get('definition'),
                        'icd': payload.get('icd'),
                        'semantic_types': payload.get('semantic_types'),
                        'score': hit.score
                    })
                
                # Map back to mention text
                original_idx = i + j
                if original_idx < len(data):
                    mention_text = data[original_idx]['mention']
                    lut_cuis = self.lut.get(mention_text, [])
                    
                    results.append({
                        'mention': mention_text,
                        'candidates': final_candidates,
                        'lut_hits': lut_cuis
                    })
                    
        return results