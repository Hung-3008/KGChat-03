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
        device: str = "cuda"
    ):
        self.model_name_or_path = model_name_or_path
        self.entity_list_names = entity_list_names
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

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
        
        # Original logic: concatenate mention vector with itself (dim=1)
        # This matches the prototype + knowledge vector structure in the index (vector_size * 2)
        mentions_tensor = torch.cat([mentions_tensor, mentions_tensor], dim=1)
        
        results = []
        logger.info("Retrieving top hits from Qdrant (Batch)...")
        
        # Prepare batch requests
        search_limit = top_k * 10
        vectors = mentions_tensor.numpy().tolist()
        
        requests = [
            models.SearchRequest(
                vector=v,
                limit=search_limit,
                with_payload=True
            ) for v in vectors
        ]
        
        batch_results = self.qdrant.client.search_batch(
            collection_name=self.collection_name,
            requests=requests
        )
        
        for i, hits in enumerate(batch_results):
            final_candidates = []
            seen_cuis = set()
            
            for hit in hits:
                if len(final_candidates) >= top_k:
                    break
                
                payload = hit.payload or {}
                cui = payload.get('cui')
                
                # Deduplicate by CUI
                if cui in seen_cuis:
                    continue
                
                seen_cuis.add(cui)
                final_candidates.append({
                    'cui': cui,
                    'name': payload.get('name'),
                    'definition': payload.get('definition'),
                    'icd': payload.get('icd'),
                    'score': hit.score
                })
            
            mention_text = data[i]['mention']
            lut_cuis = self.lut.get(mention_text, [])
            
            results.append({
                'mention': mention_text,
                'candidates': final_candidates,
                'lut_hits': lut_cuis
            })
            
        return results