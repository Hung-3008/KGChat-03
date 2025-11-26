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
import faiss

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
)
from utils import generate_vectors, ContextualMention


# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(
        self, query_vectors: np.array, top_docs: int
    ) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".index.dpr"
            meta_file = path + ".index_meta.dpr"
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [
                np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]
            ]
            vectors = np.concatenate(vectors, axis=0)
            self._update_id_mapping(db_ids)
            self.index.add(vectors)
            
        logger.info("Total data indexed %d", len(self.index_id_to_db_id))

    def search_knn(
        self, query_vectors: np.array, top_docs: int, batch_size: int = 4096,
    ) -> List[Tuple[List[object], List[float]]]:
        num_queries = query_vectors.shape[0]
        scores, indexes = [], []
        for start in range(0, num_queries, batch_size):
            batch_vectors = query_vectors[start:start + batch_size]
            batch_scores, batch_indexes = self.index.search(batch_vectors, top_docs)
            scores.extend(batch_scores)
            indexes.extend(batch_indexes)
        
        # convert to external ids
        db_ids = []
        valid_scores = []
        for query_top_idxs, query_scores in zip(indexes, scores):
            valid_indices = [i for i in query_top_idxs if i >= 0 and i < len(self.index_id_to_db_id)]
            valid_db_ids = [self.index_id_to_db_id[i] for i in valid_indices]
            valid_query_scores = [query_scores[idx] for idx, i in enumerate(query_top_idxs) if i >= 0 and i < len(self.index_id_to_db_id)]
            db_ids.append(valid_db_ids)
            valid_scores.append(valid_query_scores)
        result = [(db_ids[i], valid_scores[i]) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"


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


class FaissRetriever(DenseRetriever):
    """
    Does entity retrieving over the provided index and encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        index: DenseIndexer,
    ):
        super().__init__(encoder, tokenizer, batch_size, max_length)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        candidate_ids: Set = None,
    ):
        buffer = []
        for file in vector_files:
            logger.info("Reading file %s", file)
            with open(file, "rb") as reader:
                for meta, vec in pickle.load(reader):
                    if candidate_ids:
                        cui = meta['cui']
                        if cui not in candidate_ids:
                            continue
                    buffer.append((meta, vec))
                    if 0 < buffer_size == len(buffer):
                        self.index.index_data(buffer)
                        buffer = []
        if buffer:
            self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_hits(
        self, mention_vectors: np.array, top_k: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching given the mention vectors batch
        """
        time0 = time.time()
        
        # Optimization: Skip multiprocessing for small batches
        if mention_vectors.shape[0] < 100:
            results = self.index.search_knn(mention_vectors, top_docs=top_k)
        else:
            search = partial(
                self.index.search_knn,
                top_docs=top_k,
            )
            results = []
            num_processes = min(multiprocessing.cpu_count(), 8) # Cap processes
            shard_size = math.ceil(mention_vectors.shape[0] / num_processes)
            shards = []
            for i in range(0, mention_vectors.shape[0], shard_size):
                shards.append(mention_vectors[i:i + shard_size])
            with Pool(processes=num_processes) as pool:
                it = pool.map(search, shards)
                for ret in it:
                    results += ret
                    
        logger.info("index search time: %f sec.", time.time() - time0)
        return results


class EntityLinker:
    def __init__(
        self,
        model_name_or_path: str,
        encoded_files: List[str],
        entity_list_names: str,
        index_path: str = None,
        batch_size: int = 256,
        max_length: int = 64,
        device: str = "cuda"
    ):
        self.model_name_or_path = model_name_or_path
        self.encoded_files = encoded_files
        self.entity_list_names = entity_list_names
        
        # Auto-infer index path if not provided to enable caching
        if index_path is None and encoded_files:
            # Use the first encoded file as base for index path
            base_path = encoded_files[0]
            # If it's a directory, use it directly, else append extension
            if os.path.isdir(base_path):
                 index_path = base_path
            else:
                 index_path = os.path.splitext(base_path)[0]
            logger.info(f"No index_path provided. Auto-inferred cache path: {index_path}")
            
        self.index_path = index_path
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
        
        self.vector_size = self.config.hidden_size
        
        # Init indexer
        self.index = DenseFlatIndexer()
        self.index.init_index(self.vector_size)
        
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

        # Index data
        self.retriever = FaissRetriever(
            self.encoder, self.tokenizer, self.batch_size, self.max_length, self.index
        )
        
        input_paths = []
        for pattern in self.encoded_files:
            pattern_files = glob.glob(pattern)
            input_paths.extend(pattern_files)
        input_paths = sorted(set(input_paths))
        
        if self.index_path and self.index.index_exists(self.index_path):
            logger.info("Loading index from %s", self.index_path)
            self.retriever.index.deserialize(self.index_path)
        else:
            logger.info("Indexing encoded data from files: %s", input_paths)
            self.retriever.index_encoded_data(
                vector_files=input_paths,
                buffer_size=self.index.buffer_size,
            )
            if self.index_path:
                pathlib.Path(os.path.dirname(self.index_path)).mkdir(parents=True, exist_ok=True)
                self.retriever.index.serialize(self.index_path)

    def predict(self, data: List[Dict], top_k: int = 1) -> List[Dict]:
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
        
        # Retrieve
        logger.info("Retrieving top hits...")
        # We retrieve more candidates to handle potential duplicates
        top_ids_and_scores = self.retriever.get_top_hits(mentions_tensor.numpy(), top_k * 5)
        
        results = []
        for i in range(len(data)):
            ids, scores = top_ids_and_scores[i]
            
            # Dedup
            final_candidates = []
            seen_cuis = set()
            
            for idx, d_meta in enumerate(ids):
                cui = d_meta['cui']
                if cui in seen_cuis:
                    continue
                seen_cuis.add(cui)
                
                final_candidates.append({
                    'cui': cui,
                    'score': float(scores[idx]) if idx < len(scores) else 0.0,
                    'metadata': d_meta
                })
                if len(final_candidates) >= top_k:
                    break
            
            mention_text = data[i]['mention']
            lut_cuis = self.lut.get(mention_text, [])
            
            results.append({
                'mention': mention_text,
                'candidates': final_candidates,
                'lut_hits': lut_cuis
            })
            
        return results