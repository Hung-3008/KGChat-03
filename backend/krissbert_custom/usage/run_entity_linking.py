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
from typing import List, Tuple, Dict, Iterator, Set
from functools import partial
from multiprocessing.dummy import Pool

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
import faiss

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
)
from utils import generate_vectors


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
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(
        self, query_vectors: np.array, top_docs: int, batch_size: int = 4096,
    ) -> List[Tuple[List[object], List[float]]]:
        num_queries = query_vectors.shape[0]
        scores, indexes = [], []
        for start in range(0, num_queries, batch_size):
            logger.info(f"Searched {start} queries.")
            batch_vectors = query_vectors[start:start + batch_size]
            batch_scores, batch_indexes = self.index.search(batch_vectors, top_docs)
            scores.extend(batch_scores)
            indexes.extend(batch_indexes)
        # convert to external ids
        # Filter out -1 indices which FAISS returns when there aren't enough results
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
        """
        Indexes encoded data takes form a list of files
        :param vector_files: a list of files
        :param buffer_size: size of a buffer to send for the indexing at once
        :return:
        """
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
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_hits(
        self, mention_vectors: np.array, top_k: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching given the mention vectors batch
        """
        time0 = time.time()
        search = partial(
            self.index.search_knn,
            top_docs=top_k,
        )
        results = []
        num_processes = multiprocessing.cpu_count()
        shard_size = math.ceil(mention_vectors.shape[0] / num_processes)
        shards = []
        for i in range(0, mention_vectors.shape[0], shard_size):
            shards.append(mention_vectors[i:i + shard_size])
        with Pool(processes=num_processes) as pool:
            it = pool.map(search, shards)
            for ret in it:
                results += ret
            # results = self.index.search_knn(mention_vectors, top_k)
        logger.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results


def hit(pred: List[str], gold: List[str]) -> bool:
    return all(p in gold for p in pred)


def dedup_ids(ids: List[Dict]) -> List[Dict]:
    deduped_ids = []
    seen_cuis = set()
    for d in ids:
        cui = d['cui']
        if cui in seen_cuis:
            continue
        seen_cuis.add(cui)
        deduped_ids.append(d)
    return deduped_ids


def evaluate(
    ds: torch.utils.data.Dataset,
    result_ent_ids: List[Tuple[List[object], List[float]]],
    lookup_table: str,
    top_ks: List[int] = (1, 5, 50, 100),
) -> List[Dict]:
    lut = {}
    with open(lookup_table, encoding='utf-8') as f:
        for ln in f:
            cuis, name = ln.strip().split('||')
            cuis = cuis.split('|')
            lut[name] = cuis

    n = len(ds)
    top_k_hits = {top_k: 0 for top_k in top_ks}
    for i in range(len(result_ent_ids)):
        d = ds[i]
        ids, _ = result_ent_ids[i]
        ids = dedup_ids(ids)
        ids = ids[:max(top_ks)]
        candidates = [
            {'cui': eid['cui'], 'hit': int(hit(pred=[eid['cui']], gold=d.cuis))}
            for eid in ids
        ]
        lut_cuis = lut.get(d.mention, [])
        if len(lut_cuis) == 1:
            # If the mention only has one ID in the look up table,
            # we use the ID as the top prediction.
            candidates.insert(
                0,
                {'cuis': lut_cuis, 'hit': int(hit(pred=lut_cuis, gold=d.cuis))}
            )
        for top_k in top_k_hits:
            if any(c['hit'] for c in candidates[:top_k]):
                top_k_hits[top_k] += 1

    top_k_acc = {top_k: v / n for top_k, v in top_k_hits.items()}
    logger.info("Top-k accuracy %s", top_k_acc)


@hydra.main(config_path="conf", config_name="run_linking", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    logger.info("Configuration:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    # Load pretrained.
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
    vector_size = config.hidden_size
    logger.info("Encoder vector_size=%d", vector_size)

    # Load test data.
    ds = hydra.utils.instantiate(cfg.test_data)

    # Init indexer.
    index = DenseFlatIndexer()
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)

    # candidate ids
    candidate_ids = None
    if cfg.entity_list_ids:
        with open(cfg.entity_list_ids, encoding='utf-8') as f:
            candidate_ids = set(f.read().split('\n'))

    # Start indexing
    input_paths = []
    for pattern in cfg.encoded_files:
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    input_paths = sorted(set(input_paths))

    retriever = FaissRetriever(
        encoder, tokenizer, cfg.batch_size, cfg.max_length, index)
    mentions_tensor = retriever.generate_mention_vectors(ds)


    index_path = cfg.index_path
    if index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        logger.info("Indexing encoded data from files: %s", input_paths)
        retriever.index_encoded_data(
            vector_files=input_paths,
            buffer_size=index_buffer_sz,
            candidate_ids=candidate_ids,
        )
        if index_path:
            pathlib.Path(os.path.dirname(index_path)).mkdir(
                parents=True, exist_ok=True)
            retriever.index.serialize(index_path)

    # Encode test data.
    # mentions_tensor already has the correct dimension

    # To get k different entities, we retrieve 32 * k mentions and then dedup.
    top_ids_and_scores = retriever.get_top_hits(
        mentions_tensor.numpy(), cfg.num_retrievals * 32)
    evaluate(ds, top_ids_and_scores, cfg.entity_list_names)
    print("\n=== SAMPLE ENTITY LINKING RESULTS ===\n")

    test_mentions = ds.mentions

    num_samples = 5
    for i in range(min(num_samples, len(test_mentions))):
        m = test_mentions[i]

        mention_text = m.mention
        gold_cuis = m.cuis                # list of CUIs
        pred_metadata = top_ids_and_scores[i][0][0] if top_ids_and_scores[i][0] else None
        pred_cui = pred_metadata['cui'] if pred_metadata else None

        print(f"Sample {i+1}:")
        print(f" MENTION: {mention_text}")
        print(f" GOLD CUIS: {gold_cuis}")
        print(f" PRED CUI: {pred_cui}")
        print("-" * 50)

if __name__ == "__main__":
    main()