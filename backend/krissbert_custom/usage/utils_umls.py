from typing import List, Dict
import os
import time
import logging
import json
import gzip
from dataclasses import dataclass, field

import torch
from torch import Tensor as T
from transformers import PreTrainedTokenizer


@dataclass
class ContextualUMLS:
    cui: str
    stn: str
    Type: str
    aliases: List[str]
    

    def to_tensors(self, tokenizer: PreTrainedTokenizer, max_length: int) -> List[T]:
        if not self.aliases:
            return []
        
        # Encode stn and Type once (shared across all aliases)
        stn_ids = tokenizer.encode(
            text=self.stn,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        
        type_ids = tokenizer.encode(
            text=self.Type,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        
        # Create one tensor per alias
        tensors = []
        for alias in self.aliases:
            # Encode this specific alias
            alias_ids = tokenizer.encode(
                text=alias,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            
            # Build sequence: [CLS] stn [SEP] Type [SEP] alias [SEP]
            token_ids = [tokenizer.cls_token_id]
            token_ids += stn_ids
            token_ids += [tokenizer.sep_token_id]
            token_ids += type_ids
            token_ids += [tokenizer.sep_token_id]
            
            # Calculate space for this alias
            remaining_space = max_length - len(token_ids) - 1  # Reserve for final [SEP]
            
            if len(alias_ids) > remaining_space:
                alias_ids = alias_ids[:remaining_space]  # Truncate if needed
            
            token_ids += alias_ids
            token_ids += [tokenizer.sep_token_id]
            
            # Pad to max_length
            if len(token_ids) < max_length:
                token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))
            
            # Ensure exact max_length
            token_ids = token_ids[:max_length]
            
            tensors.append(torch.tensor(token_ids))
        
        return tensors

logger = logging.getLogger()

class PreprocessedUMLS(torch.utils.data.Dataset):
    """
    PyTorch Dataset for preprocessed UMLS data.
    Load UMLS concepts from JSON file.
    """
    
    def __init__(self, UMLS_path: str) -> None:
        super().__init__()
        self.file = UMLS_path
        self.data = []
        self.load_umls_json()
    
    def load_umls_json(self) -> None:
        """Load UMLS data from JSON file."""
        with open(self.file, 'r', encoding='utf-8') as f:
            logger.info(f"Reading UMLS file {self.file}")
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} UMLS concepts")
    
    def __getitem__(self, index: int) -> ContextualUMLS:
        """Get a single UMLS concept as ContextualUMLS."""
        d = self.data[index]
        return ContextualUMLS(
            cui=d['cui'],
            stn=d['stn'],
            Type=d['type'],
            aliases=d.get('aliases', [])
        )
    
    def __len__(self) -> int:
        """Return total number of UMLS concepts."""
        return len(self.data)



# ============================================================================
# Vector Generation Function
# ============================================================================

def generate_vectors(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    max_length: int,
    is_prototype: bool = False,
    start_index: int = 0,
    duck_con=None,
    duck_query: str = None,
):
    """
    Encode UMLS concepts into dense vectors.
    
    Encodes ALL aliases of each concept - each alias gets its own vector.
    
    Args:
        encoder: BERT/BioBERT model
        tokenizer: Tokenizer
        dataset: PyTorch Dataset (PreprocessedUMLS)
        batch_size: Batch size (number of ALIASES per batch, not concepts)
        max_length: Max sequence length
        is_prototype: If True, return (metadata, vector) tuples
        
    Yields:
        If is_prototype=False: Tensor [batch_size, hidden_size]
        If is_prototype=True: List of (metadata, vector) tuples for each alias in the batch
    """
    device = next(encoder.parameters()).device
    
    # Flatten all aliases from all concepts
    # If using DuckDB, we don't need to flatten manually, the query should return flat aliases
    all_alias_data = []
    
    if duck_con and duck_query:
        # We will iterate cursor directly, so we don't build all_alias_data
        pass
    else:
        for idx in range(len(dataset)):
            concept = dataset[idx]
            for alias_idx, alias in enumerate(concept.aliases):
                all_alias_data.append({
                    'concept': concept,
                    'alias': alias,
                    'alias_idx': alias_idx
                })
    
    if duck_con and duck_query:
        # For DuckDB, we assume the query returns (cui, stn, type, alias)
        # We can't easily know 'n' without a count query, but we can just iterate.
        n = "Unknown" 
        total = start_index
    else:
        n = len(all_alias_data)
        total = start_index
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info(f"Start encoding UMLS aliases...")
    if dataset:
        logger.info(f"Total concepts: {len(dataset)}")
    else:
        logger.info(f"Total concepts: Unknown (Streaming from DuckDB)")
    logger.info(f"Total aliases: {n}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 80)
    
    logger.info(f"Resuming from index {start_index}..." if start_index > 0 else "Starting from beginning...")
    
    if duck_con and duck_query:
        # Execute query using a separate cursor to avoid interference
        cursor = duck_con.cursor()
        cursor.execute(duck_query)
        
        while True:
            # Fetch batch
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            batch_data = []
            for row in rows:
                # row: (cui, stn, type, alias)
                # We need to construct 'concept' object or similar structure expected below
                # The code below expects item['concept'] with .cui, .stn, .Type attributes
                # and item['alias']
                
                cui, stn, type_, alias = row
                
                # Create a dummy concept object
                @dataclass
                class SimpleConcept:
                    cui: str
                    stn: str
                    Type: str
                
                concept = SimpleConcept(cui=cui, stn=stn, Type=type_)
                
                batch_data.append({
                    'concept': concept,
                    'alias': alias,
                    'alias_idx': 0 # Dummy index, not used for vectors
                })
            
            # Process batch (shared logic)
            yield from process_batch(batch_data, encoder, tokenizer, max_length, device, is_prototype)
            
            total += len(rows)
            if total % 1000 == 0:
                 logger.info(f"Encoded {total} aliases...")

    else:
        for i, batch_start in enumerate(range(start_index, n, batch_size)):
            # Get batch of alias data
            batch_data = all_alias_data[batch_start:min(n, batch_start + batch_size)]
            yield from process_batch(batch_data, encoder, tokenizer, max_length, device, is_prototype)
            
            # Log progress
            if (i + 1) % 10 == 0:
                eta = (n - total) * (time.time() - start_time) / 60 / total if total > 0 else 0
                logger.info(f"Batch={i+1}, Encoded={total}/{n} aliases, ETA={eta:.1f}m")

    logger.info("=" * 80)
    logger.info(f"✅ Encoding completed!")
    logger.info(f"Total aliases encoded: {total}")
    logger.info(f"Total time: {(time.time() - start_time) / 60:.2f}m")
    logger.info("=" * 80)

def process_batch(batch_data, encoder, tokenizer, max_length, device, is_prototype):
    # Encode each alias
    batch_tensors = []
    for item in batch_data:
        concept = item['concept']
        alias = item['alias']
        
        # Create ContextualUMLS with single alias
        single_alias_concept = ContextualUMLS(
            cui=concept.cui,
            stn=concept.stn,
            Type=concept.Type,
            aliases=[alias]  # Only this one alias
        )
        
        # Get tensor for this alias
        tensors = single_alias_concept.to_tensors(tokenizer, max_length)
        if tensors:
            batch_tensors.append(tensors[0])
        else:
            # Fallback to zero tensor
            batch_tensors.append(torch.zeros(max_length, dtype=torch.long))
    
    if not batch_tensors:
        return
    
    # Stack and move to device
    ids_batch = torch.stack(batch_tensors, dim=0).to(device)
    seg_batch = torch.zeros_like(ids_batch)
    attn_mask = (ids_batch != tokenizer.pad_token_id)
    
    # Encode
    with torch.inference_mode():
        out = encoder(input_ids=ids_batch, token_type_ids=seg_batch, attention_mask=attn_mask)
        out = out[0][:, 0, :]  # [CLS] token
    out = out.cpu()
    
    num = out.size(0)
    
    # Store results
    if is_prototype:
        meta = [
            {
                'cui': item['concept'].cui,
                'stn': item['concept'].stn,
                'type': item['concept'].Type,
                'alias': item['alias'],
                'alias_idx': item['alias_idx']
            }
            for item in batch_data
        ]
        batch_results = [(meta[j], out[j].view(-1).numpy()) for j in range(num)]
    else:
        batch_results = out.cpu().split(1, dim=0)
    
    yield batch_results
