# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


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


logger = logging.getLogger()


@dataclass
class Mention:
    cui: str
    start: int
    end: int
    text: str
    types: str


@dataclass
class ContextualMention:
    mention: str
    cuis: List[str]
    ctx_l: str
    ctx_r: str

    def to_tensor(self, tokenizer: PreTrainedTokenizer, max_length: int) -> T:
        ctx_l_ids = tokenizer.encode(
            text=self.ctx_l,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        ctx_r_ids = tokenizer.encode(
            text=self.ctx_r,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        mention_ids = tokenizer.encode(
            text=self.mention,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        # Concatenate context and mention to the max length.
        token_ids = tokenizer.convert_tokens_to_ids(['<ENT>']) + mention_ids \
            + tokenizer.convert_tokens_to_ids(['</ENT>'])
        max_ctx_len = max_length - len(token_ids) - 2   # Exclude [CLS] and [SEP]
        max_ctx_l_len = max_ctx_len // 2
        max_ctx_r_len = max_ctx_len - max_ctx_l_len
        if len(ctx_l_ids) < max_ctx_l_len and len(ctx_r_ids) < max_ctx_r_len:
            token_ids = ctx_l_ids + token_ids + ctx_r_ids
        elif len(ctx_l_ids) >= max_ctx_l_len and len(ctx_r_ids) >= max_ctx_r_len:
            token_ids = ctx_l_ids[-max_ctx_l_len:] + token_ids \
                + ctx_r_ids[:max_ctx_r_len]
        elif len(ctx_l_ids) >= max_ctx_l_len:
            ctx_l_len = max_ctx_len - len(ctx_r_ids)
            token_ids = ctx_l_ids[-ctx_l_len:] + token_ids + ctx_r_ids
        else:
            ctx_r_len = max_ctx_len - len(ctx_l_ids)
            token_ids = ctx_l_ids + token_ids + ctx_r_ids[:ctx_r_len]

        token_ids = [tokenizer.cls_token_id] + token_ids

        # The above snippet doesn't guarantee the max length limit.
        token_ids = token_ids[:max_length - 1] + [tokenizer.sep_token_id]

        if len(token_ids) < max_length:
            token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))

        return torch.tensor(token_ids)


@dataclass
class Document:
    id: str = None
    title: str = None
    abstract: str = None
    mentions: List[Mention] = field(default_factory=list)

    def concatenate_text(self) -> str:
        return ' '.join([self.title, self.abstract])

    @classmethod
    def from_PubTator(cls, path: str, split_path_prefix: str) -> Dict[str, List]:
        docs = []
        with gzip.open(path, 'rb') as f:
            for b in f.read().decode().strip().split('\n\n'):
                d = cls()
                s = ''
                for i, ln in enumerate(b.split('\n')):
                    if i == 0:
                        id, type, text = ln.strip().split('|', 2)
                        assert type == 't'
                        d.id, d.title = id, text
                    elif i == 1:
                        id, type, text = ln.strip().split('|', 2)
                        assert type == 'a'
                        assert d.id == id
                        d.abstract = text
                        s = d.concatenate_text()
                    else:
                        items = ln.strip().split('\t')
                        assert d.id == items[0]
                        cui = items[5].split('UMLS:')[-1]
                        assert len(cui) == 8, breakpoint()
                        m = Mention(
                            cui=cui,
                            start=int(items[1]),
                            end=int(items[2]),
                            text=items[3],
                            types=items[4].split(',')
                        )
                        assert m.text == s[m.start: m.end]
                        d.mentions.append(m)
                docs.append(d)
        dataset = split_dataset(docs, split_path_prefix)
        print_dataset_stats(dataset)
        return dataset

    def to_contextual_mentions(self, max_length: int = 64) -> List[ContextualMention]:
        text = self.concatenate_text()
        mentions = []
        for m in self.mentions:
            assert m.text == text[m.start:m.end]
            # Context
            ctx_l, ctx_r = text[:m.start].strip().split(), text[m.end:].strip().split()
            ctx_l, ctx_r = ' '.join(ctx_l[-max_length:]), ' '.join(ctx_r[:max_length])
            cm = ContextualMention(
                mention=m.text,
                cuis=[m.cui],
                ctx_l=ctx_l,
                ctx_r=ctx_r,
            )
            mentions.append(cm)
        return mentions


def split_dataset(docs: List, split_path_prefix: str) -> Dict[str, List]:
    split_kv = {'train': 'trng', 'dev': 'dev', 'test': 'test'}
    id_to_split = {}
    dataset = {}
    for k, v in split_kv.items():
        dataset[k] = []
        path = split_path_prefix + v + '.txt'
        for i in open(path, encoding='utf-8').read().strip().split('\n'):
            assert i not in id_to_split, breakpoint()
            id_to_split[i] = k
    for doc in docs:
        split = id_to_split[doc.id]
        dataset[split].append(doc)
    return dataset


def print_dataset_stats(dataset: Dict[str, List[Document]]) -> None:
    all_docs = []
    for v in dataset.values():
        all_docs.extend(v)
    for split, docs in {'all': all_docs, **dataset}.items():
        logger.info(f"***** {split} *****")
        logger.info(f"Documents: {len(docs)}")
        logger.info(f"Mentions: {sum(len(d.mentions) for d in docs)}")
        cuis = set()
        for d in docs:
            for m in d.mentions:
                cuis.add(m.cui)
        logger.info(f"Mentioned concepts: {len(cuis)}")


class MedMentionsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path: str, split: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.docs = Document.from_PubTator(
            path=os.path.join(self.dataset_path, 'corpus_pubtator.txt.gz'),
            split_path_prefix=os.path.join(self.dataset_path, 'corpus_pubtator_pmids_')
        )[split]
        self.mentions = []
        self.name_to_cuis = {}
        self._post_init()

    def _post_init(self):
        for d in self.docs:
            self.mentions.extend(d.to_contextual_mentions())
        for m in self.mentions:
            if m.mention not in self.name_to_cuis:
                self.name_to_cuis[m.mention] = set()
            self.name_to_cuis[m.mention].update(m.cuis)

    def __getitem__(self, index: int) -> ContextualMention:
        return self.mentions[index]

    def __len__(self) -> int:
        return len(self.mentions)


class PreprocessedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.file = dataset_path
        self.data = []
        self.load_data()

    def load_data(self) -> None:
        with open(self.file, encoding='utf-8') as f:
            logger.info("Reading file %s" % self.file)
            for ln in f:
                if ln.strip():
                    self.data.append(json.loads(ln))
        logger.info("Loaded data size: {}".format(len(self.data)))

    def __getitem__(self, index: int) -> ContextualMention:
        d = self.data[index]
        return ContextualMention(
            ctx_l=d['context_left'],
            ctx_r=d['context_right'],
            mention=d['mention'],
            cuis=d['cuis'],
        )

    def __len__(self) -> int:
        return len(self.data)


def generate_vectors(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    max_length: int,
    is_prototype: bool = False,
):
    n = len(dataset)
    total = 0
    results = []
    start_time = time.time()
    logger.info("Start encoding...")
    for i, batch_start in enumerate(range(0, n, batch_size)):
        batch = [dataset[i] for i in range(batch_start, min(n, batch_start + batch_size))]
        batch_token_tensors = [m.to_tensor(tokenizer, max_length) for m in batch]

        ids_batch = torch.stack(batch_token_tensors, dim=0).to(encoder.device)
        seg_batch = torch.zeros_like(ids_batch)
        attn_mask = (ids_batch != tokenizer.pad_token_id)

        with torch.inference_mode():
            out = encoder(
                input_ids=ids_batch,
                token_type_ids=seg_batch,
                attention_mask=attn_mask
            )
            out = out[0][:, 0, :]
        out = out.cpu()

        num_mentions = out.size(0)
        total += num_mentions

        if is_prototype:
            meta_batch = [{'cuis': m.cuis} for m in batch]
            assert len(meta_batch) == num_mentions
            results.extend([(meta_batch[i], out[i].view(-1).numpy()) for i in range(num_mentions)])
        else:
            results.extend(out.cpu().split(1, dim=0))

        if (i + 1) % 10 == 0:
            eta = (n - total) * (time.time() - start_time) / 60 / total
            logger.info(f"Batch={i + 1}, Encoded mentions={total}, ETA={eta:.1f}m")

    assert len(results) == n
    logger.info(f"Total encoded mentions={n}")
    if not is_prototype:
        results = torch.cat(results, dim=0)

    return results

