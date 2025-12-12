"""Pipeline: split text into sentences by . ? !; index sentences; build buffered groups around each sentence; compute embeddings for sentences using the chosen encoder; represent each group by the mean of its sentence embeddings; iteratively merge adjacent groups when cosine similarity >= similarity_threshold; finally split groups at boundaries where adjacent sentence similarity < split_threshold; return list of chunked paragraph strings."""

from typing import List, Optional
import re

import numpy as np
import torch

from backend.encoders.transformer_encoder import TransformerEncoder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


class SemanticChunker:
    def __init__(self, encoder_name: str = "transformer", embedding_model: Optional[str] = None, device: str = "cpu", buffer: int = 1, similarity_threshold: float = 0.75, split_threshold: float = 0.6, batch_size: int = 32):
        self.encoder_name = encoder_name
        self.embedding_model = embedding_model
        self.device = device
        self.buffer = max(0, int(buffer))
        self.similarity_threshold = float(similarity_threshold)
        self.split_threshold = float(split_threshold)
        self.batch_size = int(batch_size)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        if self.encoder_name == "transformer":
            model_name = self.embedding_model or "NeuML/pubmedbert-base-embeddings"
            self._encoder = TransformerEncoder(model_name=model_name, device=self.device)
            return self._encoder
        raise ValueError("Unsupported encoder: %s" % self.encoder_name)

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        sentences = [s.strip() for s in re.split(r'(?<=[\.\?!])\s+', text.strip()) if s.strip()]
        if not sentences:
            return []
        encoder = self._get_encoder()
        sent_embs = encoder.embed(sentences, batch_size=self.batch_size)
        sent_embs_np = sent_embs.numpy()
        n = len(sentences)
        if n == 1:
            return [sentences[0]]
        sims = []
        for i in range(n - 1):
            a_start = max(0, i - self.buffer)
            a_end = min(n, i + self.buffer + 1)
            b_start = max(0, i + 1 - self.buffer)
            b_end = min(n, i + 1 + self.buffer + 1)
            a_emb = np.mean(sent_embs_np[a_start:a_end, :], axis=0)
            b_emb = np.mean(sent_embs_np[b_start:b_end, :], axis=0)
            sims.append(_cosine(a_emb, b_emb))
        boundaries = [i + 1 for i, s in enumerate(sims) if s < self.split_threshold]
        chunks = []
        start = 0
        for b in boundaries:
            chunks.append(" ".join(sentences[start:b]))
            start = b
        chunks.append(" ".join(sentences[start:]))
        if self.similarity_threshold is not None:
            # optional merge small adjacent chunks if they are very similar
            merged = []
            i = 0
            while i < len(chunks):
                if i + 1 < len(chunks):
                    a_emb = np.mean(encoder.embed([chunks[i]]).numpy(), axis=0)
                    b_emb = np.mean(encoder.embed([chunks[i + 1]]).numpy(), axis=0)
                    if _cosine(a_emb, b_emb) >= self.similarity_threshold:
                        chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                        i += 1
                        continue
                merged.append(chunks[i])
                i += 1
            chunks = merged
        return [c for c in chunks if c.strip()]


__all__ = ["SemanticChunker"]
