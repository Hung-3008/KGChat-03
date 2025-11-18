from typing import List, Dict, Optional
from backend.encoders.transformer_encoder import TransformerEncoder
from backend.graph_extractor.schema import ExtractionOutput
from backend.graph_extractor.prompts import PROMPT_TEMPLATE


class NodeExtractor:
    def __init__(self, llm_client, model_name: str, embedding_model: str, encoder: Optional[TransformerEncoder] = None, device: str = "cpu"):
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.encoder = encoder or TransformerEncoder(model_name=embedding_model, device=device)

    def extract(self, text: str) -> List[Dict]:
        if not text or not text.strip():
            return []
        prompt = PROMPT_TEMPLATE + text
        try:
            resp = self.llm_client.generate(prompt=prompt, format=ExtractionOutput)
            entities = resp.get("entities", []) if isinstance(resp, dict) else []
        except Exception:
            return []
        if not entities:
            return []
        names = []
        mentions = []
        for e in entities:
            if isinstance(e, dict):
                name = e.get("name", "").strip()
                mention = e.get("mention", "").strip()
            else:
                name = getattr(e, "name", "").strip()
                mention = getattr(e, "mention", "").strip()
            if name and mention:
                names.append(name)
                mentions.append(mention)
        if not names or not mentions:
            return []
        try:
            name_embs = self.encoder.embed_to_numpy(names).tolist()
            mention_embs = self.encoder.embed_to_numpy(mentions).tolist()
        except Exception:
            return []
        out = []
        for i in range(len(names)):
            if i < len(name_embs) and i < len(mention_embs):
                out.append({
                    "name": names[i],
                    "mention": mentions[i],
                    "name_embedding": name_embs[i],
                    "mention_embedding": mention_embs[i],
                })
        return out
