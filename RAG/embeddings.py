import re, math, hashlib, warnings
from typing import List, Dict

warnings.filterwarnings("ignore")

from langchain_core.embeddings import Embeddings


class TFIDFEmbeddings(Embeddings):
    DIM = 256

    def _hash_term(self, term: str) -> int:
        return int(hashlib.md5(term.encode()).hexdigest(), 16) % self.DIM

    def _embed(self, text: str) -> List[float]:
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        if not words:
            return [0.0] * self.DIM
        tf: Dict[str, float] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        total = sum(tf.values())
        vec = [0.0] * self.DIM
        for term, cnt in tf.items():
            freq = cnt / total
            idf  = 1.0 / (1 + math.log(1 + freq * 100))
            vec[self._hash_term(term)] += freq * idf
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)
