import hashlib
import math
from typing import List
from src.core.ports import IEmbeddingProvider


class MockEmbeddingProvider(IEmbeddingProvider):
    """Proveedor determinista de embeddings para tests unitarios y desarrollo offline a costo $0.
    Simula similitud semántica mediante bolsa de palabras y n-gramas.
    """

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def generate_embedding(self, text: str) -> List[float]:
        words = text.lower().replace(":", " ").replace(".", " ").replace(",", " ").split()
        if not words:
            return [0.0] * self.dimensions

        vector = [0.0] * self.dimensions

        for word in words:
            # Ponderar tokens relevantes
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            for i in range(16):
                idx = (digest[i] * 31 + i) % self.dimensions
                vector[idx] += 1.0

        # Normalizar vector (L2 norm)
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [round(x / norm, 6) for x in vector]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.generate_embedding(t) for t in texts]
