import os
from typing import List

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.core.ports import IEmbeddingProvider


class GeminiEmbeddingProvider(IEmbeddingProvider):
    """Adaptador que genera embeddings usando el modelo oficial de Google text-embedding-004."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "models/text-embedding-004",
    ):
        if not HAS_GENAI:
            raise ImportError(
                "El paquete 'google-generativeai' no está instalado.\n"
                "Instálalo con: pip install google-generativeai"
            )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name

    def generate_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada.")

        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada.")

        # Batching en Google Gemini
        results = []
        # Agrupar en batches de 20 para evitar límites de payload
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="retrieval_document",
            )
            # response['embedding'] puede ser lista de listas
            results.extend(response["embedding"])
        return results
