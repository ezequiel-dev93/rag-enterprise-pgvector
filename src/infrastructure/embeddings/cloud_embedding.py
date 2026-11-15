import os
import threading
from typing import List

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.core.ports import IEmbeddingProvider

_EMBEDDING_LOCK = threading.Lock()


def _sanitize_error(err: Exception, api_key: str) -> str:
    msg = str(err)
    if api_key and len(api_key) > 4:
        msg = msg.replace(api_key, "[REDACTED_API_KEY]")
    return msg


class CloudEmbeddingProvider(IEmbeddingProvider):
    """
    Adaptador que genera embeddings semánticos (text-embedding-004) con aislamiento
    efímero de credenciales (Zero-Retention en estado global de proceso).
    """

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
        self._api_key = (
            api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        ).strip()
        self.model_name = model_name

    def __repr__(self) -> str:
        return f"<CloudEmbeddingProvider model={self.model_name} key=REDACTED>"

    def generate_embedding(self, text: str) -> List[float]:
        if not self._api_key:
            raise ValueError("LLM_API_KEY no está configurada.")

        with _EMBEDDING_LOCK:
            try:
                genai.configure(api_key=self._api_key)
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query",
                )
                return result["embedding"]
            except Exception as e:
                raise RuntimeError(
                    f"Error al generar embedding en el proveedor LLM: {_sanitize_error(e, self._api_key)}"
                ) from None
            finally:
                genai.configure(api_key="cleared")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._api_key:
            raise ValueError("LLM_API_KEY no está configurada.")

        results = []
        batch_size = 20
        with _EMBEDDING_LOCK:
            try:
                genai.configure(api_key=self._api_key)
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    response = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="retrieval_document",
                    )
                    results.extend(response["embedding"])
                return results
            except Exception as e:
                raise RuntimeError(
                    f"Error en batch de embeddings del proveedor LLM: {_sanitize_error(e, self._api_key)}"
                ) from None
            finally:
                genai.configure(api_key="cleared")
