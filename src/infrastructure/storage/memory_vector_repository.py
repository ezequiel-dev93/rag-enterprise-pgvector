import math
from typing import Dict, List
from src.core.models import Document, DocumentChunk, SearchResult
from src.core.ports import IVectorRepository


class InMemoryVectorRepository(IVectorRepository):
    """Adaptador de almacenamiento en memoria para desarrollo local ágil y tests con $0 de costo."""

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.chunks: List[DocumentChunk] = []

    def init_storage(self) -> None:
        pass

    def save_document(self, document: Document) -> None:
        self.documents[document.id] = document

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.chunks.extend(chunks)

    def search_similar_chunks(
        self,
        query_vector: List[float],
        top_k: int = 4,
        min_similarity: float = 0.5,
        filter_filenames: List[str] | None = None,
    ) -> List[SearchResult]:
        scored: List[SearchResult] = []

        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            if filter_filenames and chunk.document_name not in filter_filenames:
                continue
            sim = self._cosine_similarity(query_vector, chunk.embedding)
            if sim >= min_similarity:
                scored.append(SearchResult(chunk=chunk, similarity_score=round(sim, 4)))

        # Ordenar de mayor a menor similitud
        scored.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored[:top_k]

    def list_documents(self) -> List[Document]:
        return list(self.documents.values())

    def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            self.chunks = [c for c in self.chunks if c.document_id != document_id]
            return True
        return False

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)
