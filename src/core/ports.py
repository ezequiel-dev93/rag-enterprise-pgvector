from abc import ABC, abstractmethod
from typing import List, Tuple
from src.core.models import Document, DocumentChunk, RAGResponse, SearchResult


class IDocumentParser(ABC):
    """Puerto para extracción de texto estructurado página por página."""

    @abstractmethod
    def extract_pages(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """Extrae el contenido de un archivo retornando una lista de tuplas (num_pagina, texto)."""
        pass


class IEmbeddingProvider(ABC):
    """Puerto para generación de embeddings numéricos vectoriales."""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Genera el vector de embedding para un fragmento de texto."""
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings en lote para optimizar latencia de red."""
        pass


class IVectorRepository(ABC):
    """Puerto para persistencia de documentos y búsqueda por similitud vectorial."""

    @abstractmethod
    def init_storage(self) -> None:
        """Crea tablas e índices HNSW en la base de datos si no existen."""
        pass

    @abstractmethod
    def save_document(self, document: Document) -> None:
        """Registra los metadatos de un nuevo documento."""
        pass

    @abstractmethod
    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Almacena en lote los fragmentos con sus vectores embebidos."""
        pass

    @abstractmethod
    def search_similar_chunks(
        self,
        query_vector: List[float],
        top_k: int = 4,
        min_similarity: float = 0.5,
        filter_filenames: List[str] | None = None,
    ) -> List[SearchResult]:
        """Recupera los fragmentos más cercanos semánticamente usando similitud coseno, con filtro opcional por documento."""
        pass

    @abstractmethod
    def list_documents(self) -> List[Document]:
        """Lista todos los documentos indexados en el sistema."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Elimina un documento y todos sus fragmentos asociados en cascada."""
        pass


class ILLMProvider(ABC):
    """Puerto para síntesis de respuestas con citación obligatoria de fuentes."""

    @abstractmethod
    def generate_grounded_answer(
        self, query: str, context_chunks: List[DocumentChunk]
    ) -> RAGResponse:
        """Sintetiza la respuesta basándose exclusivamente en el contexto provisto."""
        pass
