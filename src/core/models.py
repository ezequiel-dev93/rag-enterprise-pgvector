from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(description="UUID único del documento")
    filename: str = Field(description="Nombre original del archivo")
    content_hash: str = Field(description="Hash SHA-256 para evitar reprocesar duplicados")
    total_pages: int = Field(default=1, description="Cantidad de páginas del documento")
    total_chunks: int = Field(default=0, description="Cantidad de fragmentos indexados")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Fecha y hora de ingesta",
    )


class DocumentChunk(BaseModel):
    id: str = Field(description="UUID único del fragmento")
    document_id: str = Field(description="ID del documento padre")
    document_name: str = Field(description="Nombre del documento para citaciones")
    chunk_index: int = Field(description="Posición secuencial del fragmento")
    page_number: int = Field(default=1, description="Número de página de origen")
    content: str = Field(description="Texto del fragmento")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector de embedding (ej. 768 dimensiones)"
    )


class SearchResult(BaseModel):
    chunk: DocumentChunk
    similarity_score: float = Field(
        description="Puntaje de similitud (1.0 = idéntico, 0.0 = opuesto)"
    )


class Citation(BaseModel):
    index: int = Field(description="Número de cita [^1], [^2], etc.")
    document_name: str = Field(description="Archivo de donde proviene la información")
    page_number: int = Field(description="Página de donde se extrajo")
    excerpt: str = Field(description="Texto exacto citado como soporte de la respuesta")


class RAGResponse(BaseModel):
    answer: str = Field(description="Respuesta redactada fundamentada en los documentos")
    citations: List[Citation] = Field(
        default_factory=list, description="Lista de fuentes verificables citadas"
    )
    retrieved_chunks: List[SearchResult] = Field(
        default_factory=list, description="Fragmentos recuperados del almacenamiento vectorial"
    )
    has_sufficient_context: bool = Field(
        default=True,
        description="Si la información en la base fue suficiente para responder sin alucinar",
    )
    latency_seconds: float = Field(
        default=0.0, description="Tiempo de procesamiento total en segundos"
    )
    top_similarity: float = Field(
        default=0.0, description="Porcentaje de similitud más alto recuperado"
    )
    estimated_tokens: int = Field(
        default=0, description="Tokens aproximados procesados en la consulta"
    )
