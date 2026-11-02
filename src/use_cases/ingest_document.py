import hashlib
import uuid
from typing import Tuple
from src.core.models import Document
from src.core.ports import IDocumentParser, IEmbeddingProvider, IVectorRepository
from src.infrastructure.parsers.semantic_chunker import SemanticChunker


class IngestDocumentUseCase:
    """Caso de uso para procesar, fragmentar, vectorizar y persistir un documento."""

    def __init__(
        self,
        parser: IDocumentParser,
        chunker: SemanticChunker,
        embedder: IEmbeddingProvider,
        repository: IVectorRepository,
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.repository = repository

    def execute(self, file_bytes: bytes, filename: str) -> Tuple[Document, int]:
        # 1. Calcular hash SHA-256 para evitar duplicación
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        # Verificar si el documento ya fue indexado
        existing_docs = self.repository.list_documents()
        for doc in existing_docs:
            if doc.content_hash == content_hash:
                return doc, doc.total_chunks

        # 2. Extraer páginas de texto
        pages = self.parser.extract_pages(file_bytes, filename)
        doc_id = str(uuid.uuid4())

        # 3. Fragmentación semántica con metadatos de página
        chunks = self.chunker.create_chunks(
            pages=pages,
            document_id=doc_id,
            document_name=filename,
        )

        if not chunks:
            raise ValueError(f"El documento '{filename}' no contiene texto extraíble.")

        # 4. Generación de vectores en lote (Batch Embedding)
        texts_to_embed = [c.content for c in chunks]
        embeddings = self.embedder.generate_embeddings_batch(texts_to_embed)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 5. Persistencia del documento y sus fragmentos
        document = Document(
            id=doc_id,
            filename=filename,
            content_hash=content_hash,
            total_pages=len(pages),
            total_chunks=len(chunks),
        )

        self.repository.save_document(document)
        self.repository.save_chunks(chunks)

        return document, len(chunks)
