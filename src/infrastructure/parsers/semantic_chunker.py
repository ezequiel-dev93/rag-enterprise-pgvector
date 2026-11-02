import uuid
from typing import List, Tuple
from src.core.models import DocumentChunk


class SemanticChunker:
    """Divide el texto de las páginas en fragmentos con solapamiento y preserva metadatos."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(
        self,
        pages: List[Tuple[int, str]],
        document_id: str,
        document_name: str,
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        global_index = 0

        for page_num, text in pages:
            if not text.strip():
                continue

            words = text.split()
            current_start = 0

            # Si la página entera es menor al tamaño de chunk
            if len(text) <= self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        document_name=document_name,
                        chunk_index=global_index,
                        page_number=page_num,
                        content=text,
                    )
                )
                global_index += 1
                continue

            # Chunking por ventanas de palabras con overlap
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunk_text = text[i : i + self.chunk_size]
                if len(chunk_text.strip()) > 30:  # Ignorar fragmentos diminutos
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            document_id=document_id,
                            document_name=document_name,
                            chunk_index=global_index,
                            page_number=page_num,
                            content=chunk_text.strip(),
                        )
                    )
                    global_index += 1

        return chunks
