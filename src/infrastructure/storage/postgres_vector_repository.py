import json
from typing import List

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        String,
        Text,
        create_engine,
        text,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker

    Base = declarative_base()
    HAS_SQLALCHEMY = True

    class DocumentORM(Base):
        __tablename__ = "rag_documents"

        id = Column(String(36), primary_key=True)
        filename = Column(String(255), nullable=False)
        content_hash = Column(String(64), nullable=False, index=True)
        total_pages = Column(Integer, default=1)
        total_chunks = Column(Integer, default=0)
        created_at = Column(DateTime, nullable=False)

    class ChunkORM(Base):
        __tablename__ = "rag_chunks"

        id = Column(String(36), primary_key=True)
        document_id = Column(
            String(36),
            ForeignKey("rag_documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
        document_name = Column(String(255), nullable=False)
        chunk_index = Column(Integer, nullable=False)
        page_number = Column(Integer, default=1)
        content = Column(Text, nullable=False)

except ImportError:
    HAS_SQLALCHEMY = False
    Base = None
    DocumentORM = None
    ChunkORM = None

from src.core.models import Document, DocumentChunk, SearchResult
from src.core.ports import IVectorRepository


class PostgresVectorRepository(IVectorRepository):
    """Adaptador de producción para PostgreSQL 16 con la extensión pgvector e índices HNSW."""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_storage(self) -> None:
        with self.engine.begin() as conn:
            # 1. Habilitar extensión vectorial
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            # 2. Crear tablas de documentos y fragmentos
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id VARCHAR(36) PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    total_pages INTEGER DEFAULT 1,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id VARCHAR(36) PRIMARY KEY,
                    document_id VARCHAR(36) REFERENCES rag_documents(id) ON DELETE CASCADE,
                    document_name VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER DEFAULT 1,
                    content TEXT NOT NULL,
                    embedding vector(768)
                );
                """
                )
            )

            # 3. Crear índice HNSW para búsquedas en <10ms
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_hnsw 
                ON rag_chunks 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
                )
            )

    def save_document(self, document: Document) -> None:
        with self.SessionLocal() as session:
            doc_orm = DocumentORM(
                id=document.id,
                filename=document.filename,
                content_hash=document.content_hash,
                total_pages=document.total_pages,
                total_chunks=document.total_chunks,
                created_at=document.created_at,
            )
            session.merge(doc_orm)
            session.commit()

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        with self.engine.begin() as conn:
            for c in chunks:
                if not c.embedding:
                    continue
                emb_str = f"[{','.join(str(x) for x in c.embedding)}]"
                conn.execute(
                    text(
                        """
                        INSERT INTO rag_chunks (id, document_id, document_name, chunk_index, page_number, content, embedding)
                        VALUES (:id, :document_id, :document_name, :chunk_index, :page_number, :content, :embedding::vector)
                        ON CONFLICT (id) DO NOTHING;
                        """
                    ),
                    {
                        "id": c.id,
                        "document_id": c.document_id,
                        "document_name": c.document_name,
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number,
                        "content": c.content,
                        "embedding": emb_str,
                    },
                )

    def search_similar_chunks(
        self,
        query_vector: List[float],
        top_k: int = 4,
        min_similarity: float = 0.5,
        filter_filenames: List[str] | None = None,
    ) -> List[SearchResult]:
        emb_str = f"[{','.join(str(x) for x in query_vector)}]"
        results: List[SearchResult] = []

        where_clause = "WHERE 1 - (embedding <=> :query_vector::vector) >= :min_sim"
        params = {
            "query_vector": emb_str,
            "min_sim": min_similarity,
            "top_k": top_k,
        }

        if filter_filenames:
            where_clause += " AND document_name = ANY(:filter_filenames)"
            params["filter_filenames"] = filter_filenames

        with self.engine.connect() as conn:
            query = text(
                f"""
                SELECT id, document_id, document_name, chunk_index, page_number, content,
                       1 - (embedding <=> :query_vector::vector) AS similarity
                FROM rag_chunks
                {where_clause}
                ORDER BY embedding <=> :query_vector::vector
                LIMIT :top_k;
                """
            )
            rows = conn.execute(query, params).fetchall()

            for r in rows:
                chunk = DocumentChunk(
                    id=r[0],
                    document_id=r[1],
                    document_name=r[2],
                    chunk_index=r[3],
                    page_number=r[4],
                    content=r[5],
                )
                results.append(
                    SearchResult(chunk=chunk, similarity_score=round(float(r[6]), 4))
                )

        return results

    def list_documents(self) -> List[Document]:
        with self.SessionLocal() as session:
            docs = session.query(DocumentORM).order_by(DocumentORM.created_at.desc()).all()
            return [
                Document(
                    id=d.id,
                    filename=d.filename,
                    content_hash=d.content_hash,
                    total_pages=d.total_pages,
                    total_chunks=d.total_chunks,
                    created_at=d.created_at,
                )
                for d in docs
            ]

    def delete_document(self, document_id: str) -> bool:
        with self.SessionLocal() as session:
            doc = session.query(DocumentORM).filter_by(id=document_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False
