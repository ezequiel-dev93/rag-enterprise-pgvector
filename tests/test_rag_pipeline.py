import os
import sys
import unittest

# Asegurar que la raíz esté en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.models import DocumentChunk
from src.infrastructure.embeddings.mock_embedding import MockEmbeddingProvider
from src.infrastructure.llm.mock_llm import MockLLMProvider
from src.infrastructure.parsers.pdf_parser import UniversalDocumentParser
from src.infrastructure.parsers.semantic_chunker import SemanticChunker
from src.infrastructure.storage.memory_vector_repository import InMemoryVectorRepository
from src.use_cases.ingest_document import IngestDocumentUseCase
from src.use_cases.query_knowledge_base import QueryKnowledgeBaseUseCase


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.parser = UniversalDocumentParser()
        self.chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
        self.embedder = MockEmbeddingProvider(dimensions=768)
        self.repo = InMemoryVectorRepository()
        self.llm = MockLLMProvider()

        self.ingest_use_case = IngestDocumentUseCase(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            repository=self.repo,
        )

        self.query_use_case = QueryKnowledgeBaseUseCase(
            embedder=self.embedder,
            repository=self.repo,
            llm=self.llm,
            default_top_k=2,
            default_min_similarity=0.30,
        )

    def test_chunker_preserves_page_numbers(self):
        sample_pages = [
            (1, "Texto introductorio de la primera página del manual de mantenimiento."),
            (2, "Especificaciones técnicas del motor: Requiere aceite sintético 5W-30."),
        ]
        chunks = self.chunker.create_chunks(sample_pages, "doc_123", "manual.pdf")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[1].page_number, 2)

    def test_ingest_and_query_with_citations(self):
        sample_doc_content = (
            "Garantía oficial del vehículo: La cobertura es de 3 años o 100.000 kilómetros. "
            "El service oficial debe realizarse obligatoriamente cada 10.000 km."
        ).encode("utf-8")

        # 1. Ingestar
        doc, chunks_count = self.ingest_use_case.execute(
            sample_doc_content, "garantia_toyota.txt"
        )
        self.assertIsNotNone(doc.id)
        self.assertGreater(chunks_count, 0)

        # 2. Consultar
        response = self.query_use_case.execute("garantía del vehículo")
        self.assertTrue(response.has_sufficient_context)
        self.assertGreater(len(response.citations), 0)
        self.assertIn("garantia_toyota.txt", response.citations[0].document_name)

    def test_deduplication_prevents_duplicate_indexing(self):
        content = "Política de devoluciones dentro de los 30 días posteriores a la compra.".encode(
            "utf-8"
        )

        # Ingesta inicial
        doc1, count1 = self.ingest_use_case.execute(content, "politicas.txt")
        # Ingesta del mismo contenido
        doc2, count2 = self.ingest_use_case.execute(content, "politicas.txt")

        self.assertEqual(doc1.id, doc2.id)
        self.assertEqual(len(self.repo.list_documents()), 1)

    def test_empty_knowledge_base_handles_no_hallucination(self):
        empty_repo = InMemoryVectorRepository()
        query_uc = QueryKnowledgeBaseUseCase(
            embedder=self.embedder,
            repository=empty_repo,
            llm=self.llm,
        )

        response = query_uc.execute("¿Cuál es el horario de atención?")
        self.assertFalse(response.has_sufficient_context)
        self.assertEqual(len(response.citations), 0)
        self.assertIn("No se encontró información", response.answer)


if __name__ == "__main__":
    unittest.main()
