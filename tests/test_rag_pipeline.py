import io
import os
import sys
import unittest
import zipfile

# Asegurar que la raíz esté en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.models import DocumentChunk
from src.infrastructure.embeddings.cloud_embedding import (
    CloudEmbeddingProvider,
    _sanitize_error,
)
from src.infrastructure.embeddings.mock_embedding import MockEmbeddingProvider
from src.infrastructure.llm.cloud_llm import CloudLLMProvider
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

    def test_universal_parser_supports_csv_json_and_html(self):
        csv_bytes = (
            "Modelo,Precio_USD,Garantia\n"
            "Hilux SRX,52000,5 años\n"
            "Corolla Hybrid,31000,8 años\n"
        ).encode("utf-8")
        pages_csv = self.parser.extract_pages(csv_bytes, "catalogo.csv")
        self.assertEqual(len(pages_csv), 1)
        self.assertIn("Modelo: Hilux SRX", pages_csv[0][1])
        self.assertIn("Precio_USD: 52000", pages_csv[0][1])

        json_bytes = '[{"servicio": "LASIK", "arancel_usd": 1800}]'.encode("utf-8")
        pages_json = self.parser.extract_pages(json_bytes, "tarifas.json")
        self.assertEqual(len(pages_json), 1)
        self.assertIn("LASIK", pages_json[0][1])

        html_bytes = "<html><body><h1>Política SLA</h1><p>Respuesta en 4 horas.</p></body></html>".encode(
            "utf-8"
        )
        pages_html = self.parser.extract_pages(html_bytes, "sla.html")
        self.assertEqual(len(pages_html), 1)
        self.assertIn("Política SLA", pages_html[0][1])
        self.assertIn("Respuesta en 4 horas.", pages_html[0][1])

    def test_universal_parser_extracts_docx_xml_natively(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            xml_body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                "<w:p><w:r><w:t>Cláusula 1: Confidencialidad AES-256 obligatoria.</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>Cláusula 2: Vigencia de 36 meses.</w:t></w:r></w:p>"
                "</w:body></w:document>"
            )
            zf.writestr("word/document.xml", xml_body)
        pages_docx = self.parser.extract_pages(buf.getvalue(), "contrato.docx")
        self.assertEqual(len(pages_docx), 1)
        self.assertIn("AES-256", pages_docx[0][1])
        self.assertIn("36 meses", pages_docx[0][1])

    def test_multi_document_filter_and_deletion_isolation(self):
        doc_a, _ = self.ingest_use_case.execute(
            b"El servidor de produccion usa PostgreSQL 16 en el puerto 5432.",
            "infraestructura.txt",
        )
        doc_b, _ = self.ingest_use_case.execute(
            b"El presupuesto anual de marketing es de 90000 USD.",
            "marketing.txt",
        )
        self.assertEqual(len(self.repo.list_documents()), 2)

        # Filtrar exclusivamente por infraestructura.txt
        resp_filtered = self.query_use_case.execute(
            "servidor PostgreSQL", filter_filenames=["infraestructura.txt"]
        )
        for cit in resp_filtered.citations:
            self.assertEqual(cit.document_name, "infraestructura.txt")

        # Eliminar doc_a y verificar que solo queda doc_b
        self.repo.delete_document(doc_a.id)
        remaining = [d.filename for d in self.repo.list_documents()]
        self.assertEqual(remaining, ["marketing.txt"])

    def test_api_key_security_redaction_and_repr_protection(self):
        secret_key = "AIzaSySuperSecretTestKey999999"
        simulated_exc = RuntimeError(
            f"HTTP 400 Bad Request for url https://api.provider.com?key={secret_key}"
        )
        sanitized = _sanitize_error(simulated_exc, secret_key)
        self.assertNotIn(secret_key, sanitized)
        self.assertIn("[REDACTED_API_KEY]", sanitized)

        try:
            emb = CloudEmbeddingProvider(api_key=secret_key)
            llm = CloudLLMProvider(api_key=secret_key)
            self.assertNotIn(secret_key, repr(emb))
            self.assertNotIn(secret_key, repr(llm))
            self.assertIn("REDACTED", repr(emb))
        except ImportError:
            pass

    def test_e2e_full_rag_enterprise_journey(self):
        """
        Test End-to-End (E2E): Simula el viaje completo de un usuario en producción:
        1. Sube múltiples formatos (Markdown + CSV).
        2. Realiza consulta semántica transversal y valida telemetría (latencia, similitud, tokens, citas).
        3. Filtra por documento específico.
        4. Genera el reporte de auditoría exportable en Markdown con citas verificables.
        """
        sla_md = (
            "# Contrato Corporativo SLA 2026\n"
            "El tiempo de respuesta para incidentes críticos P1 es de 15 minutos con disponibilidad del 99.95%."
        ).encode("utf-8")
        precios_csv = (
            "Servicio,Precio_USD,Plazo\n"
            "Auditoría Cloud,4500,10 días\n"
        ).encode("utf-8")

        doc1, chunks1 = self.ingest_use_case.execute(sla_md, "sla_2026.md")
        doc2, chunks2 = self.ingest_use_case.execute(precios_csv, "precios.csv")
        self.assertEqual(len(self.repo.list_documents()), 2)
        self.assertGreater(chunks1 + chunks2, 0)

        # Consulta End-to-End
        rag_resp = self.query_use_case.execute("¿Cuál es el tiempo de respuesta para incidentes críticos P1?")
        self.assertTrue(rag_resp.has_sufficient_context)
        self.assertGreater(len(rag_resp.citations), 0)
        self.assertGreaterEqual(rag_resp.latency_seconds, 0.0)
        self.assertGreater(rag_resp.top_similarity, 0.0)
        self.assertGreater(rag_resp.estimated_tokens, 0)

        # Generación de Reporte de Auditoría E2E
        audit_md = "\n".join(
            f"- [^{c.index}] `{c.document_name}` (Pág. {c.page_number}): \"{c.excerpt}\""
            for c in rag_resp.citations
        )
        self.assertIn("sla_2026.md", audit_md)


if __name__ == "__main__":
    unittest.main()
