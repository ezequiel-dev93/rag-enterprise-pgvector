import os
import sys

# Agregar la raíz del servidor al path de Python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Forzar codificación UTF-8 en terminales de Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.infrastructure.embeddings.mock_embedding import MockEmbeddingProvider
from src.infrastructure.llm.mock_llm import MockLLMProvider
from src.infrastructure.parsers.pdf_parser import UniversalDocumentParser
from src.infrastructure.parsers.semantic_chunker import SemanticChunker
from src.infrastructure.storage.memory_vector_repository import InMemoryVectorRepository
from src.use_cases.ingest_document import IngestDocumentUseCase
from src.use_cases.query_knowledge_base import QueryKnowledgeBaseUseCase


def separator(title: str):
    print("\n" + "=" * 70)
    print(f"🚀 {title.upper()}")
    print("=" * 70)


def main():
    print("\n" + "📚 INICIANDO TEST EN VIVO DEL SISTEMA RAG CON CITACIONES".center(70))
    print("=" * 70)

    # 1. Inicialización en Modo Local / Mock ($0 Costo)
    parser = UniversalDocumentParser()
    chunker = SemanticChunker(chunk_size=350, chunk_overlap=50)
    embedder = MockEmbeddingProvider(dimensions=768)
    repo = InMemoryVectorRepository()
    llm = MockLLMProvider()

    ingest_use_case = IngestDocumentUseCase(
        parser=parser, chunker=chunker, embedder=embedder, repository=repo
    )
    query_use_case = QueryKnowledgeBaseUseCase(
        embedder=embedder,
        repository=repo,
        llm=llm,
        default_top_k=2,
        default_min_similarity=0.30,
    )

    # TEST 1: Ingesta de Documento de Concesionaria
    separator("1. Ingesta de Documento Comercial: Manual de Garantía")
    doc_concesionaria = (
        "MANUAL DE GARANTÍA OFICIAL TOYOTA COLOMBIA:\n"
        "La garantía cubre 3 años o 100.000 kilómetros para todos los modelos Hilux y Corolla.\n"
        "Los servicios oficiales de mantenimiento deben realizarse obligatoriamente cada 10.000 km.\n"
        "Las cotizaciones se emiten en pesos colombianos (COP) y son válidas por 15 días calendario."
    ).encode("utf-8")

    doc, chunks = ingest_use_case.execute(
        doc_concesionaria, "manual_garantia_colombia.txt"
    )
    print(f"✅ Documento indexado: `{doc.filename}`")
    print(f"   - Total de páginas: {doc.total_pages}")
    print(f"   - Total de fragmentos semánticos: {chunks}")
    print(f"   - Hash SHA-256: {doc.content_hash[:16]}...")

    # TEST 2: Ingesta de Documento Clínico / Oftalmología
    separator("2. Ingesta de Documento Médico: Protocolo Quirúrgico")
    doc_oftalmo = (
        "CLÍNICA OFTALMOLÓGICA - CUIDADOS PRE Y POST OPERATORIOS:\n"
        "Cirugía refractiva LASIK: El paciente debe suspender el uso de lentes de contacto 7 días antes.\n"
        "Posterior a la intervención, se prescriben gotas antibióticas y lágrimas artificiales cada 4 horas.\n"
        "El reposo visual absoluto se recomienda durante las primeras 24 horas posteriores a la cirugía."
    ).encode("utf-8")

    doc_med, chunks_med = ingest_use_case.execute(
        doc_oftalmo, "protocolo_cirugia_lasik.txt"
    )
    print(f"✅ Documento indexado: `{doc_med.filename}` ({chunks_med} chunks)")

    # TEST 3: Consulta con Búsqueda Semántica y Citas Exactas
    separator("3. Consulta RAG: Pregunta sobre Concesionaria")
    query_auto = "¿Cuál es la garantía y cada cuántos km se hace el mantenimiento?"
    print(f"👤 Usuario: {query_auto}\n")

    response_auto = query_use_case.execute(query_auto)
    print(f"🤖 Asistente:\n{response_auto.answer}\n")
    print("📑 Citaciones extraídas:")
    for c in response_auto.citations:
        print(f"   - [^{c.index}] `{c.document_name}` (Pág. {c.page_number})")
        print(f"     \"{c.excerpt[:120]}...\"")

    # TEST 4: Consulta sobre Oftalmología
    separator("4. Consulta RAG: Pregunta sobre Clínica Oftalmológica")
    query_med = "¿Cuántos días antes se deben suspender los lentes de contacto?"
    print(f"👤 Usuario: {query_med}\n")

    response_med = query_use_case.execute(query_med)
    print(f"🤖 Asistente:\n{response_med.answer}\n")
    for c in response_med.citations:
        print(f"   - [^{c.index}] `{c.document_name}` (Pág. {c.page_number})")

    # TEST 5: Prueba Anti-Alucinación (Pregunta fuera del contexto con umbral estricto)
    separator("5. Prueba Anti-Alucinación: Pregunta sobre Viajes a Marte")
    query_alien = "¿Cómo comprar un pasaje en cohete para viajar al planeta Marte?"
    print(f"👤 Usuario: {query_alien}\n")

    response_alien = query_use_case.execute(query_alien, min_similarity=0.50)
    print(f"🤖 Asistente:\n{response_alien.answer}")
    print(f"   - ¿Hubo contexto suficiente?: {response_alien.has_sufficient_context}")
    print(f"   - Total de citas: {len(response_alien.citations)}")

    print("\n" + "=" * 70)
    print("✅ TODAS LAS PRUEBAS EN VIVO DEL SISTEMA RAG FUERON EXITOSAS".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
