import os
import sys

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from src.core.ports import IEmbeddingProvider, ILLMProvider, IVectorRepository
from src.infrastructure.embeddings.gemini_embedding import GeminiEmbeddingProvider
from src.infrastructure.embeddings.mock_embedding import MockEmbeddingProvider
from src.infrastructure.llm.gemini_llm import GeminiLLMProvider
from src.infrastructure.llm.mock_llm import MockLLMProvider
from src.infrastructure.parsers.pdf_parser import UniversalDocumentParser
from src.infrastructure.parsers.semantic_chunker import SemanticChunker
from src.infrastructure.storage.memory_vector_repository import InMemoryVectorRepository
from src.infrastructure.storage.postgres_vector_repository import PostgresVectorRepository
from src.use_cases.ingest_document import IngestDocumentUseCase
from src.use_cases.query_knowledge_base import QueryKnowledgeBaseUseCase

# 1. Configuración de Página en Streamlit
st.set_page_config(
    page_title="RAG Enterprise | Búsqueda Fundamentada con Citaciones",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inicialización de Estado de Sesión (Singleton / Cache)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_repo_memory" not in st.session_state:
    st.session_state.vector_repo_memory = InMemoryVectorRepository()

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# 3. Barra Lateral (Configuración y Gestión de Documentos)
with st.sidebar:
    st.title("⚙️ Configuración RAG")
    st.caption("Arquitectura Hexagonal + PostgreSQL pgvector")

    # Selector de Motor de Ejecución
    engine_mode = st.radio(
        "Modo de Operación:",
        options=["Producción (Gemini + PostgreSQL)", "Demo Local / Mock ($0 Costo)"],
        index=0 if os.getenv("GEMINI_API_KEY") else 1,
    )

    is_production = "Producción" in engine_mode

    # Credenciales y Conexión
    api_key = os.getenv("GEMINI_API_KEY", "")
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgrespassword@localhost:5432/rag_knowledge_base",
    )

    if is_production:
        api_key = st.text_input(
            "Gemini API Key:",
            value=api_key,
            type="password",
            help="Obtenla gratis en https://aistudio.google.com/",
        )
        db_url = st.text_input(
            "PostgreSQL URL (Docker):",
            value=db_url,
            type="password",
            help="Conexión a postgres con pgvector",
        )

    st.divider()

    # Inyección de Dependencias según el Modo
    parser = UniversalDocumentParser()
    chunker = SemanticChunker(chunk_size=700, chunk_overlap=100)

    try:
        if is_production:
            if not api_key:
                st.warning("⚠️ Ingresa tu Gemini API Key para Producción, o usa el Modo Demo abajo.")
                embedder: IEmbeddingProvider = MockEmbeddingProvider()
                llm: ILLMProvider = MockLLMProvider()
                repo: IVectorRepository = st.session_state.vector_repo_memory
            else:
                try:
                    embedder = GeminiEmbeddingProvider(api_key=api_key)
                    llm = GeminiLLMProvider(api_key=api_key)
                except ImportError as ie:
                    st.warning(f"⚠️ {ie}. Usando modelos Mock temporales.")
                    embedder = MockEmbeddingProvider()
                    llm = MockLLMProvider()

                try:
                    repo = PostgresVectorRepository(database_url=db_url)
                    repo.init_storage()
                    st.success("🟢 Conectado a PostgreSQL + pgvector")
                except Exception:
                    st.info("ℹ️ PostgreSQL en Docker no detectado. Usando base vectorial en memoria.")
                    repo = st.session_state.vector_repo_memory
        else:
            embedder = MockEmbeddingProvider()
            llm = MockLLMProvider()
            repo = st.session_state.vector_repo_memory
            st.info("💡 Modo Demo Activo ($0 Costo / Sin dependencias externas)")
    except Exception as e:
        st.error(f"Error de inicialización: {e}")
        st.stop()

    ingest_use_case = IngestDocumentUseCase(
        parser=parser, chunker=chunker, embedder=embedder, repository=repo
    )
    query_use_case = QueryKnowledgeBaseUseCase(
        embedder=embedder, repository=repo, llm=llm
    )

    # Subida de Documentos
    st.subheader("📁 Ingesta de Documentos")
    uploaded_files = st.file_uploader(
        "Cargar archivos PDF, Markdown o TXT:",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Indexar Documentos", use_container_width=True):
            for file in uploaded_files:
                file_bytes = file.read()
                with st.status(f"Indexando `{file.name}`...", expanded=True) as status:
                    st.write("1. Extrayendo páginas...")
                    st.write("2. Generando fragmentos semánticos...")
                    st.write("3. Calculando embeddings vectoriales...")
                    doc, num_chunks = ingest_use_case.execute(file_bytes, file.name)
                    status.update(
                        label=f"✅ `{file.name}` indexado ({num_chunks} chunks)",
                        state="complete",
                    )
            st.rerun()

    # Lista de Documentos y Filtros
    st.divider()
    st.subheader("📚 Documentos Indexados")
    docs = repo.list_documents()
    available_doc_names = [d.filename for d in docs]

    selected_doc_filter = []
    if docs:
        selected_doc_filter = st.multiselect(
            "🎯 Filtrar Búsqueda por Documento:",
            options=available_doc_names,
            default=[],
            help="Deja vacío para buscar en toda la biblioteca.",
        )

        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 **{d.filename}** ({d.total_chunks} chunks)")
            if col2.button("🗑️", key=f"del_{d.id}", help="Eliminar documento"):
                repo.delete_document(d.id)
                st.rerun()
    else:
        st.caption("No hay documentos indexados todavía.")

    # Botón para Exportar Auditoría si hay mensajes
    if st.session_state.messages:
        st.divider()
        report_lines = ["# 📋 REPORTE DE AUDITORÍA RAG", ""]
        for m in st.session_state.messages:
            role_label = "👤 Usuario" if m["role"] == "user" else "🤖 Asistente"
            report_lines.append(f"### {role_label}:")
            report_lines.append(m["content"])
            if "citations" in m and m["citations"]:
                report_lines.append("\n**Citaciones Verificables:**")
                for c in m["citations"]:
                    report_lines.append(f"- [^{c.index}] `{c.document_name}` (Pág. {c.page_number}): \"{c.excerpt}\"")
            report_lines.append("\n---")
        
        st.download_button(
            "📥 Descargar Informe con Citas (MD)",
            data="\n".join(report_lines),
            file_name="reporte_auditoria_rag.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.caption("👨‍💻 Desarrollado por **Ezequiel** — AI Engineer")

# 4. Área Principal (Chat con Citaciones y Métricas)
st.title("📚 Sistema RAG con Citaciones Verificables")
st.markdown(
    """
    Asistente conversacional con **búsqueda semántica híbrida** y **cero alucinaciones**.
    Cada dato relevante incluye su fuente original verificable `[^1]` con página y fragmento de texto exacto.
    """
)

# 5. Preguntas Rápidas Sugeridas (Quick Starter Chips)
if not st.session_state.messages:
    st.info("💡 **Preguntas rápidas para probar con tus documentos:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    if col_q1.button("📄 ¿Puntos clave del documento?", use_container_width=True):
        st.session_state.quick_prompt = "¿Cuáles son los puntos clave y temas principales que se detallan en los documentos?"
    if col_q2.button("⏱️ ¿Plazos, fechas y requisitos?", use_container_width=True):
        st.session_state.quick_prompt = "¿Qué plazos, fechas límites o requisitos obligatorios se mencionan?"
    if col_q3.button("⚠️ ¿Restricciones o advertencias?", use_container_width=True):
        st.session_state.quick_prompt = "¿Qué advertencias, restricciones o excepciones importantes se especifican?"

# Renderizar Historial de Mensajes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Métricas en vivo (KPI Cards)
        if "metrics" in msg and msg["metrics"]:
            met = msg["metrics"]
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("⏱️ Latencia", f"{met['latency']}s")
            m_col2.metric("🎯 Similitud Máx", f"{met['top_similarity']}%")
            m_col3.metric("📑 Chunks", f"{met['chunks']}")
            m_col4.metric("💰 Costo Aprox", f"< ${met['cost']}")

        st.markdown(msg["content"])

        if "citations" in msg and msg["citations"]:
            with st.expander(f"📑 Fuentes y Citaciones Verificables ({len(msg['citations'])})"):
                for cit in msg["citations"]:
                    st.markdown(
                        f"**[^{cit.index}] Archivo:** `{cit.document_name}` | **Página:** `{cit.page_number}`"
                    )
                    st.caption(f"> \"{cit.excerpt}...\"")

# Determinar si hay un prompt ingresado (por chat_input o por quick chips)
user_input = st.chat_input("Haz una pregunta sobre los documentos indexados...")
prompt_to_process = None

if user_input:
    prompt_to_process = user_input
elif st.session_state.quick_prompt:
    prompt_to_process = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

# Procesamiento de la Consulta RAG
if prompt_to_process:
    # 1. Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    # 2. Generar respuesta con Caso de Uso
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base vectorial y analizando fuentes..."):
            filter_targets = selected_doc_filter if selected_doc_filter else None
            response = query_use_case.execute(
                query=prompt_to_process, filter_filenames=filter_targets
            )

            # Tarjetas de Telemetría en Vivo
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("⏱️ Latencia", f"{response.latency_seconds}s")
            kpi_col2.metric("🎯 Similitud Máx", f"{response.top_similarity}%")
            kpi_col3.metric("📑 Chunks", f"{len(response.retrieved_chunks)}")
            est_cost = round(response.estimated_tokens * 0.0000003, 5)
            kpi_col4.metric("💰 Costo Aprox", f"< ${max(0.0001, est_cost)}")

            st.markdown(response.answer)

            # Mostrar Citas Desplegables
            if response.citations:
                with st.expander(
                    f"📑 Fuentes y Citaciones Verificables ({len(response.citations)})"
                ):
                    for cit in response.citations:
                        st.markdown(
                            f"**[^{cit.index}] Archivo:** `{cit.document_name}` | **Página:** `{cit.page_number}`"
                        )
                        st.info(f"\"{cit.excerpt}\"")

        # Guardar en sesión para persistir las tarjetas
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "citations": response.citations,
                "metrics": {
                    "latency": response.latency_seconds,
                    "top_similarity": response.top_similarity,
                    "chunks": len(response.retrieved_chunks),
                    "cost": max(0.0001, est_cost),
                },
            }
        )
