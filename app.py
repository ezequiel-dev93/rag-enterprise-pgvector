import os
import sys

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from src.core.ports import IEmbeddingProvider, ILLMProvider, IVectorRepository
from src.infrastructure.embeddings.cloud_embedding import CloudEmbeddingProvider
from src.infrastructure.embeddings.mock_embedding import MockEmbeddingProvider
from src.infrastructure.llm.cloud_llm import CloudLLMProvider
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

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

SAMPLE_DEMO_DOCS = {
    "politica_corporativa_sla_2026.md": (
        "# Política Corporativa de Garantías, SLA y Seguridad Cloud (2026)\n\n"
        "## 1. Tiempos de Respuesta y SLA Crítico\n"
        "Todo incidente de severidad crítica (P1) en producción debe recibir respuesta de ingeniería en un plazo máximo "
        "de 15 minutos y resolución o mitigación en menos de 4 horas. La disponibilidad mensual garantizada es del 99.95%.\n\n"
        "## 2. Política de Devoluciones y Garantía de Equipos\n"
        "Los clientes corporativos cuentan con una ventana de 30 días calendario para solicitar cambios sin costo "
        "y una garantía oficial extendida de 3 años o 100,000 kilómetros en flota vehicular.\n\n"
        "## 3. Restricciones de Seguridad y Protección de Datos\n"
        "Queda estrictamente prohibido almacenar claves privadas o credenciales de producción en texto plano. "
        "Todos los documentos sensibles deben cifrarse bajo el estándar AES-256 y auditarse trimestralmente."
    ).encode("utf-8"),
    "catalogo_precios_y_servicios.csv": (
        "Servicio,Categoria,Precio_USD,Plazo_Entrega,Requisitos_Obligatorios\n"
        "Auditoría Cloud Enterprise,Seguridad,4500,10 días hábiles,Acceso IAM de solo lectura y arquitectura actual\n"
        "Implementación RAG Corporativo,Inteligencia Artificial,6800,15 días hábiles,Corpus documental en PDF/DOCX/CSV y API Key\n"
        "Cirugía Refractiva LASIK Bilateral,Salud Oftalmológica,1850,Programación 48h,Paquimetría corneal mayor a 500 micras y topografía normal\n"
    ).encode("utf-8"),
}

# 2. Inicialización de Estado de Sesión (Aislado por Visitante Web)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_repo_memory" not in st.session_state:
    st.session_state.vector_repo_memory = InMemoryVectorRepository()

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# 3. Barra Lateral (Configuración BYOK + BYOD Multi-Formato)
with st.sidebar:
    st.title("⚙️ Configuración RAG")
    st.caption("Arquitectura Hexagonal · BYOK & Multi-Formato")

    # Selector de Motor de Ejecución (Seguro para Web Pública)
    engine_mode = st.radio(
        "🧠 Motor de IA:",
        options=[
            "🔑 Usar mi API Key (BYOK — IA en Vivo)",
            "⚡ Modo Demo Instantáneo ($0 Costo — Offline)",
        ],
        index=0 if st.session_state.user_api_key else 1,
        help="En la versión web puedes ingresar tu propia API Key de LLM (no se guarda en ningún servidor) o probar el flujo completo en Modo Demo a $0 costo.",
    )

    use_live_llm = "BYOK" in engine_mode

    api_key = ""
    if use_live_llm:
        api_key = st.text_input(
            "🔑 Tu LLM API Key:",
            value=st.session_state.user_api_key,
            type="password",
            placeholder="Pega tu clave AIzaSy...",
            help="Obtenla gratis en https://aistudio.google.com/ — Se usa de forma efímera por petición y jamás se guarda en disco ni en logs.",
        ).strip()
        st.session_state.user_api_key = api_key
        if st.session_state.user_api_key:
            if st.button("🔒 Borrar mi API Key de la sesión", use_container_width=True):
                st.session_state.user_api_key = ""
                st.rerun()

    # Configuración Opcional de Base de Datos Externa (PostgreSQL + pgvector)
    use_postgres = False
    db_url = ""
    with st.expander("🐘 Base de Datos Vectorial (Avanzado)", expanded=False):
        st.caption(
            "Por defecto tus documentos se indexan en **Memoria Aislada de Sesión** (privado por pestaña). "
            "Activa PostgreSQL si tienes un contenedor `pgvector` o Supabase/Neon."
        )
        use_postgres = st.checkbox("Conectar a PostgreSQL + pgvector", value=False)
        if use_postgres:
            db_url = st.text_input(
                "DATABASE_URL:",
                value=os.getenv(
                    "DATABASE_URL",
                    "postgresql://postgres:postgrespassword@localhost:5432/rag_knowledge_base",
                ),
                type="password",
            )

    st.divider()

    # Inyección de Dependencias según el Modo
    parser = UniversalDocumentParser()
    chunker = SemanticChunker(chunk_size=700, chunk_overlap=100)

    try:
        if use_live_llm and api_key:
            try:
                embedder: IEmbeddingProvider = CloudEmbeddingProvider(api_key=api_key)
                llm: ILLMProvider = CloudLLMProvider(api_key=api_key)
                st.success("🟢 Motor Activo: Cloud LLM + Embeddings Semánticos (BYOK)")
            except ImportError as ie:
                st.warning(f"⚠️ {ie}. Usando modelos Mock temporales.")
                embedder = MockEmbeddingProvider()
                llm = MockLLMProvider()
        else:
            if use_live_llm and not api_key:
                st.warning("⚠️ Pega tu LLM API Key arriba para usar IA real. Mientras tanto, opera en Modo Demo ($0).")
            else:
                st.info("💡 Modo Demo Activo ($0 Costo · Sin consumo de API)")
            embedder = MockEmbeddingProvider()
            llm = MockLLMProvider()

        current_embedder_type = type(embedder).__name__
        if st.session_state.get("last_embedder_type") and st.session_state.last_embedder_type != current_embedder_type:
            st.session_state.vector_repo_memory = InMemoryVectorRepository()
            st.toast("🔄 Motor vectorial cambiado: se reinició la memoria de sesión para mantener consistencia de embeddings.")
        st.session_state.last_embedder_type = current_embedder_type

        if use_postgres and db_url:
            try:
                repo: IVectorRepository = PostgresVectorRepository(database_url=db_url)
                repo.init_storage()
                st.success("🟢 Almacenamiento: PostgreSQL 16 + pgvector (HNSW)")
            except Exception:
                st.info("ℹ️ PostgreSQL no disponible. Usando bóveda vectorial en memoria de sesión.")
                repo = st.session_state.vector_repo_memory
        else:
            repo = st.session_state.vector_repo_memory
    except Exception as e:
        st.error(f"Error de inicialización: {e}")
        st.stop()

    ingest_use_case = IngestDocumentUseCase(
        parser=parser, chunker=chunker, embedder=embedder, repository=repo
    )
    query_use_case = QueryKnowledgeBaseUseCase(
        embedder=embedder, repository=repo, llm=llm
    )

    # Subida de Documentos (BYOD Multi-Formato)
    st.subheader("📁 Subir tus Documentos")
    uploaded_files = st.file_uploader(
        f"Formatos soportados: PDF, DOCX, CSV, JSON, MD, TXT, HTML (Máx {MAX_FILE_SIZE_MB}MB):",
        type=list(UniversalDocumentParser.SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Indexar Mis Documentos", type="primary", use_container_width=True):
            for file in uploaded_files:
                file_bytes = file.read()
                if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                    st.error(
                        f"❌ `{file.name}` supera el límite web de {MAX_FILE_SIZE_MB} MB."
                    )
                    continue
                with st.status(f"Indexando `{file.name}`...", expanded=True) as status:
                    st.write("1. Extrayendo páginas y secciones...")
                    st.write("2. Generando fragmentos semánticos...")
                    st.write("3. Calculando embeddings vectoriales...")
                    doc, num_chunks = ingest_use_case.execute(file_bytes, file.name)
                    status.update(
                        label=f"✅ `{file.name}` indexado ({num_chunks} chunks)",
                        state="complete",
                    )
            st.rerun()

    # Botones Rápidos: Cargar Pack Demo o Limpiar Sesión
    col_demo, col_clear = st.columns(2)
    if col_demo.button(
        "⚡ Cargar Demo",
        help="¿No tienes un archivo a mano? Carga 2 documentos de ejemplo (Política SLA + Catálogo CSV) para probar ahora mismo.",
        use_container_width=True,
    ):
        for fname, fbytes in SAMPLE_DEMO_DOCS.items():
            ingest_use_case.execute(fbytes, fname)
        st.rerun()

    if col_clear.button(
        "🧹 Limpiar Todo",
        help="Borra los documentos y el historial de chat de tu sesión actual.",
        use_container_width=True,
    ):
        st.session_state.vector_repo_memory = InMemoryVectorRepository()
        st.session_state.messages = []
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
