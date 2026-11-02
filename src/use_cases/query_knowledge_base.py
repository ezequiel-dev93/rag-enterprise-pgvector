import time
from typing import List
from src.core.models import RAGResponse
from src.core.ports import IEmbeddingProvider, ILLMProvider, IVectorRepository


class QueryKnowledgeBaseUseCase:
    """Caso de uso para buscar por similitud vectorial y generar respuestas con citaciones exactas."""

    def __init__(
        self,
        embedder: IEmbeddingProvider,
        repository: IVectorRepository,
        llm: ILLMProvider,
        default_top_k: int = 4,
        default_min_similarity: float = 0.50,
    ):
        self.embedder = embedder
        self.repository = repository
        self.llm = llm
        self.default_top_k = default_top_k
        self.default_min_similarity = default_min_similarity

    def execute(
        self,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        filter_filenames: List[str] | None = None,
    ) -> RAGResponse:
        start_time = time.perf_counter()
        clean_query = query.strip()
        if not clean_query:
            return RAGResponse(
                answer="Por favor ingresa una pregunta válida.",
                citations=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
                latency_seconds=0.0,
            )

        k = top_k or self.default_top_k
        min_sim = min_similarity or self.default_min_similarity

        # 1. Generar embedding de la consulta del usuario
        query_vector = self.embedder.generate_embedding(clean_query)

        # 2. Búsqueda vectorial en el repositorio (con filtro opcional)
        search_results = self.repository.search_similar_chunks(
            query_vector=query_vector,
            top_k=k,
            min_similarity=min_sim,
            filter_filenames=filter_filenames,
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        if not search_results:
            return RAGResponse(
                answer="No se encontró información relevante en los documentos indexados para responder esta pregunta.",
                citations=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
                latency_seconds=elapsed,
            )

        # 3. Extraer fragmentos para alimentar el prompt
        chunks_for_context = [r.chunk for r in search_results]

        # 4. Generación fundamentada con citas
        rag_response = self.llm.generate_grounded_answer(
            query=clean_query,
            context_chunks=chunks_for_context,
        )

        total_latency = round(time.perf_counter() - start_time, 2)
        top_sim = round(search_results[0].similarity_score * 100, 1) if search_results else 0.0

        # Estimación de tokens (~4 caracteres por token)
        total_chars = (
            len(clean_query)
            + sum(len(c.content) for c in chunks_for_context)
            + len(rag_response.answer)
        )
        est_tokens = max(1, total_chars // 4)

        rag_response.retrieved_chunks = search_results
        rag_response.latency_seconds = total_latency
        rag_response.top_similarity = top_sim
        rag_response.estimated_tokens = est_tokens

        return rag_response
