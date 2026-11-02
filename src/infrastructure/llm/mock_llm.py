from typing import List
from src.core.models import Citation, DocumentChunk, RAGResponse, SearchResult
from src.core.ports import ILLMProvider


class MockLLMProvider(ILLMProvider):
    """Adaptador de pruebas unitarias que simula respuestas fundamentadas y citaciones."""

    def generate_grounded_answer(
        self, query: str, context_chunks: List[DocumentChunk]
    ) -> RAGResponse:
        if not context_chunks:
            return RAGResponse(
                answer="No dispongo de información suficiente en los documentos provistos para responder esta consulta.",
                citations=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
            )

        citations = []
        answer_parts = [
            f"Basado en los documentos consultados sobre '{query}':"
        ]

        for idx, chunk in enumerate(context_chunks, start=1):
            answer_parts.append(
                f"- Según se indica en la fuente [^{idx}], {chunk.content[:90]}..."
            )
            citations.append(
                Citation(
                    index=idx,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    excerpt=chunk.content[:150],
                )
            )

        return RAGResponse(
            answer="\n".join(answer_parts),
            citations=citations,
            retrieved_chunks=[
                SearchResult(chunk=c, similarity_score=0.95 - (i * 0.05))
                for i, c in enumerate(context_chunks)
            ],
            has_sufficient_context=True,
        )
