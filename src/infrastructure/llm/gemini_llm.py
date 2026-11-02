import os
import re
from typing import List

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.core.models import Citation, DocumentChunk, RAGResponse, SearchResult
from src.core.ports import ILLMProvider


class GeminiLLMProvider(ILLMProvider):
    """Adaptador oficial de Google Gemini (2.5 Flash / 1.5 Flash) con fundamentación estricta y citaciones."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ):
        if not HAS_GENAI:
            raise ImportError(
                "El paquete 'google-generativeai' no está instalado.\n"
                "Instálalo con: pip install google-generativeai"
            )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name

    def generate_grounded_answer(
        self, query: str, context_chunks: List[DocumentChunk]
    ) -> RAGResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada.")

        if not context_chunks:
            return RAGResponse(
                answer="No se encontraron fragmentos relevantes en la base de conocimiento para responder esta consulta.",
                citations=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
            )

        # 1. Construir el contexto numerado
        context_blocks = []
        citations_map = {}

        for idx, chunk in enumerate(context_chunks, start=1):
            context_blocks.append(
                f"--- FUENTE [^{idx}] ---\n"
                f"Documento: {chunk.document_name} | Página: {chunk.page_number}\n"
                f"Contenido: {chunk.content}\n"
            )
            citations_map[idx] = Citation(
                index=idx,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                excerpt=chunk.content[:200],
            )

        context_str = "\n".join(context_blocks)

        system_instruction = (
            "Eres un asistente de Inteligencia Artificial para búsqueda y extracción de información técnica y empresarial. "
            "Tu objetivo es responder con absoluta precisión y fundamentación.\n\n"
            "REGLAS OBLIGATORIAS:\n"
            "1. Basar tu respuesta ÚNICAMENTE en el CONTEXTO provisto. Nunca inventes datos ni utilices conocimientos previos no corroborados.\n"
            "2. Si la información solicitada no está en el CONTEXTO, responde de forma educada y directa:\n"
            "   'No dispongo de información suficiente en los documentos provistos para responder esta consulta.'\n"
            "3. Cada dato relevante debe incluir su cita explícita en formato [^1], [^2], etc., correspondiente a la fuente de donde lo extrajiste.\n"
            "4. Sé claro, profesional y estructurado en tu respuesta."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"=== CONTEXTO DISPONIBLE ===\n"
            f"{context_str}\n\n"
            f"=== PREGUNTA DEL USUARIO ===\n"
            f"{query}\n\n"
            f"=== RESPUESTA FUNDAMENTADA ==="
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
        except Exception:
            # Fallback a gemini-1.5-flash si 2.5 no estuviera disponible en la cuenta
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

        # 2. Extraer qué citas fueron realmente referenciadas en el texto
        used_indices = set(re.findall(r"\[\^(\d+)\]", raw_text))
        active_citations = [
            citations_map[int(idx)]
            for idx in sorted(int(i) for i in used_indices)
            if int(idx) in citations_map
        ]

        # Si no usó citas específicas pero respondió con éxito, adjuntar las fuentes consultadas
        if not active_citations and "no dispongo de información" not in raw_text.lower():
            active_citations = list(citations_map.values())

        has_sufficient = "no dispongo de información" not in raw_text.lower()

        return RAGResponse(
            answer=raw_text,
            citations=active_citations,
            retrieved_chunks=[
                SearchResult(chunk=c, similarity_score=1.0) for c in context_chunks
            ],
            has_sufficient_context=has_sufficient,
        )
