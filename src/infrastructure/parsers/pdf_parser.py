import io
from typing import List, Tuple
from src.core.ports import IDocumentParser

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class UniversalDocumentParser(IDocumentParser):
    """Adaptador para extraer texto con número de página de PDFs, Markdown y TXT."""

    def extract_pages(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        lower_name = filename.lower()

        if lower_name.endswith(".pdf"):
            if not HAS_PYPDF:
                raise ImportError(
                    "El paquete 'pypdf' es requerido para procesar archivos PDF.\n"
                    "Instálalo con: pip install pypdf"
                )
            return self._parse_pdf(file_bytes)
        elif lower_name.endswith((".md", ".txt", ".csv")):
            return self._parse_text(file_bytes)
        else:
            return self._parse_text(file_bytes)

    def _parse_pdf(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        results: List[Tuple[int, str]] = []
        reader = PdfReader(io.BytesIO(file_bytes))

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned = " ".join(page_text.split())
            if cleaned:
                results.append((idx + 1, cleaned))

        return results if results else [(1, "")]

    def _parse_text(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")

        cleaned = " ".join(text.split())
        return [(1, cleaned)]
