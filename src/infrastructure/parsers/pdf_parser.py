import csv
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from typing import List, Tuple
from src.core.ports import IDocumentParser

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class _HTMLTextExtractor(HTMLParser):
    """Extractor ligero de texto limpio desde archivos HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


class UniversalDocumentParser(IDocumentParser):
    """
    Adaptador universal para extraer texto paginado/seccionado de:
    PDF, DOCX (Word), CSV, JSON, HTML, Markdown y TXT.
    """

    SUPPORTED_EXTENSIONS = ("pdf", "docx", "csv", "json", "md", "txt", "html", "htm")

    def extract_pages(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        lower_name = filename.lower()

        if lower_name.endswith(".pdf"):
            if not HAS_PYPDF:
                raise ImportError(
                    "El paquete 'pypdf' es requerido para procesar archivos PDF.\n"
                    "Instálalo con: pip install pypdf"
                )
            return self._parse_pdf(file_bytes)
        elif lower_name.endswith(".docx"):
            return self._parse_docx(file_bytes)
        elif lower_name.endswith(".csv"):
            return self._parse_csv(file_bytes)
        elif lower_name.endswith(".json"):
            return self._parse_json(file_bytes)
        elif lower_name.endswith((".html", ".htm")):
            return self._parse_html(file_bytes)
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

    def _parse_docx(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        """Extrae párrafos de un archivo Word (.docx) nativamente desde su estructura XML."""
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if "word/document.xml" not in zf.namelist():
                    return [(1, "")]
                xml_content = zf.read("word/document.xml")
            root = ET.fromstring(xml_content)
            paragraphs: List[str] = []
            for elem in root.iter():
                if elem.tag.endswith("}p"):
                    texts = [
                        node.text
                        for node in elem.iter()
                        if node.tag.endswith("}t") and node.text
                    ]
                    if texts:
                        para_str = "".join(texts).strip()
                        if para_str:
                            paragraphs.append(para_str)

            # Agrupar cada ~15 párrafos como una "página" lógica para citación precisa
            pages: List[Tuple[int, str]] = []
            block_size = 15
            for i in range(0, len(paragraphs), block_size):
                page_num = (i // block_size) + 1
                block_text = " ".join(" ".join(paragraphs[i : i + block_size]).split())
                if block_text:
                    pages.append((page_num, block_text))
            return pages if pages else [(1, "")]
        except Exception:
            return self._parse_text(file_bytes)

    def _parse_csv(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        """Convierte filas CSV en registros semánticos 'Columna: Valor' agrupados por bloque."""
        raw_text = self._decode_bytes(file_bytes)
        reader = csv.reader(io.StringIO(raw_text))
        rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not rows:
            return [(1, "")]

        headers = [h.strip() or f"col_{idx+1}" for idx, h in enumerate(rows[0])]
        data_rows = rows[1:] if len(rows) > 1 else rows

        formatted_rows: List[str] = []
        for row_idx, row in enumerate(data_rows, start=1):
            pairs = [
                f"{headers[i] if i < len(headers) else f'col_{i+1}'}: {val.strip()}"
                for i, val in enumerate(row)
                if val.strip()
            ]
            if pairs:
                formatted_rows.append(f"[Fila {row_idx}] " + " | ".join(pairs) + ".")

        # Agrupar cada 25 filas como página lógica
        pages: List[Tuple[int, str]] = []
        rows_per_page = 25
        for i in range(0, len(formatted_rows), rows_per_page):
            page_num = (i // rows_per_page) + 1
            pages.append((page_num, " ".join(formatted_rows[i : i + rows_per_page])))

        return pages if pages else [(1, "")]

    def _parse_json(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        """Aplana estructuras JSON en bloques legibles para embeddings."""
        raw_text = self._decode_bytes(file_bytes)
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return self._parse_text(file_bytes)

        if isinstance(data, list):
            pages: List[Tuple[int, str]] = []
            items_per_page = 10
            for i in range(0, len(data), items_per_page):
                page_num = (i // items_per_page) + 1
                chunk_str = json.dumps(data[i : i + items_per_page], ensure_ascii=False)
                pages.append((page_num, " ".join(chunk_str.split())))
            return pages if pages else [(1, "")]
        else:
            pretty = json.dumps(data, ensure_ascii=False)
            return [(1, " ".join(pretty.split()))]

    def _parse_html(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        raw_text = self._decode_bytes(file_bytes)
        extractor = _HTMLTextExtractor()
        extractor.feed(raw_text)
        cleaned = " ".join(extractor.get_text().split())
        return [(1, cleaned)]

    def _parse_text(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        text = self._decode_bytes(file_bytes)
        cleaned = " ".join(text.split())
        return [(1, cleaned)]

    @staticmethod
    def _decode_bytes(file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")
