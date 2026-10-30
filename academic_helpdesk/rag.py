"""Small ChromaDB RAG pipeline with page-aware citations."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from . import config
from .ollama_client import OllamaClient


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + int(max_chars * 0.65), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _text_metadata(text: str, filename: str) -> tuple[str, str, str]:
    source_pdf = re.search(r"^SOURCE_PDF:\s*(.+)$", text, flags=re.MULTILINE)
    source_url = re.search(r"^SOURCE_URL:\s*(.+)$", text, flags=re.MULTILINE)
    date = re.search(r"^DOCUMENT_DATE:\s*(.+)$", text, flags=re.MULTILINE)
    return (
        source_pdf.group(1).strip() if source_pdf else filename,
        source_url.group(1).strip() if source_url else "",
        date.group(1).strip() if date else "",
    )


def load_source_chunks(documents_dir: Path, transcriptions_dir: Path) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    chunks: list[str] = []
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    files = sorted(list(documents_dir.rglob("*.pdf")) + list(documents_dir.rglob("*.txt")))
    files += sorted(transcriptions_dir.rglob("*.txt"))

    for path in files:
        if path.name.startswith("._"):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
                document = fitz.open(path)
            except Exception as exc:
                warnings.append(f"Could not open {path.name}: {exc}")
                continue
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if len(page_text) < 20:
                    warnings.append(f"No extractable text in {path.name}, page {page_index}; check whether the PDF is scanned.")
                    continue
                for chunk_index, chunk in enumerate(_split_text(page_text, config.CHUNK_CHARS, config.CHUNK_OVERLAP), start=1):
                    chunks.append(chunk)
                    metadata.append({
                        "source": path.name,
                        "page": page_index,
                        "chunk": chunk_index,
                        "source_type": "pdf",
                        "source_url": "https://www.iitj.ac.in/office-of-academics/en/academic-regulations" if "academic_regulations" in path.name else "",
                        "document_date": "",
                    })
            document.close()
            continue

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        source, source_url, document_date = _text_metadata(raw_text, path.name)
        # Manual transcriptions mark each original PDF page explicitly, preserving citations.
        page_matches = list(re.finditer(r"\[\[PAGE:\s*(\d+)\s*\]\]", raw_text))
        if page_matches:
            for position, match in enumerate(page_matches):
                content_start = match.end()
                content_end = page_matches[position + 1].start() if position + 1 < len(page_matches) else len(raw_text)
                page_text = raw_text[content_start:content_end].strip()
                page_num = int(match.group(1))
                for chunk_index, chunk in enumerate(_split_text(page_text, config.CHUNK_CHARS, config.CHUNK_OVERLAP), start=1):
                    chunks.append(chunk)
                    metadata.append({
                        "source": source,
                        "page": page_num,
                        "chunk": chunk_index,
                        "source_type": "manual_transcription",
                        "source_url": source_url,
                        "document_date": document_date,
                    })
        else:
            for chunk_index, chunk in enumerate(_split_text(raw_text, config.CHUNK_CHARS, config.CHUNK_OVERLAP), start=1):
                chunks.append(chunk)
                metadata.append({
                    "source": source,
                    "page": 0,
                    "chunk": chunk_index,
                    "source_type": "txt",
                    "source_url": source_url,
                    "document_date": document_date,
                })

    return chunks, metadata, warnings


class AcademicRAG:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self._collection = None

    def _get_collection(self):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed. Run `pip install -r requirements.txt`.") from exc
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_or_create_collection(
            name="iitj_academic_documents",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self) -> dict[str, Any]:
        chunks, metadata, warnings = load_source_chunks(config.DOCUMENTS_DIR, config.TRANSCRIPTIONS_DIR)
        if not chunks:
            raise RuntimeError("No readable document text was found. Check data/documents.")
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed. Run `pip install -r requirements.txt`.") from exc
        db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            db.delete_collection("iitj_academic_documents")
        except Exception:
            pass
        collection = db.create_collection(
            name="iitj_academic_documents",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 24
        for start in range(0, len(chunks), batch_size):
            batch_text = chunks[start:start + batch_size]
            batch_meta = metadata[start:start + batch_size]
            embeddings = self.client.embed(batch_text)
            ids = [
                hashlib.sha1((batch_meta[i]["source"] + str(batch_meta[i]["page"]) + str(start + i) + batch_text[i]).encode("utf-8")).hexdigest()
                for i in range(len(batch_text))
            ]
            collection.add(ids=ids, documents=batch_text, metadatas=batch_meta, embeddings=embeddings)
        self._collection = collection
        return {"files": len(list(config.DOCUMENTS_DIR.rglob("*.pdf"))) + len(list(config.DOCUMENTS_DIR.rglob("*.txt"))) + len(list(config.TRANSCRIPTIONS_DIR.rglob("*.txt"))), "chunks": len(chunks), "warnings": warnings}

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[dict[str, Any]]:
        collection = self._collection or self._get_collection()
        if collection.count() == 0:
            raise RuntimeError("The ChromaDB collection is empty. Run `python -m academic_helpdesk.ingest` first.")
        embedding = self.client.embed([query])[0]
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[dict[str, Any]] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for document, meta, distance in zip(docs, metas, distances):
            retrieved.append({"text": document, "metadata": meta or {}, "distance": float(distance)})
        return retrieved
