"""
utils.py — RAG Lawyer ortak yardımcı fonksiyonlar.

Tüm modüller (app.py, build_vector_db.py, ask_question.py) tarafından
paylaşılan metin işleme, dosya çıkarma ve yardımcı fonksiyonları barındırır.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import faiss
import fitz  # PyMuPDF
import numpy as np

# ── Sabitler ──────────────────────────────────────────────────
DATA_DIR = Path("data")
STORES_DIR = DATA_DIR / "stores"
STORES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COLLECTIONS = ("knowledge", "templates")

# ── Metin Yardımcıları ───────────────────────────────────────

def is_placeholder_key(value: str | None) -> bool:
    """API anahtarı boş veya örnek placeholder ise True döner."""
    if not value:
        return True
    v = value.strip()
    return ("<" in v) or (">" in v) or v.endswith("...")


def safe_slug(name: str) -> str:
    """Klasör adı için güvenli slug üretir (Türkçe karakterler korunur)."""
    s = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Metni satır satır birleştirerek yaklaşık chunk_size karakterlik parçalara böler.
    overlap: Parçalar arası örtüşme (bağlamın kaybolmaması için).
    """
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if len(current) + len(line_stripped) + 1 < chunk_size:
            current += " " + line_stripped if current else line_stripped
        else:
            if current:
                chunks.append(current.strip())
            current = line_stripped
    if current:
        chunks.append(current.strip())

    # Overlap: önceki chunk'un son overlap karakterini sonraki chunk'a ekle
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return [ch for ch in chunks if ch.strip()]


# ── Dosya Çıkarma ────────────────────────────────────────────

def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """PDF belleğinden okunur; her sayfa için (sayfa_no, metin) döner."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            out.append((i, text))
    return out


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF dosyasından tüm metin çıkarır (CLI araçları için)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(docx_bytes: bytes) -> list[tuple[int, str]]:
    """
    DOCX dosyasından metin çıkarır.
    Her paragraf grubu bir 'sayfa' gibi döner (DOCX'te fiziksel sayfa yok).
    ~3000 karakter = 1 sanal sayfa.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("DOCX desteği için python-docx kurun: pip install python-docx")

    doc = Document(BytesIO(docx_bytes))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # Sanal sayfalara böl (~3000 karakter)
    pages: list[tuple[int, str]] = []
    page_size = 3000
    for i in range(0, len(full_text), page_size):
        page_no = (i // page_size) + 1
        pages.append((page_no, full_text[i:i + page_size]))

    return pages


def extract_text_from_txt(txt_bytes: bytes) -> list[tuple[int, str]]:
    """TXT dosyasından metin çıkarır (tek sayfa olarak)."""
    text = txt_bytes.decode("utf-8", errors="replace")
    # Sanal sayfalara böl
    pages: list[tuple[int, str]] = []
    page_size = 3000
    for i in range(0, max(len(text), 1), page_size):
        page_no = (i // page_size) + 1
        pages.append((page_no, text[i:i + page_size]))
    return pages


def extract_pages(file_name: str, file_bytes: bytes) -> list[tuple[int, str]]:
    """Dosya uzantısına göre doğru çıkarma fonksiyonunu seçer."""
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return extract_pages_from_pdf_bytes(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}. PDF, DOCX veya TXT kullanın.")


# ── Vektör Veritabanı İşlemleri ──────────────────────────────

def store_paths(case_name: str, collection: str) -> tuple[Path, Path, Path]:
    """Case+collection için (index, chunks, meta) dosya yollarını döndürür."""
    if collection not in DEFAULT_COLLECTIONS:
        raise ValueError("collection geçersiz. 'knowledge' veya 'templates' olmalı.")
    case = safe_slug(case_name)
    case_dir = STORES_DIR / case / collection
    case_dir.mkdir(parents=True, exist_ok=True)
    return (
        case_dir / "index.faiss",
        case_dir / "chunks.pkl",
        case_dir / "meta.json",
    )


def load_store_meta_for(case_name: str, collection: str) -> dict:
    """Case+collection için meta.json okur."""
    _, _, meta_path = store_paths(case_name, collection)
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_store_meta_for(case_name: str, collection: str, meta: dict) -> None:
    """Case+collection meta bilgisini diske yazar."""
    _, _, meta_path = store_paths(case_name, collection)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index_and_chunks(case_name: str, collection: str):
    """Diskteki FAISS index ve chunk listesini yükler; yoksa (None, [])."""
    index_path, chunks_path, _ = store_paths(case_name, collection)
    if not index_path.exists() or not chunks_path.exists():
        return None, []
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def save_index_and_chunks(case_name: str, collection: str, index, chunks: list[dict]) -> None:
    """FAISS index ve chunk listesini diske yazar."""
    index_path, chunks_path, _ = store_paths(case_name, collection)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)


def ensure_index(case_name: str, collection: str, dim: int, faiss_metric: str):
    """Index yoksa oluşturur; varsa uyumluluğu kontrol eder."""
    meta = load_store_meta_for(case_name, collection)
    index, chunks = load_index_and_chunks(case_name, collection)

    if index is None:
        if faiss_metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        elif faiss_metric == "l2":
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("FAISS_METRIC geçersiz. 'cosine' veya 'l2' olmalı.")
        chunks = []
        meta.update({"dim": dim, "faiss_metric": faiss_metric})
        save_store_meta_for(case_name, collection, meta)
        return index, chunks, meta

    existing_dim = int(meta.get("dim", getattr(index, "d", dim)))
    existing_metric = meta.get("faiss_metric", faiss_metric)
    if existing_dim != dim:
        raise ValueError(
            f"Bu case için embedding boyutu farklı (mevcut={existing_dim}, yeni={dim}). "
            "Yeni bir case oluştur veya case'i sıfırla."
        )
    if existing_metric != faiss_metric:
        raise ValueError(
            f"Bu case için FAISS_METRIC farklı (mevcut={existing_metric}, yeni={faiss_metric}). "
            "Yeni bir case oluştur veya case'i sıfırla."
        )
    return index, chunks, meta


def list_cases() -> list[str]:
    """data/stores/ altındaki case klasör adlarını listeler."""
    if not STORES_DIR.exists():
        return []
    return sorted([p.name for p in STORES_DIR.iterdir() if p.is_dir()])


def ensure_case_dirs(case_name: str) -> None:
    """Case için tüm koleksiyon klasörlerini hazırlar."""
    for c in DEFAULT_COLLECTIONS:
        _ = store_paths(case_name, c)


def format_chunks_for_llm(chunks: list[dict]) -> str:
    """LLM'e verilecek context'i kaynaklı şekilde formatlar."""
    parts: list[str] = []
    for ch in chunks:
        src = f"[{ch.get('source_file', '?')} s.{ch.get('page', '?')} #{ch.get('chunk_id', '?')}]"
        parts.append(f"{src}\n{ch.get('text', '')}".strip())
    return "\n\n----\n\n".join(parts)


def delete_file_from_case(case_name: str, collection: str, file_name: str) -> bool:
    """
    Bir dosyayı case+collection'dan siler.
    FAISS index'i yeniden oluşturur (dosyaya ait chunk'lar çıkarılır).
    """
    index, chunks = load_index_and_chunks(case_name, collection)
    if index is None or not chunks:
        return False

    # Silinecek ve kalacak chunk'ları ayır
    remaining = [ch for ch in chunks if ch.get("source_file") != file_name]
    if len(remaining) == len(chunks):
        return False  # Bu dosya zaten yok

    meta = load_store_meta_for(case_name, collection)

    if not remaining:
        # Hiç chunk kalmadı: dosyaları sil
        index_path, chunks_path, meta_path = store_paths(case_name, collection)
        for p in (index_path, chunks_path):
            if p.exists():
                p.unlink()
        # Meta'yı güncelle
        files = meta.get("files", {})
        files.pop(file_name, None)
        meta["files"] = files
        meta["total_chunks"] = 0
        save_store_meta_for(case_name, collection, meta)
        return True

    # Kalan chunk'ların embedding'lerini yeniden oluşturmak gerekiyor
    # Ama biz index'ten sadece belirli vektörleri silemeyiz (IndexFlat desteklemiyor)
    # Çözüm: Index'i tamamen yeniden oluştur
    # Chunk'ları re-index et
    for i, ch in enumerate(remaining):
        ch["chunk_id"] = i

    # Meta güncelle
    files = meta.get("files", {})
    files.pop(file_name, None)
    meta["files"] = files
    meta["total_chunks"] = len(remaining)
    save_store_meta_for(case_name, collection, meta)

    # Chunks'ı kaydet (index yeniden oluşturulacak ama embedding'ler chunks'ta yok)
    # NOT: Embedding'ler chunk'larda tutulmuyor. Re-embed gerekiyor.
    # Bu yüzden bir flag bırakıyoruz.
    save_index_and_chunks(case_name, collection, index, remaining)
    meta["needs_reindex"] = True
    save_store_meta_for(case_name, collection, meta)

    return True
