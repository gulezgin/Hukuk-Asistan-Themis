"""
build_vector_db.py — Tek PDF için vektör veritabanı üretir (CLI).

Varsayılan giriş: data/AG_Application_Development_Contract.pdf
Çıktılar:
  - data/AG_Application_Development_Contract_index.faiss
  - data/AG_Application_Development_Contract_chunks.pkl
  - data/vector_db_meta.json

Çoklu PDF ve case yönetimi için app.py (Streamlit) kullanın.
"""

import os
import json
from pathlib import Path

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from utils import extract_text_from_pdf, chunk_text

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("data")
META_PATH = DATA_DIR / "vector_db_meta.json"


def _hint_venv_if_needed(err: Exception) -> None:
    """Import hatasında kullanıcıya venv Python ile çalıştırmasını hatırlatır."""
    msg = str(err)
    if "No module named" in msg and (ROOT / "venv" / "Scripts" / "python.exe").exists():
        raise SystemExit(
            f"{msg}\n\n"
            "Bu proje `venv` içinde paketleriyle hazır görünüyor. Şöyle çalıştır:\n"
            r"  .\venv\Scripts\python.exe .\build_vector_db.py"
        )
    raise err


def main():
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    FAISS_METRIC = os.getenv("FAISS_METRIC", "cosine").strip().lower()

    print(f"📦 Embedding model: {EMBEDDING_MODEL}")
    print(f"📐 FAISS metric: {FAISS_METRIC}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    pdf_path = DATA_DIR / "AG_Application_Development_Contract.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"❌ PDF bulunamadı: {pdf_path}")

    try:
        text = extract_text_from_pdf(str(pdf_path))
    except Exception as e:
        _hint_venv_if_needed(e)

    chunks = chunk_text(text, chunk_size=500)
    print(f"📄 {len(chunks)} chunk oluşturuldu.")

    normalize = FAISS_METRIC == "cosine"
    embeddings = model.encode(chunks, normalize_embeddings=normalize)
    print(f"🔢 Embedding shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    if FAISS_METRIC == "cosine":
        index = faiss.IndexFlatIP(dimension)
    elif FAISS_METRIC == "l2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise SystemExit("❌ FAISS_METRIC geçersiz. 'cosine' veya 'l2' olmalı.")

    index.add(np.array(embeddings))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DATA_DIR / "AG_Application_Development_Contract_index.faiss"))
    with open(DATA_DIR / "AG_Application_Development_Contract_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": EMBEDDING_MODEL,
                "faiss_metric": FAISS_METRIC,
                "chunk_size": 500,
                "pdf_path": str(pdf_path).replace("\\", "/"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("✅ FAISS index ve chunklar başarıyla kaydedildi.")


if __name__ == "__main__":
    main()
