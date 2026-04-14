"""
ask_question.py — Komut satırı RAG soru-cevap (tek vektör koleksiyonu).

Önkoşul: build_vector_db.py ile üretilmiş dosyalar.
Çoklu PDF ve case yönetimi için app.py (Streamlit) kullanın.
"""

import os
import pickle
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

from utils import is_placeholder_key
from prompts import SYSTEM_PROMPT

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "AG_Application_Development_Contract_index.faiss"
CHUNKS_PATH = DATA_DIR / "AG_Application_Development_Contract_chunks.pkl"
META_PATH = DATA_DIR / "vector_db_meta.json"


def _load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _hint_venv_if_needed(err: Exception) -> None:
    msg = str(err)
    if "No module named" in msg and (ROOT / "venv" / "Scripts" / "python.exe").exists():
        raise SystemExit(
            f"{msg}\n\n"
            "Bu proje `venv` içinde paketleriyle hazır görünüyor. Şöyle çalıştır:\n"
            r"  .\venv\Scripts\python.exe .\ask_question.py"
        )
    raise err


def main():
    META = _load_meta()

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", META.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    FAISS_METRIC = os.getenv("FAISS_METRIC", META.get("faiss_metric", "cosine")).strip().lower()

    print(f"📦 Model: {EMBEDDING_MODEL} | Metric: {FAISS_METRIC}")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    normalize = FAISS_METRIC == "cosine"

    # LLM Client
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

    if LLM_PROVIDER == "openai":
        llm_api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif LLM_PROVIDER == "groq":
        llm_api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if "LLM_MODEL" not in os.environ:
            LLM_MODEL = "llama-3.1-8b-instant"
    else:
        llm_api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

    client = None
    if llm_api_key and not is_placeholder_key(llm_api_key):
        client = OpenAI(api_key=llm_api_key, base_url=base_url) if base_url else OpenAI(api_key=llm_api_key)
        print(f"🤖 LLM: {LLM_PROVIDER} / {LLM_MODEL}")
    else:
        print("⚠️  LLM anahtarı yok — sadece retrieval modu.")

    # FAISS & Chunks yükle
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        if not INDEX_PATH.exists():
            raise SystemExit(
                f"❌ FAISS index bulunamadı: {INDEX_PATH}\n"
                "Önce vektör veritabanını oluştur:\n"
                r"  .\venv\Scripts\python.exe .\build_vector_db.py"
            )
        _hint_venv_if_needed(e)

    try:
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
    except Exception as e:
        if not CHUNKS_PATH.exists():
            raise SystemExit(
                f"❌ Chunk dosyası bulunamadı: {CHUNKS_PATH}\n"
                "Önce vektör veritabanını oluştur:\n"
                r"  .\venv\Scripts\python.exe .\build_vector_db.py"
            )
        _hint_venv_if_needed(e)

    print(f"\n✅ Hazır. {index.ntotal} vektör yüklendi. Çıkmak için 'exit' yazın.\n")

    while True:
        query = input("⚖️  Sorunuz: ")

        if query.strip().lower() == "exit":
            print("👋 Çıkılıyor...")
            break
        if not query.strip():
            continue

        query_embedding = embedder.encode([query], normalize_embeddings=normalize)

        k = min(5, index.ntotal)  # k'yı toplam vektör sayısıyla sınırla
        distances, indices = index.search(np.array(query_embedding), k)

        retrieved_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
        context = "\n ---- \n".join(
            ch if isinstance(ch, str) else ch.get("text", str(ch))
            for ch in retrieved_chunks
        )

        prompt = f"""{SYSTEM_PROMPT}

Aşağıdaki bağlama dayanarak soruyu yanıtla.

Bağlam:
{context}

Soru:
{query}

Yanıt:"""

        try:
            if client is None:
                print(
                    "\n⚠️  LLM bağlantısı yok. İlgili metin parçaları:\n"
                )
                print(context)
                continue

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            print(f"\n🤖 Yanıt:\n{response.choices[0].message.content.strip()}\n")
        except Exception as e:
            print(f"\n❌ Hata: {e}\n")


if __name__ == "__main__":
    main()