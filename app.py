"""
RAG Lawyer — Profesyonel Avukat Asistanı (Streamlit).

Gelişmiş özellikler:
  - Chat tabanlı soru-cevap arayüzü (konuşma hafızalı)
  - PDF / DOCX / TXT dosya desteği
  - Case (dosya grubu) yönetimi
  - Knowledge + Templates koleksiyonları
  - Dilekçe / sözleşme taslağı üretme
  - Hızlı soru şablonları
  - Dosya silme & export
"""

# ── Standart Kütüphaneler ─────────────────────────────────────
import os
import datetime
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# ── Üçüncü Parti ──────────────────────────────────────────────
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ── Proje Modülleri ────────────────────────────────────────────
from utils import (
    is_placeholder_key,
    safe_slug,
    chunk_text,
    extract_pages,
    store_paths,
    load_store_meta_for,
    save_store_meta_for,
    load_index_and_chunks,
    save_index_and_chunks,
    ensure_index,
    list_cases,
    ensure_case_dirs,
    format_chunks_for_llm,
    delete_file_from_case,
    DEFAULT_COLLECTIONS,
    STORES_DIR,
)
from prompts import (
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
    DRAFT_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    QUICK_QUESTIONS,
    DRAFT_TYPES,
)

load_dotenv()

# ╔══════════════════════════════════════════════════════════════╗
#  SAYFA KONFİGÜRASYONU
# ╚══════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Hukuk Asistanı AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════╗
#  CUSTOM CSS — Profesyonel Koyu Tema
# ╚══════════════════════════════════════════════════════════════╝

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0A0E1A;
    --bg-secondary: #111827;
    --bg-card: #1a2235;
    --bg-glass: rgba(26, 34, 53, 0.7);
    --accent-gold: #D4A843;
    --accent-gold-light: #E8C975;
    --accent-gold-dark: #B8912E;
    --text-primary: #E5E7EB;
    --text-secondary: #9CA3AF;
    --text-muted: #6B7280;
    --border-subtle: rgba(212, 168, 67, 0.15);
    --border-glow: rgba(212, 168, 67, 0.3);
    --shadow-gold: 0 0 20px rgba(212, 168, 67, 0.1);
    --gradient-gold: linear-gradient(135deg, #D4A843 0%, #E8C975 50%, #D4A843 100%);
    --gradient-dark: linear-gradient(180deg, #0A0E1A 0%, #111827 100%);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ── */
html, body, [class*="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: var(--gradient-dark) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1225 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-gold) !important;
    font-weight: 600 !important;
}

/* ── Header / Branding ── */
.brand-header {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(212,168,67,0.08) 0%, rgba(212,168,67,0.02) 100%);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-subtle);
}

.brand-header .logo-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    display: block;
    filter: drop-shadow(0 0 12px rgba(212,168,67,0.4));
}

.brand-header h1 {
    background: var(--gradient-gold);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.brand-header .subtitle {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 0.3rem;
    font-weight: 400;
}

/* ── Cards ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: var(--transition);
}

.info-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-gold);
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-gold);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Chat Messages ── */
.stChatMessage {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    margin-bottom: 0.5rem !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button:hover {
    border-color: var(--accent-gold) !important;
    box-shadow: var(--shadow-gold) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background: var(--gradient-gold) !important;
    color: #0A0E1A !important;
    font-weight: 600 !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    filter: brightness(1.1) !important;
    box-shadow: 0 4px 20px rgba(212,168,67,0.3) !important;
}

/* ── Quick Action Buttons ── */
.quick-btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    color: var(--text-secondary);
    font-size: 0.82rem;
    cursor: pointer;
    transition: var(--transition);
    margin: 0.2rem;
    text-decoration: none;
}

.quick-btn:hover {
    border-color: var(--accent-gold);
    color: var(--accent-gold);
    background: rgba(212,168,67,0.08);
}

/* ── Dividers ── */
hr {
    border-color: var(--border-subtle) !important;
    opacity: 0.5 !important;
}

/* ── File Uploader ── */
.stFileUploader {
    border-radius: var(--radius-md) !important;
}

.stFileUploader > div {
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border-subtle) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    border: 1px solid var(--border-subtle) !important;
    border-bottom: none !important;
}

.stTabs [aria-selected="true"] {
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
}

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.status-ready { background: rgba(16,185,129,0.15); color: #10B981; }
.status-empty { background: rgba(245,158,11,0.15); color: #F59E0B; }
.status-error { background: rgba(239,68,68,0.15); color: #EF4444; }

/* ── Source Citation ── */
.source-cite {
    background: rgba(212,168,67,0.08);
    border-left: 3px solid var(--accent-gold);
    padding: 0.6rem 1rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* ── Welcome Screen ── */
.welcome-container {
    text-align: center;
    padding: 3rem 1rem;
}

.welcome-container h2 {
    color: var(--accent-gold);
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.welcome-container p {
    color: var(--text-secondary);
    max-width: 500px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1rem;
    color: var(--text-muted);
    font-size: 0.72rem;
    border-top: 1px solid var(--border-subtle);
    margin-top: 2rem;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeIn 0.4s ease-out forwards;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.loading-shimmer {
    background: linear-gradient(90deg, var(--bg-card) 25%, rgba(212,168,67,0.1) 50%, var(--bg-card) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-gold-dark); }

/* ── Toast / Notifications ── */
.stAlert {
    border-radius: var(--radius-md) !important;
}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
#  YARDIMCI FONKSİYONLAR
# ╚══════════════════════════════════════════════════════════════╝

@st.cache_resource
def get_embedder(model_name: str) -> SentenceTransformer:
    """Embedding model'ini önbelleğe alarak yükler."""
    return SentenceTransformer(model_name)


def get_llm_client():
    """OpenAI uyumlu LLM client'ı döndürür."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LLM_MODEL", "gpt-4.1-nano")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if "LLM_MODEL" not in os.environ:
            model = "llama-3.1-8b-instant"
    else:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

    if api_key and not is_placeholder_key(api_key):
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return client, model, provider

    return None, model, provider


def add_file_to_case(
    case_name: str,
    collection: str,
    file_name: str,
    file_bytes: bytes,
    embedding_model: str,
    faiss_metric: str,
    chunk_size: int,
):
    """Dosyayı (PDF/DOCX/TXT) parçalara ayırıp vektörlerini case+collection index'ine ekler."""
    pages = extract_pages(file_name, file_bytes)

    embedder = get_embedder(embedding_model)
    normalize = faiss_metric == "cosine"

    all_chunks: list[dict] = []
    for page_no, page_text in pages:
        page_chunks = chunk_text(page_text, chunk_size=chunk_size)
        for ch in page_chunks:
            if ch.strip():
                all_chunks.append({
                    "text": ch,
                    "source_file": file_name,
                    "page": page_no,
                    "collection": collection,
                })

    if not all_chunks:
        raise ValueError("Dosyadan metin çıkarılamadı (boş dosya).")

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=normalize)
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = int(embeddings.shape[1])

    index, existing_chunks, meta = ensure_index(case_name, collection, dim=dim, faiss_metric=faiss_metric)

    base_id = len(existing_chunks)
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = base_id + i

    index.add(embeddings)
    merged = existing_chunks + all_chunks
    save_index_and_chunks(case_name, collection, index, merged)

    files = meta.get("files", {})
    files[file_name] = {
        "n_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "added_at": datetime.datetime.now().isoformat(),
    }
    meta.update({
        "embedding_model": embedding_model,
        "faiss_metric": faiss_metric,
        "chunk_size": chunk_size,
        "dim": dim,
        "files": files,
        "total_chunks": len(merged),
    })
    save_store_meta_for(case_name, collection, meta)

    return {"added_chunks": len(all_chunks), "dim": dim, "total_chunks": len(merged)}


def perform_search(case_name: str, collection: str, query: str, k: int, embedding_model: str, faiss_metric: str):
    """Vektör araması yapar, en yakın k parçayı döndürür."""
    index, chunks = load_index_and_chunks(case_name, collection)
    if index is None or not chunks:
        return []

    normalize = faiss_metric == "cosine"
    embedder = get_embedder(embedding_model)
    q_emb = embedder.encode([query], normalize_embeddings=normalize)

    # k'yı mevcut toplam chunk sayısıyla sınırla (out-of-range hatasını önle)
    actual_k = min(k, index.ntotal)
    if actual_k <= 0:
        return []

    distances, indices = index.search(np.asarray(q_emb, dtype="float32"), actual_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


# ╔══════════════════════════════════════════════════════════════╗
#  SESSION STATE İLK KURULUMU
# ╚══════════════════════════════════════════════════════════════╝

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "qa"


# ╔══════════════════════════════════════════════════════════════╗
#  SIDEBAR
# ╚══════════════════════════════════════════════════════════════╝

with st.sidebar:
    # ── Branding ──
    st.markdown("""
    <div class="brand-header">
        <span class="logo-icon">⚖️</span>
        <h1>Hukuk Asistanı Themis</h1>
        <div class="subtitle">Yapay Zeka Destekli Avukat Asistanı</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Case Seçimi ──
    st.markdown("### 📁 Dava / Dosya Grubu")
    cases = list_cases()
    selected_case = st.selectbox(
        "Case seçin",
        options=cases or ["default"],
        index=0,
        label_visibility="collapsed",
    )
    if not cases and selected_case == "default":
        ensure_case_dirs("default")
    else:
        ensure_case_dirs(selected_case)

    # Yeni case oluştur
    with st.expander("➕ Yeni dosya grubu oluştur"):
        new_case = st.text_input("Dosya grubu adı", value="", key="new_case_input")
        if st.button("Oluştur", disabled=not new_case.strip(), key="create_case_btn"):
            slug = safe_slug(new_case.strip())
            ensure_case_dirs(slug)
            st.success(f"✅ '{slug}' oluşturuldu! Sayfayı yenileyerek seçin.")
            st.rerun()

    st.divider()

    # ── Ayarlar ──
    st.markdown("### ⚙️ Ayarlar")

    meta_k = load_store_meta_for(selected_case, "knowledge")

    with st.expander("Gelişmiş Ayarlar", expanded=False):
        # ── Embedding Model Seçimi ──
        EMBEDDING_OPTIONS = {
            "BAAI/bge-m3": {
                "label": "🌍 BAAI/bge-m3 — Çok Dilli (Önerilen)",
                "desc": "Türkçe dahil 100+ dil destekler. En yüksek doğruluk. İlk yükleme ~2 dk.",
                "lang": "🌍 Çok Dilli", "speed": "🐢 Yavaş", "quality": "⭐⭐⭐⭐⭐",
            },
            "intfloat/multilingual-e5-small": {
                "label": "🌍 multilingual-e5-small — Çok Dilli Hafif",
                "desc": "Türkçe destekli, hızlı ve hafif. Orta düzey doğruluk.",
                "lang": "🌍 Çok Dilli", "speed": "⚡ Hızlı", "quality": "⭐⭐⭐",
            },
            "BAAI/bge-small-en-v1.5": {
                "label": "🇬🇧 bge-small-en — Sadece İngilizce",
                "desc": "Sadece İngilizce belgeler için. Çok hızlı ama Türkçe metinleri tam anlayamaz.",
                "lang": "🇬🇧 İngilizce", "speed": "⚡ Çok Hızlı", "quality": "⭐⭐",
            },
            "all-MiniLM-L6-v2": {
                "label": "🇬🇧 MiniLM-L6 — Sadece İngilizce Hafif",
                "desc": "En hafif model. Sadece İngilizce. Denemeler için uygundur.",
                "lang": "🇬🇧 İngilizce", "speed": "⚡ En Hızlı", "quality": "⭐⭐",
            },
        }

        current_model = os.getenv("EMBEDDING_MODEL", meta_k.get("embedding_model", "BAAI/bge-small-en-v1.5"))
        model_keys = list(EMBEDDING_OPTIONS.keys())
        current_idx = model_keys.index(current_model) if current_model in model_keys else 0

        embedding_model = st.selectbox(
            "🧠 Embedding Model",
            options=model_keys,
            index=current_idx,
            format_func=lambda x: EMBEDDING_OPTIONS[x]["label"],
        )

        # Seçilen modelin bilgi kartı
        selected_info = EMBEDDING_OPTIONS[embedding_model]
        st.markdown(f"""
        <div class="info-card" style="font-size: 0.82rem;">
            <div style="margin-bottom: 0.4rem;">{selected_info['desc']}</div>
            <table style="width:100%; font-size:0.78rem; color: var(--text-secondary);">
                <tr>
                    <td><strong>Dil:</strong> {selected_info['lang']}</td>
                    <td><strong>Hız:</strong> {selected_info['speed']}</td>
                    <td><strong>Kalite:</strong> {selected_info['quality']}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        if embedding_model != current_model and meta_k.get("total_chunks", 0) > 0:
            st.warning("⚠️ Model değiştirildi! Mevcut case'deki belgeler eski modelle oluşturuldu. "
                       "Yeni model kullanmak için case'i sıfırlayıp dosyaları tekrar yükleyin.")

        st.markdown("---")

        # ── FAISS Metric Seçimi ──
        METRIC_OPTIONS = {
            "cosine": {
                "label": "📐 Cosine (Kosinüs Benzerliği) — Önerilen",
                "desc": "İki metnin **anlam yönünü** karşılaştırır. Metin uzunluğundan etkilenmez. "
                        "Hukuk belgeleri için en doğru sonuçları verir.",
                "detail": "Kısa bir soru ile uzun bir paragraf aynı konudaysa yine 'benzer' bulur."
            },
            "l2": {
                "label": "📏 L2 (Öklid Mesafesi)",
                "desc": "İki vektör arasındaki **düz çizgi mesafesini** ölçer. "
                        "Metin uzunluğundan etkilenebilir.",
                "detail": "Genelde görüntü/ses araması için tercih edilir. Metin aramasında cosine daha iyidir."
            },
        }

        current_metric = os.getenv("FAISS_METRIC", meta_k.get("faiss_metric", "cosine"))
        metric_keys = list(METRIC_OPTIONS.keys())
        current_metric_idx = metric_keys.index(current_metric) if current_metric in metric_keys else 0

        faiss_metric = st.selectbox(
            "📐 FAISS Metric",
            options=metric_keys,
            index=current_metric_idx,
            format_func=lambda x: METRIC_OPTIONS[x]["label"],
        )

        # Seçilen metriğin bilgi kartı
        metric_info = METRIC_OPTIONS[faiss_metric]
        st.markdown(f"""
        <div class="info-card" style="font-size: 0.82rem;">
            <div style="margin-bottom: 0.3rem;">{metric_info['desc']}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); font-style: italic;">
                💡 {metric_info['detail']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if faiss_metric != current_metric and meta_k.get("total_chunks", 0) > 0:
            st.warning("⚠️ Metric değiştirildi! Mevcut case'deki index eski metric ile oluşturuldu. "
                       "Yeni metric kullanmak için case'i sıfırlayıp dosyaları tekrar yükleyin.")

        st.markdown("---")

        # ── Chunk Boyutu ──
        chunk_size = st.slider(
            "📏 Chunk boyutu (karakter)",
            min_value=200,
            max_value=1200,
            value=int(meta_k.get("chunk_size", 500) or 500),
            step=50,
            help="Her metin parçasının yaklaşık karakter uzunluğu. Küçük = daha detaylı arama, Büyük = daha fazla bağlam.",
        )
        st.caption("💡 500-600 arası çoğu hukuk belgesi için idealdir.")

    st.divider()

    # ── Case İçeriği Özet ──
    st.markdown("### 📊 Dosya Durumu")

    for coll, icon, title in [("knowledge", "📚", "Bilgi Dosyaları"), ("templates", "📝", "Şablonlar")]:
        meta_now = load_store_meta_for(selected_case, coll)
        files = meta_now.get("files", {})
        total = meta_now.get("total_chunks", 0) if files else 0
        status_class = "status-ready" if files else "status-empty"
        status_text = f"{len(files)} dosya" if files else "Boş"

        st.markdown(f"""
        <div class="info-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span>{icon} <strong>{title}</strong></span>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
            <div style="margin-top:0.4rem;">
                <span class="stat-value">{total}</span>
                <span class="stat-label"> chunk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if files:
            with st.expander(f"{title} detay", expanded=False):
                for fn, info in files.items():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.caption(f"📄 {fn} ({info.get('n_chunks', '?')} chunk)")
                    with col_b:
                        if st.button("🗑️", key=f"del_{coll}_{fn}", help=f"{fn} dosyasını sil"):
                            delete_file_from_case(selected_case, coll, fn)
                            st.warning(f"'{fn}' silindi. ⚠️ Index yeniden oluşturulmalı.")
                            st.rerun()

    st.divider()

    # ── Case Sıfırlama ──
    with st.expander("🔄 Case Yönetimi"):
        if st.button("🗑️ Bu case'i tamamen sıfırla", type="secondary"):
            for c in DEFAULT_COLLECTIONS:
                index_path, chunks_path, meta_path = store_paths(selected_case, c)
                for p in (index_path, chunks_path, meta_path):
                    if p.exists():
                        p.unlink()
            st.warning("Case sıfırlandı.")
            st.session_state.chat_history = []
            st.rerun()

        if st.button("💬 Sohbet geçmişini temizle"):
            st.session_state.chat_history = []
            st.rerun()

    # ── LLM Durum ──
    st.divider()
    client_check, model_name, provider_name = get_llm_client()
    if client_check:
        st.markdown(f"""
        <div class="info-card">
            <span class="status-badge status-ready">Bağlı</span>
            <div style="margin-top:0.4rem; font-size:0.8rem; color: var(--text-secondary);">
                {provider_name.upper()} / {model_name}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
            <span class="status-badge status-error">LLM Yok</span>
            <div style="margin-top:0.4rem; font-size:0.78rem; color: var(--text-muted);">
                .env dosyasına API key ekleyin
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        ⚖️ Hukuk Asistanı AI v2.0<br>
        <span style="font-size:0.65rem;">Bu yazılım hukuki danışmanlık yerine geçmez.</span>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
#  ANA İÇERİK — Tabs
# ╚══════════════════════════════════════════════════════════════╝

tab_chat, tab_upload, tab_draft = st.tabs(["💬 Soru-Cevap", "📎 Dosya Yükle", "📝 Taslak Üret"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1: SORU-CEVAP (Chat)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_chat:
    index_k, chunks_k = load_index_and_chunks(selected_case, "knowledge")

    if index_k is None or not chunks_k:
        st.markdown("""
        <div class="welcome-container animate-in">
            <h2>⚖️ Hoşgeldiniz</h2>
            <p>
                Hukuk Asistanı AI ile belgelerinizi analiz edebilir, sorularınızı yanıtlayabilir
                ve profesyonel hukuki taslaklar oluşturabilirsiniz.
            </p>
            <p style="color: var(--accent-gold); font-weight: 500;">
                Başlamak için <strong>"📎 Dosya Yükle"</strong> sekmesinden belgelerinizi yükleyin.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Sohbet Ayarları ──
        with st.expander("🔍 Arama Ayarları", expanded=False):
            k_val = st.slider("Kaç parça getirilsin (k)", min_value=1, max_value=20, value=6, key="qa_k")
            show_sources = st.checkbox("Kaynak parçalarını göster", value=True)

        # ── Hızlı Sorular ──
        st.markdown("**⚡ Hızlı Sorular:**")
        quick_cols = st.columns(3)
        for idx, qq in enumerate(QUICK_QUESTIONS):
            col_idx = idx % 3
            with quick_cols[col_idx]:
                if st.button(qq["label"], key=f"quick_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": qq["prompt"]})
                    # Arama yap
                    retrieved = perform_search(
                        selected_case, "knowledge", qq["prompt"],
                        k=k_val, embedding_model=embedding_model, faiss_metric=faiss_metric,
                    )
                    # LLM'e gönder
                    client, llm_model, provider = get_llm_client()
                    if client and retrieved:
                        context = format_chunks_for_llm(retrieved)
                        prompt = QA_PROMPT_TEMPLATE.format(context=context, query=qq["prompt"])
                        # Konuşma hafızası: son 6 mesajı ekle
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                        for msg in st.session_state.chat_history[-6:]:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                        messages.append({"role": "user", "content": prompt})
                        resp = client.chat.completions.create(
                            model=llm_model,
                            messages=messages,
                            temperature=0.2,
                        )
                        answer = resp.choices[0].message.content.strip()
                    elif retrieved:
                        answer = "⚠️ LLM bağlantısı yok. İlgili belgeler:\n\n" + format_chunks_for_llm(retrieved)
                    else:
                        answer = "❌ İlgili belge bulunamadı."

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

        st.divider()

        # ── Chat Geçmişi ──
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
                st.markdown(msg["content"])

        # ── Chat Input ──
        if user_query := st.chat_input("Belgeleriniz hakkında soru sorun..."):
            # Kullanıcı mesajını ekle
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)

            # Arama yap
            with st.chat_message("assistant", avatar="⚖️"):
                with st.spinner("🔍 Belgeler aranıyor..."):
                    retrieved = perform_search(
                        selected_case, "knowledge", user_query,
                        k=k_val, embedding_model=embedding_model, faiss_metric=faiss_metric,
                    )

                if not retrieved:
                    answer = "❌ İlgili belge parçası bulunamadı. Lütfen farklı bir soru deneyin veya daha fazla belge yükleyin."
                    st.markdown(answer)
                else:
                    # Kaynakları göster
                    if show_sources:
                        with st.expander(f"📎 {len(retrieved)} kaynak parça bulundu", expanded=False):
                            for i, ch in enumerate(retrieved, start=1):
                                src = f"📄 {ch.get('source_file', '?')} — Sayfa {ch.get('page', '?')}"
                                st.markdown(f"""
                                <div class="source-cite">
                                    <strong>#{i}</strong> {src}<br>
                                    <span style="font-size:0.8rem;">{ch.get('text', '')[:200]}...</span>
                                </div>
                                """, unsafe_allow_html=True)

                    # LLM yanıt üret
                    client, llm_model, provider = get_llm_client()
                    if client is None:
                        answer = "⚠️ **LLM bağlantısı yok.** Yalnızca ilgili belge parçaları gösterildi.\n\n"
                        answer += "`.env` dosyasına geçerli bir API anahtarı ekleyerek tam yanıt alabilirsiniz."
                        st.warning(answer)
                    else:
                        context = format_chunks_for_llm(retrieved)
                        prompt = QA_PROMPT_TEMPLATE.format(context=context, query=user_query)

                        # Konuşma hafızası
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                        recent = st.session_state.chat_history[-6:]
                        for msg in recent:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                        messages.append({"role": "user", "content": prompt})

                        with st.spinner(f"✨ Yanıt üretiliyor ({provider} / {llm_model})..."):
                            resp = client.chat.completions.create(
                                model=llm_model,
                                messages=messages,
                                temperature=0.2,
                            )
                        answer = resp.choices[0].message.content.strip()
                        st.markdown(answer)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2: DOSYA YÜKLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_upload:
    st.markdown("""
    <div class="animate-in">
        <h3 style="color: var(--accent-gold);">📎 Belge Yükle</h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">
            PDF, DOCX veya TXT dosyalarını yükleyerek bilgi tabanınızı oluşturun.
        </p>
    </div>
    """, unsafe_allow_html=True)

    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        target_collection = st.radio(
            "Koleksiyon türü",
            options=["knowledge", "templates"],
            format_func=lambda x: "📚 Bilgi Dosyaları (dava dosyası, sözleşme, delil)" if x == "knowledge" else "📝 Şablon Dosyaları (örnek dilekçe/sözleşme)",
            horizontal=True,
        )

        uploaded_files = st.file_uploader(
            "Dosya seçin",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="PDF, DOCX veya TXT formatında dosyalar yükleyebilirsiniz.",
        )

        if st.button("📤 Dosyaları Yükle ve İşle", type="primary", disabled=not uploaded_files):
            try:
                total_added = 0
                progress_bar = st.progress(0, text="Dosyalar işleniyor...")

                for idx, uf in enumerate(uploaded_files):
                    progress_bar.progress(
                        (idx + 1) / len(uploaded_files),
                        text=f"İşleniyor: {uf.name} ({idx+1}/{len(uploaded_files)})"
                    )
                    res = add_file_to_case(
                        case_name=selected_case,
                        collection=target_collection,
                        file_name=uf.name,
                        file_bytes=uf.getvalue(),
                        embedding_model=embedding_model,
                        faiss_metric=faiss_metric,
                        chunk_size=chunk_size,
                    )
                    total_added += int(res["added_chunks"])

                progress_bar.progress(1.0, text="✅ Tamamlandı!")
                st.success(f"✅ **{len(uploaded_files)} dosya** başarıyla eklendi. Toplam **{total_added} chunk** oluşturuldu.")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Hata: {str(e)}")

    with upload_col2:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: var(--accent-gold); margin:0 0 0.5rem;">💡 İpuçları</h4>
            <ul style="color: var(--text-secondary); font-size: 0.82rem; padding-left: 1.2rem;">
                <li><strong>Bilgi dosyaları:</strong> Dava dosyası, sözleşme metni, yazışmalar, deliller</li>
                <li><strong>Şablon dosyaları:</strong> Örnek dilekçe, sözleşme formatları (taslak üretmek için)</li>
                <li>Çoklu dosya aynı anda yüklenebilir</li>
                <li>PDF, DOCX ve TXT formatları desteklenir</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Mevcut dosya özeti
        st.markdown("#### 📋 Mevcut Dosyalar")
        for coll, icon, title in [("knowledge", "📚", "Bilgi"), ("templates", "📝", "Şablon")]:
            meta_now = load_store_meta_for(selected_case, coll)
            files = meta_now.get("files", {})
            if files:
                st.markdown(f"**{icon} {title}:**")
                for fn, info in files.items():
                    added = info.get("added_at", "")
                    st.caption(f"• {fn} — {info.get('n_chunks', '?')} chunk")
            else:
                st.caption(f"{icon} {title}: henüz dosya yok")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3: TASLAK ÜRET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_draft:
    st.markdown("""
    <div class="animate-in">
        <h3 style="color: var(--accent-gold);">📝 Hukuki Taslak Üretici</h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">
            Bilgi dosyalarınız ve şablonlarınıza dayanarak profesyonel hukuki belge taslakları oluşturun.
        </p>
    </div>
    """, unsafe_allow_html=True)

    index_k_d, chunks_k_d = load_index_and_chunks(selected_case, "knowledge")
    index_t_d, chunks_t_d = load_index_and_chunks(selected_case, "templates")

    has_knowledge = index_k_d is not None and len(chunks_k_d) > 0
    has_templates = index_t_d is not None and len(chunks_t_d) > 0

    if not has_knowledge:
        st.info("📚 Taslak üretmek için önce **Bilgi** koleksiyonuna dosya yükleyin.")
    if not has_templates:
        st.info("📝 Taslak üretmek için **Şablon** koleksiyonuna örnek dilekçe/sözleşme yükleyin.")

    if has_knowledge and has_templates:
        draft_col1, draft_col2 = st.columns([2, 1])

        with draft_col1:
            draft_type = st.selectbox("Taslak türü", options=DRAFT_TYPES, index=0)

            instruction = st.text_area(
                "Talimat",
                height=100,
                placeholder="Örn: İşçilik alacakları için dava dilekçesi yaz. Talepler: kıdem, ihbar, fazla mesai...",
            )

            extra_details = st.text_area(
                "Ek bilgiler",
                height=80,
                placeholder="Örn: Davalı ünvanı, tarihler, tutarlar, yetkili mahkeme, deliller...",
            )

        with draft_col2:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: var(--accent-gold); margin: 0 0 0.5rem;">📐 Taslak Ayarları</h4>
            </div>
            """, unsafe_allow_html=True)
            k_knowledge = st.slider("Bilgi parça sayısı (k)", 3, 30, 12, key="draft_k_know")
            k_templates = st.slider("Şablon parça sayısı (k)", 1, 20, 6, key="draft_k_tmpl")

        if st.button("✨ Taslak Üret", type="primary", disabled=not instruction.strip(), use_container_width=True):
            client, llm_model, provider = get_llm_client()
            if client is None:
                st.error("❌ Taslak üretmek için LLM bağlantısı gerekli. `.env` dosyasına API key ekleyin.")
            else:
                with st.spinner("📝 Taslak hazırlanıyor..."):
                    # Bilgi ve şablon araması
                    q = f"{draft_type}: {instruction}\n{extra_details}".strip()
                    retrieved_k = perform_search(selected_case, "knowledge", q, k_knowledge, embedding_model, faiss_metric)
                    retrieved_t = perform_search(selected_case, "templates", q, k_templates, embedding_model, faiss_metric)

                    context_k = format_chunks_for_llm(retrieved_k) if retrieved_k else "(bilgi bulunamadı)"
                    context_t = format_chunks_for_llm(retrieved_t) if retrieved_t else "(şablon bulunamadı)"

                    prompt = DRAFT_PROMPT_TEMPLATE.format(
                        draft_type=draft_type,
                        instruction=instruction,
                        extra_details=extra_details or "(ek bilgi girilmedi)",
                        context_knowledge=context_k,
                        context_templates=context_t,
                    )

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]

                    resp = client.chat.completions.create(
                        model=llm_model,
                        messages=messages,
                        temperature=0.2,
                    )
                    draft_text = resp.choices[0].message.content.strip()

                # Sonuçları göster
                st.markdown("---")
                st.markdown("### 📄 Oluşturulan Taslak")
                st.markdown(draft_text)

                # Kaynak gösterimi
                with st.expander("📎 Kullanılan Kaynaklar"):
                    src_col1, src_col2 = st.columns(2)
                    with src_col1:
                        st.markdown("**📚 Bilgi Dosyaları:**")
                        for ch in retrieved_k:
                            st.caption(f"• {ch.get('source_file', '?')} s.{ch.get('page', '?')}")
                    with src_col2:
                        st.markdown("**📝 Şablonlar:**")
                        for ch in retrieved_t:
                            st.caption(f"• {ch.get('source_file', '?')} s.{ch.get('page', '?')}")

                # Export butonları
                st.markdown("---")
                export_col1, export_col2, export_col3 = st.columns(3)
                with export_col1:
                    st.download_button(
                        "📥 TXT İndir",
                        data=draft_text,
                        file_name=f"{safe_slug(draft_type)}_taslak.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with export_col2:
                    # Markdown olarak indir
                    st.download_button(
                        "📥 Markdown İndir",
                        data=draft_text,
                        file_name=f"{safe_slug(draft_type)}_taslak.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with export_col3:
                    st.download_button(
                        "📋 Panoya Kopyala (TXT)",
                        data=draft_text,
                        file_name=f"{safe_slug(draft_type)}_taslak_copy.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
