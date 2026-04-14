"""
prompts.py — RAG Lawyer profesyonel hukuk prompt şablonları.

Farklı görev modları için optimize edilmiş sistem ve kullanıcı prompt'ları.
"""

# ╔══════════════════════════════════════════════════════════════╗
#  SYSTEM PROMPT — Genel avukat asistanı kimliği
# ╚══════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = """Sen "Hukuk Asistanı AI" adlı profesyonel bir yapay zeka avukat asistanısın.

KİMLİĞİN:
- Türk hukuku konusunda uzmanlaşmış, kıdemli bir hukuk danışmanı gibi davranırsın.
- Yanıtların her zaman profesyonel, açık ve hukuki terminolojiye uygun olmalıdır.
- Yasal uyarı: Verdiğin bilgiler hukuki danışmanlık yerine geçmez, bilgilendirme amaçlıdır.

KURALLAR:
1. SADECE verilen bağlam belgelerine dayanarak cevap ver.
2. Uydurma bilgi ASLA ekleme — bağlamda yoksa "Bu bilgi mevcut belgelerde bulunamadı" de.
3. Kaynak atfı yap: [Dosya adı, s.X] formatında.
4. Yanıtları maddeli, yapılandırılmış ve okunabilir ver.
5. Belirsiz durumlarda hangi ek belge/bilgi gerektiğini açıkça belirt.
6. Her zaman Türkçe yanıt ver (teknik terimler orijinal kalabilir).
"""

# ╔══════════════════════════════════════════════════════════════╗
#  SORU-CEVAP (QA) PROMPT
# ╚══════════════════════════════════════════════════════════════╝

QA_PROMPT_TEMPLATE = """Aşağıdaki KAYNAKLI BAĞLAM belgelerine dayanarak kullanıcının sorusunu yanıtla.

KAYNAKLI BAĞLAM:
{context}

KULLANICININ SORUSU:
{query}

YANITLAMA KURALLARI:
- Bağlamda bulunan bilgilere dayanan, maddeli ve yapılandırılmış bir yanıt ver.
- Her bilgi için kaynak atfı yap: [Dosya s.X #id]
- Bağlam yetersizse hangi belge/bilgi gerektiğini açıkça belirt.
- Hukuki analiz yapıyorsan ilgili maddeleri ve yorumlarını ayrı ayrı ele al.

YANIT:"""

# ╔══════════════════════════════════════════════════════════════╗
#  TASLAK ÜRETME (DRAFT) PROMPT
# ╚══════════════════════════════════════════════════════════════╝

DRAFT_PROMPT_TEMPLATE = """Aşağıdaki kaynaklara dayanarak profesyonel bir {draft_type} taslağı hazırla.

KULLANICI TALİMATI:
{instruction}

EK BİLGİLER:
{extra_details}

BİLGİ DOSYALARI (olay/evrak bağlamı):
{context_knowledge}

ŞABLONLAR (format/stil referansı):
{context_templates}

TASLAK HAZIRLAMA KURALLARI:
1. Şablonlardan yapısal formatı (başlıklar, bölümler, imza alanları) al.
2. Bilgi dosyalarından ve kullanıcı detaylarından somut bilgileri (taraf adları, tarihler, tutarlar, olaylar) doldur.
3. Resmi Türkçe dil ve hukuki terminoloji kullan.
4. Eksik olan bilgileri "⚠️ EKSİK BİLGİ" olarak işaretle ve sonunda liste halinde topla.
5. Mümkün olduğunda kaynak atfı yap: [Dosya s.X #id]
6. Mevzuat referansları ekle (varsa).

ÇIKTI FORMATI:
1. TASLAK METİN (profesyonel, resmi format)
2. ⚠️ EKSİK BİLGİ LİSTESİ (varsa)
3. 📝 NOTLAR (varsa — örn. dikkat edilmesi gereken hukuki noktalar)"""

# ╔══════════════════════════════════════════════════════════════╗
#  BELGE ANALİZ PROMPT
# ╚══════════════════════════════════════════════════════════════╝

ANALYSIS_PROMPT_TEMPLATE = """Aşağıdaki belge bağlamını analiz et ve kapsamlı bir hukuki değerlendirme hazırla.

BAĞLAM:
{context}

ANALİZ TALEP EDİLEN KONULAR:
{query}

ANALİZ FORMATI:
1. **ÖZET**: Belgenin genel amacı ve kapsamı
2. **ÖNEMLİ MADDELER**: Kritik hükümler ve yükümlülükler
3. **RİSK DEĞERLENDİRMESİ**: Potansiyel hukuki riskler
4. **TAVSİYELER**: Dikkat edilmesi gereken noktalar
5. **EKSİK/BELİRSİZ NOKTALAR**: Netleştirilmesi gereken hususlar

Her madde için kaynak atfı yap: [Dosya s.X #id]"""

# ╔══════════════════════════════════════════════════════════════╗
#  HIZLI SORU ŞEKLONLARı
# ╚══════════════════════════════════════════════════════════════╝

QUICK_QUESTIONS = [
    {
        "label": "📋 Belge Özeti",
        "prompt": "Bu belgelerin genel özetini çıkar. Temel konular, taraflar ve önemli tarihler nelerdir?"
    },
    {
        "label": "⚖️ Hukuki Riskler",
        "prompt": "Bu belgelerdeki potansiyel hukuki riskleri ve dikkat edilmesi gereken noktaları analiz et."
    },
    {
        "label": "📅 Önemli Tarihler",
        "prompt": "Bu belgelerdeki tüm önemli tarihleri, süreleri ve zaman aşımı sürelerini listele."
    },
    {
        "label": "💰 Mali Yükümlülükler",
        "prompt": "Bu belgelerdeki tüm mali yükümlülükleri, ödeme şartlarını ve cezai şartları listele."
    },
    {
        "label": "🔍 Taraf Yükümlülükleri",
        "prompt": "Her tarafın yükümlülüklerini ve sorumluluklarını maddeler halinde listele."
    },
    {
        "label": "📝 Fesih Koşulları",
        "prompt": "Fesih koşullarını, ihbar sürelerini ve fesih sonucu doğacak yükümlülükleri açıkla."
    },
]

# ╔══════════════════════════════════════════════════════════════╗
#  TASLAK TÜRLERİ
# ╚══════════════════════════════════════════════════════════════╝

DRAFT_TYPES = [
    "Dilekçe",
    "Sözleşme",
    "İhtarname",
    "Vekaletname",
    "Tutanak",
    "Bilirkişi Raporu",
    "Savunma Dilekçesi",
    "Temyiz Dilekçesi",
    "İcra Takibi",
    "Sulh Protokolü",
    "Diğer",
]
