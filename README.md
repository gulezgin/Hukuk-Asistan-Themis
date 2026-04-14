# Hukuk Asistanı Themis

Hukuk Asistanı Themis, avukatlar ve hukuk profesyonelleri için tasarlanmış, yapay zeka destekli bir asistan uygulamasıdır. Bu uygulama, dava yönetimi, belge analizi ve yapay zeka tabanlı soru-cevap özellikleri sunarak hukuk süreçlerini daha verimli hale getirmeyi amaçlar.

---

## 🚀 Özellikler

- **Chat Tabanlı Soru-Cevap Arayüzü**: Yapay zeka destekli bir sohbet arayüzü ile hızlı ve doğru yanıtlar alın.
- **Belge Analizi**: PDF, DOCX ve TXT formatındaki belgeleri yükleyerek analiz edin.
- **Dava Yönetimi**: Dava dosyalarını gruplar halinde organize edin ve yönetin.
- **Bilgi ve Şablon Koleksiyonları**: Hukuki bilgi ve şablonları kolayca yönetin.
- **Dilekçe ve Sözleşme Taslağı Üretimi**: Hızlı ve doğru taslaklar oluşturun.
- **Hızlı Soru Şablonları**: Sıkça sorulan sorular için hazır şablonlar.
- **Dosya Silme ve Dışa Aktarma**: Belgeleri kolayca silin ve dışa aktarın.

---

## 📂 Proje Yapısı

```
.eski_prototip/
│
├── app.py                # Ana uygulama dosyası (Streamlit tabanlı)
├── ask_question.py       # Komut satırı tabanlı soru-cevap aracı
├── build_vector_db.py    # Vektör veritabanı oluşturma aracı
├── prompts.py            # Yapay zeka istem şablonları
├── utils.py              # Yardımcı fonksiyonlar
├── requirements.txt      # Python bağımlılıkları
├── .env.example          # Çevresel değişkenler için örnek dosya
├── .streamlit/           # Streamlit yapılandırma dosyaları
├── data/                 # Vektör veritabanı ve örnek dosyalar
│   ├── AG_Application_Development_Contract_index.faiss
│   ├── vector_db_meta.json
│   ├── example_case/
│       ├── dava_dosyasi.txt
│       ├── dilekce.txt
│       └── sozlesme.txt
│   └── stores/
│       └── default/
│           └── knowledge/
│           └── templates/
│               └── meta.json
```

---

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- pip (Python Paket Yöneticisi)
- Git

### Adımlar

1. **Depoyu Klonlayın**:
   ```bash
   git clone https://github.com/gulezgin/Hukuk-Asistan-Themis.git
   cd Hukuk-Asistan-Themis/eski_prototip
   ```

2. **Sanal Ortam Oluşturun ve Aktif Edin**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows için: .\venv\Scripts\activate
   ```

3. **Bağımlılıkları Yükleyin**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Çevresel Değişkenleri Ayarlayın**:
   - `.env.example` dosyasını kopyalayın ve `.env` olarak yeniden adlandırın.
   - Gerekli API anahtarlarını ve diğer bilgileri doldurun.

---

## 📖 Kullanım

### Streamlit Uygulamasını Çalıştırma

Streamlit tabanlı kullanıcı arayüzünü başlatmak için:
```bash
streamlit run app.py
```

### Vektör Veritabanı Oluşturma (Opsiyonel)

Tek bir PDF dosyasından vektör veritabanı oluşturmak için:
```bash
python build_vector_db.py
```

### Komut Satırı Üzerinden Soru Sorma (Opsiyonel)

Komut satırı üzerinden soru sormak için:
```bash
python ask_question.py
```

---

## 📋 Çevresel Değişkenler

`.env` dosyasına aşağıdaki bilgileri ekleyin:

```
# OpenAI API Anahtarı
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Embedding Model ve FAISS Ayarları
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
FAISS_METRIC=cosine

# LLM Sağlayıcı ve Model
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-nano
```

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasını inceleyebilirsiniz.

---

## Katkıda Bulunma

1. Bu projeyi forklayın.
2. Yeni bir dal oluşturun (`git checkout -b feature/AmazingFeature`).
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`).
4. Dalınızı push edin (`git push origin feature/AmazingFeature`).
5. Bir Pull Request açın.

---

## İletişim

Herhangi bir sorunuz veya öneriniz varsa, lütfen [gulezgin](https://github.com/gulezgin) ile iletişime geçin.