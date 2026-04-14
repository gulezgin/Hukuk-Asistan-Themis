!!! :bangbang: Bu çalışma, Hukuk Asistanı THEMIS projesinin teknik uygulanabilirliğini doğrulamak amacıyla hazırlanan bir Proof of Concept (PoC) çalışmasıdır. !!!

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

<img width="1916" height="910" alt="Ekran görüntüsü 2026-04-14 152530" src="https://github.com/user-attachments/assets/2e431d91-6ebd-4c93-94c1-4b88be9590ef" />
<img width="1919" height="910" alt="Ekran görüntüsü 2026-04-14 152524" src="https://github.com/user-attachments/assets/951b6f8a-5a9f-4160-8d71-6ccf77e9be9c" />
<img width="1918" height="904" alt="Ekran görüntüsü 2026-04-14 152518" src="https://github.com/user-attachments/assets/ac8777ec-2230-402a-8b6d-a005d0b05d10" />
<img width="1884" height="860" alt="Ekran görüntüsü 2026-04-14 152015" src="https://github.com/user-attachments/assets/a0b54d10-cf4b-430f-ace3-448735064b81" />
<img width="1844" height="854" alt="Ekran görüntüsü 2026-04-14 152111" src="https://github.com/user-attachments/assets/a5d6d4f0-e889-42af-9f0a-470f12e2f18e" />

