# 🩺 Basic RAG - Asisten Tanya-Jawab Kesehatan (Vector-Only)

Folder ini berisi implementasi **Basic RAG** yang berdiri sendiri (standalone) untuk sistem tanya-jawab informasi kesehatan berbasis artikel medis terpercaya.

---

## 🏗️ Arsitektur Pipeline

Basic RAG menggunakan pencarian berbasis kemiripan semantik murni (*Dense Vector Retrieval*):

```
Pertanyaan Pengguna
       │
       ▼
[Ollama Embedding: bge-m3]
       │
       ▼
[Qdrant Vector Database] ── (Top-K Cosine Similarity Search)
       │
       ▼
[Prompt Formatter] ── Context & Guardrail Medis
       │
       ▼
[Ollama LLM: qwen3:1.7b] ── Structured Output (JSON Schema)
       │
       ▼
Jawaban Utama + Referensi Artikel + Disclaimer Medis
```

---

## 📁 Struktur Direktori

```
basic/
├── .env                     # Variabel konfigurasi lingkungan
├── requirements.txt         # Daftar pustaka Python yang dibutuhkan
├── README_BASIC.md          # Dokumentasi panduan penggunaan ini
├── logs/                    # Folder log proses eksekusi
├── src/
│   ├── __init__.py
│   ├── config.py            # Konfigurasi model, Qdrant, Ollama, & chunking
│   ├── eval_core.py         # Modul inti evaluasi metrik RAGAS & heuristic
│   ├── ingest.py            # Pipeline ingestion CSV ke Qdrant (Vector-only)
│   └── basic_rag/
│       ├── __init__.py
│       ├── rag_chain.py     # Logika retrieval semantik & chain LLM
│       └── schema.py        # Pydantic schema hasil structured output
├── ui/
│   └── basic_rag_app.py     # Aplikasi web interaktif berbasis Streamlit
└── evaluasi/
    ├── __init__.py
    ├── buat_dataset_test.py # Generator dataset kasus uji
    └── evaluasi_basic.py    # Script evaluasi RAGAS mandiri untuk Basic RAG
```

> **Catatan Data:** Dataset artikel kesehatan tersimpan di folder `basic/data/artikel_kesehatan_klikdokter.csv` (atau `artikel_kesehatan.csv`), dan database vektor lokal tersimpan di `basic/qdrant_storage/`.

---

## ⚙️ Menjalankan Sistem (Dua Metode)

### Metode A: Menggunakan Docker Compose (Full Containerized - Recommended)
Seluruh komponen (Qdrant, Ollama, dan Streamlit UI) berjalan otomatis di dalam Docker:

```powershell
# Jalankan dari folder basic/
cd basic
docker compose up -d

# Registrasi model qwen3-medis dan pull bge-m3 ke container Ollama (pertama kali saja):
docker exec -it ollama_kesehatan ollama create qwen3-medis -f /models/Modelfile
docker exec -it ollama_kesehatan ollama pull bge-m3
```
Aplikasi Streamlit langsung aktif di browser pada alamat `http://localhost:8080`.

---

### Metode B: Menjalankan Secara Manual (Host / Bare-Metal)

1. **Docker Qdrant & Ollama** (port `6333` dan `11434`):
   ```powershell
   cd basic
   docker compose up -d qdrant ollama
   ```
2. **Python Virtual Environment** (aktifkan venv project):
   ```powershell
   cd basic
   .\venv\Scripts\Activate.ps1
   ```

---

## 🚀 Panduan Penggunaan

### 1. Ingestion Data (Opsional / Jika Koleksi Baru)
Memasukkan dokumen artikel kesehatan ke Qdrant vector database:

```powershell
# Jalankan dari folder basic/
cd basic
python src/ingest.py

# Atau jika ingin mereset koleksi dari awal:
python src/ingest.py --reset

# Atau jika ingin menggunakan file dataset CSV tertentu:
python src/ingest.py --file data/artikel_kesehatan_klikdokter.csv
```

### 2. Menjalankan UI Chat (Streamlit)
Membuka antarmuka chat tanya-jawab kesehatan:

```powershell
# Jalankan dari folder basic/
cd basic
streamlit run ui/basic_rag_app.py --server.port 8080
```
Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8080`.

### 3. Menjalankan Evaluasi RAGAS

Script `evaluasi_basic.py` dapat mengevaluasi pipeline Basic RAG secara mandiri:

```powershell
# Jalankan dari folder basic/
cd basic

# Smoke test (3 pertanyaan pertama)
python evaluasi/evaluasi_basic.py --limit 3

# Evaluasi cepat dengan skor heuristik (tanpa memanggil LLM evaluator)
python evaluasi/evaluasi_basic.py --fast --limit 5

# Evaluasi kategori tertentu (contoh: happy-path, edge-cases, multi-hop)
python evaluasi/evaluasi_basic.py --category happy-path

# Evaluasi lengkap seluruh dataset
python evaluasi/evaluasi_basic.py
```

Laporan hasil evaluasi (format JSON, CSV, dan Excel) akan tersimpan di folder `hasil evaluasi/`.
