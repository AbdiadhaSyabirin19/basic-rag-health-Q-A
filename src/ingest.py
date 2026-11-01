"""Ingestion pipeline: memuat CSV, memecah menjadi chunk, dan mengirim ke Qdrant.

Mendukung:
- Checkpoint otomatis untuk melanjutkan proses yang terinterupsi.
- Graceful exit (Ctrl+C) yang menyelesaikan batch aktif sebelum keluar.
- Sinkronisasi otomatis antara status Qdrant dan file checkpoint.

Catatan: Ini adalah versi Basic RAG — tidak membangun indeks BM25.
"""

import json
import logging
import os
import signal
import sys
from pathlib import Path

# Memastikan root folder basic/ ada di sys.path agar impor 'from src import config' selalu berhasil
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Fallback: jika dijalankan via python global (misal base Anaconda), muat paket dari venv
for _sp in [
    ROOT_DIR / "venv" / "Lib" / "site-packages",
    ROOT_DIR.parent / "venv" / "Lib" / "site-packages",
]:
    if _sp.is_dir() and str(_sp) not in sys.path:
        sys.path.append(str(_sp))

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

from src import config

logger = logging.getLogger(__name__)

# Variabel global untuk mendeteksi permintaan berhenti
_stop_requested = False

def _resolusi_path(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    candidates = [
        p,
        ROOT_DIR / rel_path,
        ROOT_DIR.parent / rel_path,
        Path("C:/Dokumen/riset/project/data") / p.name,
        Path("C:/Dokumen/riset/project") / p.name,
        ROOT_DIR.parent.parent / "project" / "data" / p.name,
        ROOT_DIR.parent.parent / "project" / p.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (ROOT_DIR / rel_path).resolve()


def _cari_dataset_csv(custom_path: str | None = None) -> Path:
    """Mencari file CSV dataset artikel kesehatan dengan prioritas."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()
        if (ROOT_DIR / custom_path).exists():
            return (ROOT_DIR / custom_path).resolve()
        raise FileNotFoundError(f"File dataset CSV yang diminta tidak ditemukan: '{custom_path}'")

    nama_file_kandidat = [
        "artikel_kesehatan_klikdokter.csv",
        "artikel kesehatan klikdokter.csv",
        "artikel_kesehatan.csv",
        "artikel kesehatan.csv",
        "klikdokter_test.csv",
    ]
    folder_kandidat = [
        ROOT_DIR / "data",
        Path("data"),
        ROOT_DIR.parent / "data",
        Path("C:/Dokumen/riset/project/data"),
        Path("C:/Dokumen/riset/project"),
        ROOT_DIR.parent.parent / "project" / "data",
        ROOT_DIR.parent.parent / "project",
    ]

    for folder in folder_kandidat:
        for nama in nama_file_kandidat:
            target = folder / nama
            if target.is_file():
                return target.resolve()

    return (ROOT_DIR / "data" / "artikel_kesehatan_klikdokter.csv").resolve()


# Pastikan folder data/ ada
(ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = _resolusi_path("data/checkpoint.json")
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------


def _graceful_exit_handler(signum, frame) -> None:
    """Menangkap Ctrl+C dan menandai agar loop berhenti setelah batch aktif selesai."""
    global _stop_requested
    if not _stop_requested:
        logger.warning(
            "Menangkap sinyal berhenti (Ctrl+C). "
            "Menyelesaikan batch aktif sebelum keluar secara aman..."
        )
        _stop_requested = True
    else:
        logger.warning("Memaksa keluar segera...")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Tahap 1: Memuat dokumen
# ---------------------------------------------------------------------------


def muat_dokumen(csv_path: str | Path) -> list[Document]:
    """Memuat CSV artikel dan mengubahnya menjadi list Document LangChain.

    Args:
        csv_path: Path ke file CSV berisi kolom judul, tanggal, url, label, konten.

    Returns:
        List Document LangChain dengan metadata lengkap.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"File CSV artikel kesehatan tidak ditemukan di '{csv_path}'. "
            "Pastikan file artikel_kesehatan.csv ada di folder 'data/' atau path yang valid."
        )
    logger.info("Membaca file dataset: %s", p)
    df = pd.read_csv(p)
    df = df.dropna(subset=["konten"])

    dokumen: list[Document] = []
    for _, row in df.iterrows():
        dokumen.append(Document(
            page_content=str(row["konten"]),
            metadata={
                "judul": str(row["judul"]) if pd.notna(row["judul"]) else "",
                "tanggal": str(row["tanggal"]) if pd.notna(row["tanggal"]) else "",
                "url": str(row["url"]) if pd.notna(row["url"]) else "",
                "label": str(row["label"]) if pd.notna(row["label"]) else "",
            },
        ))
    return dokumen


# ---------------------------------------------------------------------------
# Tahap 2: Chunking
# ---------------------------------------------------------------------------


def pecah_menjadi_chunk(dokumen: list[Document]) -> list[Document]:
    """Memecah setiap dokumen artikel menjadi beberapa chunk lebih kecil.

    Menggunakan RecursiveCharacterTextSplitter dengan separator hierarkis
    untuk mempertahankan koherensi teks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(dokumen)


# ---------------------------------------------------------------------------
# Tahap 3: Setup Qdrant
# ---------------------------------------------------------------------------


def _deteksi_dimensi_embedding(embeddings: OllamaEmbeddings) -> int:
    """Menguji dimensi model embedding dengan mengirim query test."""
    logger.info("Menguji dimensi model embedding...")
    try:
        test_vector = embeddings.embed_query("test")
        dimensi = len(test_vector)
        logger.info("Dimensi model embedding terdeteksi: %d", dimensi)
        return dimensi
    except Exception as e:
        logger.error(
            "Gagal terhubung ke Ollama di %s. Detail: %s",
            config.OLLAMA_BASE_URL, e,
        )
        sys.exit(1)


def _siapkan_koleksi(
    client: QdrantClient,
    vector_size: int,
    *,
    reset: bool = False,
) -> None:
    """Membuat atau mereset koleksi Qdrant sesuai kebutuhan.

    Args:
        client: Instance QdrantClient.
        vector_size: Dimensi vektor embedding.
        reset: Jika True, hapus dan buat ulang koleksi yang sudah ada.
    """
    nama = config.COLLECTION_NAME
    vector_cfg = VectorParams(size=vector_size, distance=Distance.COSINE)

    if not client.collection_exists(nama):
        logger.info("Membuat koleksi baru '%s' di Qdrant...", nama)
        client.create_collection(collection_name=nama, vectors_config=vector_cfg)
    elif reset:
        logger.info("Menghapus dan membuat ulang koleksi '%s'...", nama)
        client.delete_collection(nama)
        client.create_collection(collection_name=nama, vectors_config=vector_cfg)
    else:
        logger.info("Koleksi '%s' sudah ada. Melanjutkan...", nama)


# ---------------------------------------------------------------------------
# Tahap 4: Sinkronisasi progress
# ---------------------------------------------------------------------------


def _muat_checkpoint() -> dict | None:
    """Membaca file checkpoint jika ada dan model embedding cocok.

    Returns:
        Dict checkpoint jika valid, None jika tidak ditemukan atau tidak cocok.
    """
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if data.get("embedding_model") == config.EMBEDDING_MODEL:
            return data
    except Exception:
        pass
    return None


def _simpan_checkpoint(last_index: int, total_chunks: int) -> None:
    """Menyimpan progress ke file checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({
            "embedding_model": config.EMBEDDING_MODEL,
            "last_index_in_qdrant": last_index,
            "total_chunks": total_chunks,
        }),
        encoding="utf-8",
    )


def _sinkronkan_progress(
    client: QdrantClient,
    total_chunks: int,
) -> int:
    """Menentukan indeks awal pengiriman berdasarkan status Qdrant dan checkpoint.

    Membandingkan jumlah point di Qdrant dengan checkpoint file,
    lalu menggunakan acuan yang paling aman (database Qdrant).

    Args:
        client: Instance QdrantClient.
        total_chunks: Total chunk yang harus diproses.

    Returns:
        Indeks awal untuk melanjutkan pengiriman.
    """
    # Ambil jumlah point aktual di Qdrant
    qdrant_count = 0
    try:
        qdrant_count = client.count(collection_name=config.COLLECTION_NAME).count
    except Exception as e:
        logger.warning("Gagal mendeteksi jumlah data di Qdrant: %s", e)

    # Ambil index dari checkpoint
    checkpoint = _muat_checkpoint()
    checkpoint_index = checkpoint.get("last_index_in_qdrant") if checkpoint else None

    # Log status sinkronisasi
    if checkpoint_index is not None:
        logger.info("Index di checkpoint.json : %d", checkpoint_index)
        logger.info("Point aktual di Qdrant   : %d", qdrant_count)
        if checkpoint_index == qdrant_count:
            logger.info("Status: SINKRON. Melanjutkan dari indeks: %d", qdrant_count)
        else:
            logger.warning(
                "Status: TIDAK SINKRON (interupsi sebelumnya). "
                "Menggunakan acuan aman database Qdrant (%d).",
                qdrant_count,
            )
    else:
        logger.info("Index di checkpoint.json : Tidak ditemukan / Baru")
        logger.info("Point aktual di Qdrant   : %d", qdrant_count)
        logger.info("Melanjutkan dari indeks database Qdrant: %d", qdrant_count)

    if qdrant_count >= total_chunks:
        logger.info(
            "Proses ingestion sudah selesai (%d/%d chunk). "
            "Gunakan '--reset' untuk mengulang.",
            qdrant_count, total_chunks,
        )
        sys.exit(0)

    return qdrant_count


# ---------------------------------------------------------------------------
# Tahap 5: Pengiriman batch
# ---------------------------------------------------------------------------


def _kirim_batch(
    vector_store: QdrantVectorStore,
    chunks: list[Document],
    start_idx: int,
    total_chunks: int,
) -> int:
    """Mengirim chunk ke Qdrant dalam batch dengan progress bar.

    Args:
        vector_store: Instance QdrantVectorStore.
        chunks: Seluruh list chunk.
        start_idx: Indeks awal pengiriman.
        total_chunks: Total jumlah chunk.

    Returns:
        Jumlah chunk yang berhasil dikirim pada sesi ini.
    """
    global _stop_requested
    chunks_terkirim = 0

    logger.info("Mengirim chunk ke Qdrant per batch (%d chunk)...", BATCH_SIZE)
    progress_bar = tqdm(
        initial=start_idx,
        total=total_chunks,
        desc="Progress Ingest",
    )

    try:
        for i in range(start_idx, total_chunks, BATCH_SIZE):
            if _stop_requested:
                break

            batch = chunks[i : i + BATCH_SIZE]
            vector_store.add_documents(batch)
            chunks_terkirim += len(batch)

            # Update progress berdasarkan index yang sudah diproses,
            # TANPA query client.count() setiap batch (menghindari network call)
            progress_sekarang = i + len(batch)
            progress_bar.n = progress_sekarang
            progress_bar.refresh()

            # Simpan checkpoint
            _simpan_checkpoint(
                last_index=progress_sekarang,
                total_chunks=total_chunks,
            )

        # Log status akhir
        total_terkirim = start_idx + chunks_terkirim
        if _stop_requested:
            logger.info(
                "Proses dihentikan dengan aman. "
                "Progress tersimpan: %d/%d chunk.",
                total_terkirim, total_chunks,
            )
        else:
            logger.info(
                "Selesai! Seluruh basis pengetahuan (%d chunk) berhasil disimpan.",
                total_terkirim,
            )
    except Exception as e:
        logger.error("Terjadi kesalahan selama ingestion: %s", e)
        logger.info("Proses dihentikan. Jalankan kembali untuk melanjutkan.")
    finally:
        progress_bar.close()

    return chunks_terkirim


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Menjalankan pipeline ingestion lengkap: load → chunk → embed → store.

    Versi Basic RAG: hanya membangun Vector Store (Qdrant).
    Tidak membuat indeks BM25.
    """
    global _stop_requested
    _stop_requested = False

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Registrasi handler Ctrl+C
    signal.signal(signal.SIGINT, _graceful_exit_handler)

    reset = "--reset" in sys.argv
    if reset and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint direset. Ingestion akan dimulai dari awal.")

    # Parse opsi opsional --file <path> jika user menentukan file khusus
    custom_file = None
    for i, arg in enumerate(sys.argv):
        if arg in ("--file", "-f", "--data") and i + 1 < len(sys.argv):
            custom_file = sys.argv[i + 1]
            break

    # Tahap 1: Muat dokumen
    logger.info("[1/3] Memuat dokumen dari CSV...")
    csv_file = _cari_dataset_csv(custom_file)
    logger.info("      File dataset: %s", csv_file)
    dokumen = muat_dokumen(csv_file)
    logger.info("      Total artikel: %d", len(dokumen))

    # Tahap 2: Pecah menjadi chunk
    logger.info("[2/3] Memecah artikel menjadi chunk...")
    chunks = pecah_menjadi_chunk(dokumen)
    total_chunks = len(chunks)
    logger.info("      Total chunk: %d", total_chunks)

    # Tahap 3: Siapkan embedding & Qdrant
    logger.info("[3/3] Menyiapkan model embedding lokal (Ollama)...")
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vector_size = _deteksi_dimensi_embedding(embeddings)

    logger.info("      Menginisialisasi koneksi Qdrant...")
    client = QdrantClient(url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}")
    _siapkan_koleksi(client, vector_size, reset=reset)

    # Tahap 4: Sinkronkan progress
    start_idx = 0
    if not reset:
        start_idx = _sinkronkan_progress(client, total_chunks)

    # Tahap 5: Kirim batch
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    _kirim_batch(vector_store, chunks, start_idx, total_chunks)


if __name__ == "__main__":
    main()
