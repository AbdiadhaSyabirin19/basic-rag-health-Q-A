"""RAG chain untuk tanya-jawab kesehatan berbasis artikel.

Modul ini menyediakan pipeline retrieval-augmented generation (RAG)
yang mengambil konteks dari Qdrant vector database dan menghasilkan
jawaban terstruktur menggunakan LLM lokal via Ollama.
"""

import functools
import logging
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from src import config
from src.basic_rag.schema import HasilRAG, JawabanTerkait

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inisialisasi komponen (lazy & cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _buat_embeddings() -> OllamaEmbeddings:
    """Membuat instance embedding Ollama (di-cache, hanya dibuat sekali)."""
    logger.info("Menginisialisasi model embedding: %s", config.EMBEDDING_MODEL)
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


@functools.lru_cache(maxsize=1)
def _buat_vector_store() -> QdrantVectorStore:
    """Membuat koneksi ke Qdrant vector store (di-cache, hanya dibuat sekali)."""
    logger.info("Menghubungkan ke Qdrant: %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)
    return QdrantVectorStore.from_existing_collection(
        embedding=_buat_embeddings(),
        url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}",
        collection_name=config.COLLECTION_NAME,
    )


def _dapatkan_retriever(score_threshold: float = config.SCORE_THRESHOLD):
    return _buat_vector_store().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": config.TOP_K_RETRIEVAL,
            "score_threshold": score_threshold, # Hanya ambil chunk dengan kemiripan >= 55%
        },
    )


def _hitung_jaccard_similarity(teks1: str, teks2: str) -> float:
    set1 = set(teks1.lower().split())
    set2 = set(teks2.lower().split())
    if not set1 or not set2:
        return 0.0
    irisan = len(set1.intersection(set2))
    gabungan = len(set1.union(set2))
    return irisan / gabungan if gabungan > 0 else 0.0


def _apakah_duplikat(chunk_baru, daftar_terpilih: list, ambang_batas: float = 0.65) -> bool:
    teks_baru = getattr(chunk_baru, "page_content", "").strip().lower()
    if not teks_baru:
        return True
    for c in daftar_terpilih:
        teks_lama = getattr(c, "page_content", "").strip().lower()
        if teks_baru == teks_lama or teks_baru in teks_lama or teks_lama in teks_baru:
            return True
        if _hitung_jaccard_similarity(teks_baru, teks_lama) >= ambang_batas:
            return True
    return False


def dapatkan_dokumen_relevan(
    pertanyaan: str,
    score_threshold: float = config.SCORE_THRESHOLD,
    k: int = config.TOP_K_RETRIEVAL,
) -> list[Document]:
    """Mengambil dokumen dari Qdrant dengan threshold dan deduplikasi otomatis."""
    vs = _buat_vector_store()
    try:
        kandidat_dengan_skor = vs.similarity_search_with_score(pertanyaan, k=k * 3)
        dokumen_terpilih: list[Document] = []
        for doc, skor in kandidat_dengan_skor:
            if skor < score_threshold:
                continue
            if not _apakah_duplikat(doc, dokumen_terpilih, ambang_batas=0.65):
                dokumen_terpilih.append(doc)
            if len(dokumen_terpilih) >= k:
                break
        return dokumen_terpilih
    except Exception as e:
        logger.warning("Gagal similarity_search_with_score terdeduplikasi: %s. Fallback ke retriever biasa.", e)
        retriever = _dapatkan_retriever(score_threshold)
        return retriever.invoke(pertanyaan)


@functools.lru_cache(maxsize=2)
def _dapatkan_llm_terstruktur(gunakan_thinking: bool):

    """Mengembalikan instance LLM dengan structured output (di-cache per mode thinking).

    Menggunakan ``lru_cache`` agar instance LLM hanya dibuat sekali per
    kombinasi parameter, menghindari overhead koneksi berulang.
    """
    logger.info(
        "Membuat instance LLM terstruktur (thinking=%s)",
        gunakan_thinking,
    )
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
        extra_body={"think": gunakan_thinking},
    )
    return llm.with_structured_output(HasilRAG)


# ---------------------------------------------------------------------------
# Utilitas format & metadata
# ---------------------------------------------------------------------------


def _dapatkan_metadata_artikel(
    meta: dict,
) -> tuple[str, str, str, str, str]:
    """Mengekstrak metadata artikel secara aman dengan penanganan format/fallback.

    Returns:
        Tuple berisi (artikel_id, judul, sumber, url, tanggal_publikasi).
    """
    judul = meta.get("judul", "Tanpa Judul")
    url = meta.get("url", "")

    # Memisahkan penulis/sumber dan tanggal dari field 'tanggal' (format CSV)
    tanggal_raw = meta.get("tanggal", "")
    if "," in tanggal_raw:
        sumber, tanggal_publikasi = [x.strip() for x in tanggal_raw.split(",", 1)]
    else:
        sumber = meta.get("sumber", "KlikDokter")
        tanggal_publikasi = meta.get("tanggal_publikasi", tanggal_raw)

    artikel_id = meta.get("artikel_id", url if url else judul)
    return artikel_id, judul, sumber, url, tanggal_publikasi


def kelompokkan_per_artikel(
    dokumen_list: list[Document],
) -> list[dict]:
    """Mengelompokkan chunk retrieval berdasarkan artikel asalnya.

    Hanya artikel yang memiliki URL yang disertakan — artikel tanpa URL
    dilewati agar tidak menjadi bahan jawaban dan tidak tampil di UI.

    Returns:
        List dict berisi metadata artikel dan gabungan teks chunk-nya,
        diurutkan berdasarkan tanggal publikasi terbaru lebih dulu.
    """
    kelompok: dict[str, list[Document]] = defaultdict(list)
    for doc in dokumen_list:
        artikel_id, *_ = _dapatkan_metadata_artikel(doc.metadata)
        kelompok[artikel_id].append(doc)

    artikel_list: list[dict] = []
    for artikel_id, chunks in kelompok.items():
        meta = chunks[0].metadata
        gabungan_teks = "\n".join(c.page_content for c in chunks)
        _, judul, sumber, url, tanggal_publikasi = _dapatkan_metadata_artikel(meta)

        if not url:
            logger.debug("Melewati artikel tanpa URL: %s", judul)
            continue

        artikel_list.append({
            "artikel_id": artikel_id,
            "judul": judul,
            "sumber": sumber,
            "url": url,
            "tanggal_publikasi": tanggal_publikasi,
            "teks": gabungan_teks,
        })

    # Urutkan artikel terbaru lebih dulu secara aman
    try:
        artikel_list.sort(key=lambda a: a["tanggal_publikasi"], reverse=True)
    except Exception:
        pass
    return artikel_list



def format_artikel_untuk_prompt(artikel_list: list[dict]) -> str:
    """Memformat list artikel menjadi teks bernomor untuk prompt LLM."""
    bagian: list[str] = []
    for i, a in enumerate(artikel_list, 1):
        bagian.append(
            f"[Artikel {i}] ID: {a['artikel_id']} | {a['judul']} "
            f"({a['sumber']}, {a['tanggal_publikasi']})\n{a['teks']}"
        )
    return "\n\n".join(bagian)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system", (
        "Anda adalah Asisten Informasi Kesehatan AI. Jawab pertanyaan HANYA berdasarkan konteks artikel kesehatan yang diberikan.\n\n"
        "PANDUAN UTAMA:\n"
        "- Bahasa: Seluruh jawaban WAJIB menggunakan bahasa Indonesia yang jelas, natural, profesional, dan mudah dipahami. "
        "Gunakan istilah medis dalam bahasa Indonesia dan sertakan istilah medis aslinya dalam kurung jika diperlukan.\n"
        "- Di Luar Domain: Tolak dengan sopan jika pertanyaan bukan tentang kesehatan/medis. "
        "Kosongkan `jawaban_terkait` dan set `perlu_rujukan_medis: false`.\n"
        "- Gejala Berat/Darurat: Sarankan segera ke IGD/faskes terdekat dan set `perlu_rujukan_medis: true`.\n"
        "- Batasan Medis: Dilarang membuat diagnosis pasti, menjanjikan kesembuhan, atau berasumsi di luar teks. "
        "Jika informasi tidak ada di konteks, nyatakan bahwa informasi tersebut tidak tersedia dalam basis data pengetahuan kesehatan kami.\n\n"
        "FORMAT OUTPUT:\n"
        "1. jawaban_utama: Jawaban terpadu dan naratif yang mengalir secara alami. "
        "Sintesis informasi dari seluruh konteks artikel secara kohesif. DILARANG KERAS menyebutkan frasa rujukan internal seperti 'Artikel 1', 'Artikel 2', 'Menurut artikel', atau sejenisnya di dalam `jawaban_utama`.\n"
        "2. jawaban_terkait: Ringkasan 1-2 kalimat untuk SETIAP artikel yang diberikan beserta judul/sumber aslinya.\n"
        "3. perlu_rujukan_medis: true / false."
    )),

    ("human", (
        "Konteks Artikel:\n{artikel_terformat}\n\n"
        "Pertanyaan: {question}"
    )),

])




# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------


def tanya(pertanyaan: str) -> HasilRAG:
    """Menjawab pertanyaan kesehatan menggunakan pipeline RAG.

    Alur:
    1. Retrieve chunk relevan dari Qdrant.
    2. Kelompokkan chunk berdasarkan artikel asal.
    3. Kirim ke LLM untuk menghasilkan jawaban terstruktur.
    4. Lengkapi metadata (url, sumber) dari data asli.

    Args:
        pertanyaan: Pertanyaan kesehatan dari pengguna.

    Returns:
        Objek ``HasilRAG`` berisi jawaban utama, jawaban per-artikel,
        dan flag ``perlu_rujukan_medis``.
    """
    dokumen_relevan = dapatkan_dokumen_relevan(pertanyaan)

    if not dokumen_relevan:
        logger.info("Tidak ditemukan dokumen yang memenuhi threshold relevansi.")
        return HasilRAG(
            jawaban_utama="Maaf, tidak ditemukan informasi yang relevan dalam basis data pengetahuan kesehatan kami untuk pertanyaan ini.",
            jawaban_terkait=[],
            perlu_rujukan_medis=False,
        )

    artikel_list = kelompokkan_per_artikel(dokumen_relevan)
    if not artikel_list:
        logger.info("Tidak ditemukan artikel yang relevan setelah pengelompokan.")
        return HasilRAG(
            jawaban_utama="Maaf, tidak ditemukan informasi yang relevan dalam basis data pengetahuan kesehatan kami untuk pertanyaan ini.",
            jawaban_terkait=[],
            perlu_rujukan_medis=False,
        )

    artikel_terformat = format_artikel_untuk_prompt(artikel_list)

    # Nyalakan thinking mode otomatis jika pertanyaan butuh menggabungkan
    # informasi dari lebih dari satu artikel (indikasi kasus multi-hop)
    perlu_thinking = len(artikel_list) > 1
    llm_terstruktur = _dapatkan_llm_terstruktur(gunakan_thinking=perlu_thinking)

    chain = PROMPT_RAG | llm_terstruktur
    hasil: HasilRAG = chain.invoke({
        "question": pertanyaan,
        "artikel_terformat": artikel_terformat,
    })

    # Bangun jawaban_terkait langsung dari artikel_list metadata asli
    # agar judul, URL, dan sumber 100% akurat dan tidak hilang karena LLM menyingkat judul
    hasil.jawaban_terkait = [
        JawabanTerkait(
            judul_artikel=a["judul"],
            ringkasan=a["teks"][:150] + "..." if len(a["teks"]) > 150 else a["teks"],
            sumber=a["sumber"],
            url=a["url"],
            tanggal_publikasi=a["tanggal_publikasi"],
        )
        for a in artikel_list
    ]

    return hasil




# ---------------------------------------------------------------------------
# Entrypoint untuk testing langsung
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Menjalankan uji coba tanya()...")
    hasil_uji = tanya("Apa gejala awal demam berdarah dengue?")
    print(f"Jawaban: {hasil_uji.jawaban_utama}")
    print(f"Artikel terkait: {len(hasil_uji.jawaban_terkait)}")
    print(f"Perlu rujukan medis: {hasil_uji.perlu_rujukan_medis}")
