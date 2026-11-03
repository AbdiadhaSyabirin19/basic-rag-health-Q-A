"""RAG chain untuk tanya-jawab kesehatan berbasis artikel.

Modul ini menyediakan pipeline retrieval-augmented generation (RAG)
yang mengambil konteks dari Qdrant vector database dan menghasilkan
jawaban terstruktur menggunakan LLM lokal via Ollama.
"""

import functools
import logging
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from src import config
from src.basic_rag.schema import HasilRAG, JawabanTerkait, ResponLLM

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
        client_kwargs={"timeout": 300.0},
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
            "score_threshold": score_threshold,
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


@functools.lru_cache(maxsize=1)
def _dapatkan_llm():
    """Mengembalikan instance LLM ChatOllama cepat dengan pembatasan token output agar tidak looping."""
    logger.info("Menginisialisasi ChatOllama: model=%s", config.LLM_MODEL)
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        repeat_penalty=1.15,
        num_ctx=4096,      # kembalikan ke 4096 agar konteks retrieval cukup lega
        num_predict=400,   # batasi max 400 token agar selesai dalam ~20-30 detik di CPU
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "\n\n\n\n",
            "Selamat mencoba!",
            "KlikDokter.com -",
        ],
        request_timeout=120.0,
    )


def _deteksi_perlu_rujukan_medis(teks_jawaban: str, pertanyaan: str) -> bool:
    """Mendeteksi apakah pertanyaan/jawaban mengindikasikan kondisi darurat medis."""
    teks_gabungan = f"{teks_jawaban} {pertanyaan}".lower()
    kata_kunci_darurat = [
        "igd", "gawat darurat", "segera ke dokter", "segera hubungi dokter",
        "segera periksakan", "pertolongan medis darurat", "nyeri dada akut",
        "serangan jantung", "stroke", "pendarahan hebat", "kehilangan kesadaran",
        "sesak napas parah", "darurat medis", "kondisi kritis", "faskes terdekat",
        "rumah sakit terdekat", "ambulans",
    ]
    return any(kata in teks_gabungan for kata in kata_kunci_darurat)


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
    """Mengelompokkan chunk retrieval berdasarkan artikel asalnya."""
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
# Pemeriksaan Relevansi & Kecukupan Konteks (Prioritas 2 - Abstention)
# ---------------------------------------------------------------------------


def _periksa_kecukupan_konteks(pertanyaan: str, dokumen_list: list[Document]) -> tuple[bool, str]:
    """Memeriksa apakah konteks hasil retrieval mencukupi untuk menjawab pertanyaan.

    Menerapkan alur ideal:
    1. Pengguna mengajukan pertanyaan.
    2. Sistem mencari artikel kesehatan.
    3. Sistem memeriksa relevansi konteks.
    4. Jika konteks mencukupi, generator menyusun jawaban.
    5. Jika konteks tidak mencukupi, sistem menyampaikan bahwa informasi belum tersedia.

    Returns:
        (cukup: bool, pesan_abstention: str)
    """
    if not dokumen_list:
        return False, "Tidak ditemukan artikel yang relevan dalam basis pengetahuan."

    stopwords = {
        "apa", "saja", "bagaimana", "mengapa", "kenapa", "apakah", "yang", "di", "ke",
        "dari", "pada", "untuk", "dan", "atau", "adalah", "ini", "itu", "adakah", "bisa",
        "bolehkah", "kita", "tidak", "setelah", "seorang", "mengalami", "penderita",
        "terkena", "dapat", "dilakukan", "benar", "bagi", "area", "timbul", "cara",
        "hal", "karena", "agar", "tetap", "saat", "oleh", "terkait", "seputar"
    }

    words = [re.sub(r"[^\w]", "", w.lower()) for w in pertanyaan.split()]
    kata_kunci = [w for w in words if len(w) > 2 and w not in stopwords]

    if not kata_kunci:
        return True, ""

    gabungan_konteks = " ".join(doc.page_content.lower() for doc in dokumen_list)

    # Deteksi entitas subjek medis spesifik (misal: insulin, tetanus, rabies, dll.)
    # Jika subjek inti tidak ada sama sekali di seluruh konteks, konteks tidak mencukupi (contoh: HP003)
    for subjek in ["insulin", "rabies", "tetanus", "hiv", "covid", "tbc", "tuberkulosis"]:
        if subjek in kata_kunci and subjek not in gabungan_konteks:
            return False, f"Informasi spesifik mengenai {subjek} belum tersedia dalam basis data pengetahuan kesehatan kami saat ini."

    # Hitung rasio kemunculan kata kunci pertanyaan di konteks
    kata_cocok = [k for k in kata_kunci if k in gabungan_konteks]
    rasio_cocok = len(kata_cocok) / len(kata_kunci) if kata_kunci else 1.0

    if rasio_cocok < 0.25 and len(kata_cocok) == 0:
        return False, "Informasi spesifik untuk menjawab pertanyaan ini belum tersedia secara memadai dalam basis data pengetahuan kesehatan kami."

    return True, ""


# ---------------------------------------------------------------------------
# Prompt template (Prioritas 3 - Peningkatan Kualitas Generator)
# ---------------------------------------------------------------------------

PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system", (
        "Anda adalah Asisten Informasi Kesehatan AI berbasis bukti ilmiah yang bertugas menjawab pertanyaan kesehatan.\n"
        "Jawab pertanyaan pengguna berdasarkan fakta medis yang termuat dalam konteks artikel kesehatan yang diberikan.\n\n"
        "PEDOMAN KUALITAS JAWABAN (WAJIB DIPATUHI):\n"
        "1. SINTESIS MAKSIMAL KONTEKS:\n"
        "   - Jelaskan informasi, penyebab, faktor risiko, maupun penanganan yang relevan sebagaimana diuraikan dalam artikel terlampir.\n"
        "   - Jika artikel membahas penyebab atau pemicu secara spesifik (misalnya batu ginjal, penurunan fungsi organ, komplikasi penyakit kronis), jelaskan hal tersebut secara jelas dan terstruktur.\n"
        "2. BAHASA NATURAL & PROFESIONAL:\n"
        "   - Rumuskan jawaban dengan kata-kata sendiri yang runut, profesional, dan mudah dipahami.\n"
        "   - DILARANG menyalin teks promosi aplikasi/web, nama dokter/tokoh berita, atau frasa rujukan seperti 'Artikel 1' dan 'berdasarkan teks di atas'.\n"
        "3. KETEPATAN FAKTUAL & BEBAS HALUSINASI:\n"
        "   - Jangan menambahkan resep obat keras atau klaim medis yang bertentangan dengan konteks.\n"
        "4. ATURAN PENGAKUAN KETIDAKSEDIAAN KONTEKS (ABSTENTION):\n"
        "   - HANYA nyatakan informasi belum tersedia jika seluruh artikel yang diberikan benar-benar sama sekali tidak membahas topik penyakit/kondisi yang ditanyakan.\n"
        "5. KESELAMATAN PENGGUNA:\n"
        "   - Untuk kondisi darurat/kritis, ingatkan untuk mencari pertolongan medis ke IGD/faskes terdekat.\n\n"
        "Berikan jawaban langsung berupa penjelasan medis yang informatif, padat, dan rapi dalam bahasa Indonesia."
    )),

    ("human", (
        "Konteks Artikel Kesehatan:\n{artikel_terformat}\n\n"
        "Pertanyaan Pengguna: {question}"
    )),
])


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------


def tanya(pertanyaan: str) -> HasilRAG:
    """Menjawab pertanyaan kesehatan menggunakan pipeline RAG.

    Alur:
    1. Pengguna mengajukan pertanyaan.
    2. Sistem mencari artikel kesehatan di Qdrant (retrieval).
    3. Sistem memeriksa relevansi & kecukupan konteks (abstention check).
    4. Jika konteks mencukupi, generator LLM menyusun jawaban informatif.
    5. Jika konteks tidak mencukupi, sistem menyampaikan informasi belum tersedia.

    Args:
        pertanyaan: Pertanyaan kesehatan dari pengguna.

    Returns:
        Objek ``HasilRAG`` berisi jawaban utama, jawaban per-artikel,
        flag ``perlu_rujukan_medis``, dan teks ``konteks``.
    """
    dokumen_relevan = dapatkan_dokumen_relevan(pertanyaan)

    if not dokumen_relevan:
        logger.info("Tidak ditemukan dokumen yang memenuhi threshold relevansi.")
        return HasilRAG(
            jawaban_utama="Maaf, tidak ditemukan informasi yang relevan dalam basis data pengetahuan kesehatan kami untuk pertanyaan ini.",
            jawaban_terkait=[],
            perlu_rujukan_medis=False,
            konteks=[],
        )

    # Prioritas 2: Periksa relevansi dan kecukupan konteks sebelum memanggil generator
    cukup, pesan_abstention = _periksa_kecukupan_konteks(pertanyaan, dokumen_relevan)
    if not cukup:
        logger.info("Konteks tidak mencukupi untuk menjawab pertanyaan: %s", pesan_abstention)
        return HasilRAG(
            jawaban_utama=f"{pesan_abstention} Silakan berkonsultasi langsung dengan dokter atau tenaga medis profesional untuk petunjuk lebih lanjut.",
            jawaban_terkait=[],
            perlu_rujukan_medis=False,
            konteks=[doc.page_content for doc in dokumen_relevan],
        )

    artikel_list = kelompokkan_per_artikel(dokumen_relevan)
    if not artikel_list:
        logger.info("Tidak ditemukan artikel yang relevan setelah pengelompokan.")
        return HasilRAG(
            jawaban_utama="Maaf, tidak ditemukan informasi yang relevan dalam basis data pengetahuan kesehatan kami untuk pertanyaan ini.",
            jawaban_terkait=[],
            perlu_rujukan_medis=False,
            konteks=[doc.page_content for doc in dokumen_relevan],
        )

    artikel_terformat = format_artikel_untuk_prompt(artikel_list)

    # Eksekusi generator LLM langsung teks tanpa GBNF grammar JSON yang membuat CPU macet
    llm = _dapatkan_llm()
    chain = PROMPT_RAG | llm | StrOutputParser()

    teks_mentah: str = chain.invoke({
        "question": pertanyaan,
        "artikel_terformat": artikel_terformat,
    })
    logger.info("LLM teks mentah (panjang=%d): %r", len(teks_mentah), teks_mentah[:300])

    jawaban_bersih = teks_mentah.strip()

    # Bersihkan jika LLM membalut jawaban dengan tag markdown atau format json yang tidak sengaja ter-generate
    if jawaban_bersih.startswith("```json"):
        jawaban_bersih = re.sub(r"^```json\s*", "", jawaban_bersih)
        jawaban_bersih = re.sub(r"\s*```$", "", jawaban_bersih).strip()
        try:
            import json
            data_json = json.loads(jawaban_bersih)
            if isinstance(data_json, dict) and "jawaban_utama" in data_json:
                jawaban_bersih = str(data_json["jawaban_utama"]).strip()
        except Exception:
            pass
    elif jawaban_bersih.startswith("```"):
        jawaban_bersih = re.sub(r"^```[a-zA-Z]*\s*", "", jawaban_bersih)
        jawaban_bersih = re.sub(r"\s*```$", "", jawaban_bersih).strip()

    # Post-processing jika output LLM kosong atau degeneratif terpotong di tengah kalimat
    if not jawaban_bersih:
        logger.warning("LLM mengembalikan output kosong. Mengambil intisari dari artikel terkait...")
        cuplikan = [a.get("teks", "")[:250].strip() for a in artikel_list[:2] if a.get("teks")]
        jawaban_bersih = (
            "Berdasarkan artikel kesehatan terkait: " + " ".join(cuplikan)
            if cuplikan else
            "Informasi spesifik untuk menjawab pertanyaan ini belum tersedia secara memadai."
        )
    elif len(jawaban_bersih.split()) <= 6 and not any(jawaban_bersih.endswith(p) for p in [".", "!", "?"]):
        jawaban_bersih = (
            "Maaf, informasi spesifik mengenai pertanyaan tersebut belum tersedia secara memadai "
            "dalam basis data pengetahuan kesehatan kami saat ini. Silakan berkonsultasi langsung dengan dokter "
            "atau tenaga medis profesional untuk petunjuk yang tepat."
        )

    # Deteksi status perlu rujukan medis
    perlu_rujukan = _deteksi_perlu_rujukan_medis(jawaban_bersih, pertanyaan)

    # Bangun jawaban_terkait langsung dari artikel_list metadata asli
    jawaban_terkait = [
        JawabanTerkait(
            judul_artikel=a["judul"],
            sumber=a["sumber"],
            url=a["url"],
            tanggal_publikasi=a["tanggal_publikasi"],
        )
        for a in artikel_list
    ]

    return HasilRAG(
        jawaban_utama=jawaban_bersih,
        jawaban_terkait=jawaban_terkait,
        perlu_rujukan_medis=perlu_rujukan,
        konteks=[doc.page_content for doc in dokumen_relevan],
    )


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
