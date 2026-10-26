from pydantic import BaseModel, Field
"""Schema Pydantic untuk output terstruktur dari RAG chain."""

class JawabanTerkait(BaseModel):
    """Ringkasan jawaban dari satu artikel sumber yang relevan."""

    judul_artikel: str = Field(
        description="Judul artikel sumber",
    )
    ringkasan: str = Field(
        description="Ringkasan jawaban 1-2 kalimat dari artikel ini",
    )
    sumber: str = Field(
        default="",
        description="Nama penerbit atau sumber artikel (misal: KlikDokter)",
    )
    url: str = Field(
        default="",
        description="URL lengkap menuju artikel asli",
    )
    tanggal_publikasi: str = Field(
        default="",
        description="Tanggal publikasi artikel",
    )


class HasilRAG(BaseModel):
    """Hasil lengkap dari pipeline RAG: jawaban utama, jawaban per-artikel, dan flag medis."""

    jawaban_utama: str = Field(
        description="Jawaban terpadu yang menggabungkan seluruh konteks artikel",
    )
    jawaban_terkait: list[JawabanTerkait] = Field(
        description="Daftar ringkasan jawaban per-artikel yang relevan",
    )
    perlu_rujukan_medis: bool = Field(
        description="True jika pertanyaan mengindikasikan kondisi berat/darurat",
    )

