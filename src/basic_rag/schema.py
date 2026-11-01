from pydantic import BaseModel, Field
"""Schema Pydantic untuk output terstruktur dari RAG chain."""

class JawabanTerkait(BaseModel):
    """Referensi artikel sumber yang relevan."""

    judul_artikel: str = Field(
        description="Judul artikel sumber",
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


class ResponLLM(BaseModel):
    """Output langsung dari LLM yang ringkas dan cepat (hanya jawaban utama & flag rujukan medis)."""

    jawaban_utama: str = Field(
        description="Jawaban terpadu yang menggabungkan seluruh konteks artikel",
    )
    perlu_rujukan_medis: bool = Field(
        description="True jika pertanyaan mengindikasikan kondisi berat/darurat",
    )


class HasilRAG(BaseModel):
    """Hasil lengkap dari pipeline RAG: jawaban utama, jawaban per-artikel, flag medis, dan teks konteks."""

    jawaban_utama: str = Field(
        description="Jawaban terpadu yang menggabungkan seluruh konteks artikel",
    )
    jawaban_terkait: list[JawabanTerkait] = Field(
        default_factory=list,
        description="Daftar ringkasan jawaban per-artikel yang relevan",
    )
    perlu_rujukan_medis: bool = Field(
        description="True jika pertanyaan mengindikasikan kondisi berat/darurat",
    )
    konteks: list[str] = Field(
        default_factory=list,
        description="Daftar teks chunk konteks yang diambil saat retrieval",
    )

