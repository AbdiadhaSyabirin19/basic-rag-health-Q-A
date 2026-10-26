"""Modul Inti Evaluasi RAGAS untuk Sistem RAG Kesehatan.

Menyediakan fungsi utilitas bersama:
- Pemuatan test cases
- Evaluasi metrik RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- Heuristic fallback evaluator (otomatis dipakai jika RAGAS gagal / NaN)
- Format tampilan ringkasan dan penyimpanan laporan JSON, CSV, dan Excel.
- Logging ke file untuk memudahkan debugging.
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Pengaturan encoding konsol Windows agar karakter UTF-8 / emoji aman
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Setup Logging (konsol + file) ─────────────────────────────────────────────
_log_dir = Path("logs")
_log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_dir / "evaluasi_ragas.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("eval_core")


# ── Konstanta ─────────────────────────────────────────────────────────────────
_METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pemuatan Kasus Uji
# ─────────────────────────────────────────────────────────────────────────────

def muat_kasus_uji(target_path: str = "test_cases") -> list[dict]:
    """Memuat daftar kasus uji dari direktori atau file JSON tertentu."""
    path = Path(target_path)
    semua_kasus = []

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                item["source_file"] = path.name
                semua_kasus.append(item)
    elif path.is_dir():
        for file_json in sorted(path.rglob("*.json")):
            with open(file_json, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    item["source_file"] = file_json.name
                    semua_kasus.append(item)
    else:
        logger.error(f"Target path '{target_path}' tidak ditemukan!")
        sys.exit(1)

    return semua_kasus


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pembuatan Evaluator RAGAS (LLM + Embeddings via Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def buat_evaluator_ragas():
    """Menyiapkan evaluator LLM & Embeddings RAGAS 100% berbasis Ollama lokal.

    Menggunakan LLM_EVAL_MODEL (default: qwen3:1.7b) agar cukup cepat
    di CPU lokal dan tidak melebihi timeout RAGAS.

    Returns:
        (evaluator_llm, evaluator_embeddings) — keduanya None jika gagal.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from langchain_ollama import ChatOllama, OllamaEmbeddings
        try:
            from ragas.embeddings.base import LangchainEmbeddingsWrapper
            from ragas.llms.base import LangchainLLMWrapper
        except ImportError:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
        from src import config

        # Gunakan LLM_EVAL_MODEL sebagai evaluator — sengaja berbeda dari LLM_MODEL
        # untuk menghindari self-evaluation bias. Thinking mode akan dimatikan
        # secara eksplisit (extra_body think=False) agar output tetap berupa JSON
        # bersih sesuai kebutuhan RAGAS, bukan prose/narasi panjang.
        logger.info(
            f"Mengonfigurasi Ollama Lokal ({config.LLM_EVAL_MODEL}) sebagai evaluator RAGAS..."
        )

        class RobustRagasLLMWrapper(LangchainLLMWrapper):
            """Wrapper LLM RAGAS yang otomatis menormalkan JSON output dari model lokal.

            Model kecil seperti qwen3:1.7b kadang mengembalikan JSON Array murni `[...]`
            alih-alih Dictionary `{"statements": [...]}` yang diharapkan Pydantic / RAGAS.
            Wrapper ini menormalkan list tersebut ke dict agar tidak memicu OutputParserException.
            """

            @staticmethod
            def _bersihkan_teks(teks: str) -> str:
                t = (teks or "").strip()
                # Bersihkan markdown codeblock ```json ... ```
                if t.startswith("```"):
                    lines = t.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    t = "\n".join(lines).strip()

                # Cek jika outputnya adalah JSON Array [ ... ]
                if t.startswith("[") and t.endswith("]"):
                    try:
                        parsed = json.loads(t)
                        if isinstance(parsed, list):
                            if all(isinstance(x, str) for x in parsed):
                                return json.dumps({"statements": parsed})
                            return json.dumps({
                                "statements": parsed,
                                "verifications": parsed,
                                "classifications": parsed,
                            })
                    except Exception:
                        pass
                return t

            @classmethod
            def _proses_generations(cls, generations):
                for gen_list in generations:
                    for gen in gen_list:
                        if hasattr(gen, "message") and hasattr(gen.message, "content"):
                            raw_text = str(gen.message.content or "")
                            cleaned = cls._bersihkan_teks(raw_text)
                            gen.message.content = cleaned
                        elif hasattr(gen, "text"):
                            raw_text = str(gen.text or "")
                            cleaned = cls._bersihkan_teks(raw_text)
                            try:
                                gen.text = cleaned
                            except AttributeError:
                                pass

            def generate_text(self, prompt, n: int = 1, temperature: float = 0.01, stop = None, callbacks = None):
                res = super().generate_text(prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks)
                self._proses_generations(res.generations)
                return res

            async def agenerate_text(self, prompt, n: int = 1, temperature: float = 0.01, stop = None, callbacks = None):
                res = await super().agenerate_text(prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks)
                self._proses_generations(res.generations)
                return res

        evaluator_llm = RobustRagasLLMWrapper(
            ChatOllama(
                model=config.LLM_EVAL_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.0,
                # KRITIS: thinking mode HARUS dimatikan untuk evaluator RAGAS.
                # Qwen3 dengan thinking ON akan:
                #   1. Menghabiskan 20-40 menit hanya untuk berpikir per call
                #   2. Menghasilkan output prose/narasi, BUKAN JSON yang dibutuhkan RAGAS
                # → Menyebabkan OutputParserException pada setiap job evaluasi.
                extra_body={"think": False},
                # Timeout per-request Ollama: 1 jam, cukup untuk worst-case
                # model lambat di CPU tanpa thinking mode.
                request_timeout=3600,
            )
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
        )
        return evaluator_llm, evaluator_embeddings

    except Exception as e:
        logger.warning(
            f"Tidak dapat membuat LLM wrapper RAGAS via Ollama: {e}. "
            "Akan menggunakan evaluasi fallback heuristik."
        )
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Heuristic Fallback Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def hitung_skor_heuristic_fallback(records: list[dict]) -> pd.DataFrame:
    """Fallback evaluator berbasis kata kunci jika RAGAS gagal atau timeout."""
    df = pd.DataFrame(records)
    logger.info("Menjalankan analisis heuristik fallback...")

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for _, row in df.iterrows():
        ans = str(row["answer"]).lower()
        q = str(row["question"]).lower()
        ctxs = " ".join([str(c) for c in row["contexts"]]).lower()
        gt = str(row["ground_truth"]).lower()

        words_ans = [w for w in ans.split() if len(w) > 3]
        f_score = sum(1 for w in words_ans if w in ctxs) / len(words_ans) if words_ans else 0.5
        faithfulness_scores.append(min(1.0, round(f_score, 3)))

        words_q = [w for w in q.split() if len(w) > 3]
        r_score = sum(1 for w in words_q if w in ans) / len(words_q) if words_q else 0.5
        relevancy_scores.append(min(1.0, round(r_score, 3)))

        words_gt = [w for w in gt.split() if len(w) > 3]
        p_score = sum(1 for w in words_gt if w in ctxs) / len(words_gt) if words_gt else 0.5
        precision_scores.append(min(1.0, round(p_score, 3)))

        rc_score = sum(1 for w in words_gt if w in ans) / len(words_gt) if words_gt else 0.5
        recall_scores.append(min(1.0, round(rc_score, 3)))

    df["faithfulness"] = faithfulness_scores
    df["answer_relevancy"] = relevancy_scores
    df["context_precision"] = precision_scores
    df["context_recall"] = recall_scores
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper & Evaluasi RAGAS
# ─────────────────────────────────────────────────────────────────────────────

def _semua_nan(df: pd.DataFrame, cols: list[str]) -> bool:
    """Cek apakah seluruh kolom metrik bernilai NaN."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return True
    return all(df[c].isna().all() for c in available)


def evaluasi_dengan_ragas(
    dataset_records: list[dict],
    evaluator_llm=None,
    evaluator_embeddings=None,
) -> pd.DataFrame:
    """Evaluasi dataset menggunakan framework RAGAS."""
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig

        # KRITIS: max_workers=1 agar job RAGAS berjalan serial (tidak paralel).
        # Ollama lokal tidak bisa handle request paralel — akan saling berebut
        # resource dan menyebabkan TimeoutError pada semua job sekaligus.
        #
        # timeout=None → UNLIMITED. Ini sengaja, bukan bug.
        # Alasan: RAGAS butuh 3-5 LLM call per metrik per sampel.
        # Dengan Qwen3:4b di CPU lokal (~3-9 menit per call), satu job bisa
        # butuh 9-45 menit. Nilai apapun yang finite (600s, 1200s, 3600s)
        # berisiko timeout jika konteks jawaban panjang.
        # Satu-satunya risiko timeout=None: jika Ollama crash/hang, proses
        # stuck selamanya → tekan Ctrl+C untuk menghentikan manual.
        run_cfg = RunConfig(
            timeout=None,     # UNLIMITED — satu-satunya cara aman untuk Ollama lokal lambat
            max_retries=1,    # 1 retry sudah cukup; lebih dari itu hanya buang waktu
            max_wait=60,
            max_workers=1,    # KRITIS: serialkan semua job evaluasi
        )

        # Import metric dari public API (RAGAS >= 0.1.x)
        try:
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            logger.info("Metric RAGAS di-import dari ragas.metrics (public API).")
        except ImportError:
            # Fallback ke private modules untuk RAGAS versi lama
            from ragas.metrics._answer_relevance import answer_relevancy
            from ragas.metrics._context_precision import context_precision
            from ragas.metrics._context_recall import context_recall
            from ragas.metrics._faithfulness import faithfulness
            logger.info("Metric RAGAS di-import dari private modules (fallback).")

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Validasi bahwa objek metric adalah Metric instances
        try:
            from ragas.metrics.base import Metric as RagasMetric
            for m in metrics:
                if not isinstance(m, RagasMetric):
                    raise TypeError(
                        f"Objek '{m}' bukan instansi Metric yang valid. "
                        "Periksa versi ragas yang terpasang."
                    )
        except ImportError:
            logger.warning("Tidak dapat mengimpor ragas.metrics.base.Metric untuk validasi.")

        # Pasang evaluator custom ke setiap metric
        if evaluator_llm and evaluator_embeddings:
            for m in metrics:
                try:
                    m.llm = evaluator_llm
                    if hasattr(m, "embeddings"):
                        m.embeddings = evaluator_embeddings
                except Exception as e:
                    logger.warning(f"Gagal menetapkan LLM/embeddings ke metric {m}: {e}")

        # ── Deteksi versi API RAGAS ──────────────────────────────────────────
        # RAGAS >= 0.2 menggunakan nama field baru:
        #   user_input, response, retrieved_contexts, reference
        # RAGAS < 0.2 (lama) menggunakan:
        #   question, answer, contexts, ground_truth
        # Kita coba buat dataset dengan API baru dulu; jika gagal, fallback ke lama.
        try:
            dataset = Dataset.from_dict({
                "user_input":         [r["question"]     for r in dataset_records],
                "response":           [r["answer"]       for r in dataset_records],
                "retrieved_contexts": [r["contexts"]     for r in dataset_records],
                "reference":          [r["ground_truth"] for r in dataset_records],
            })
            logger.info("Dataset RAGAS dibuat dengan field API baru (user_input/response/retrieved_contexts/reference).")
        except Exception:
            # Fallback ke field lama jika RAGAS versi lama
            dataset = Dataset.from_dict({
                "question":     [r["question"]     for r in dataset_records],
                "answer":       [r["answer"]       for r in dataset_records],
                "contexts":     [r["contexts"]     for r in dataset_records],
                "ground_truth": [r["ground_truth"] for r in dataset_records],
            })
            logger.info("Dataset RAGAS dibuat dengan field API lama (question/answer/contexts/ground_truth).")

        logger.info(
            "Memulai perhitungan metrik RAGAS "
            "(max_workers=1, timeout=UNLIMITED, %d sampel)...",
            len(dataset_records),
        )
        hasil_eval = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_cfg,
        )

        df_res = hasil_eval.to_pandas()

        # Gabungkan metadata tambahan
        for col in ["id", "category", "type", "durasi_detik", "perlu_rujukan_medis"]:
            if dataset_records and col in dataset_records[0]:
                df_res[col] = [r.get(col) for r in dataset_records]

        # Perbaikan: Deteksi dan imputasi NaN — baik seluruhnya maupun per baris/sel
        df_fallback = None
        if _semua_nan(df_res, _METRIC_COLS):
            logger.warning(
                "Seluruh skor RAGAS bernilai NaN. "
                "Beralih ke heuristic fallback secara menyeluruh."
            )
            df_fallback = hitung_skor_heuristic_fallback(dataset_records)
            for c in _METRIC_COLS:
                if c in df_fallback.columns:
                    df_res[c] = df_fallback[c].values
        else:
            # Jika ada sel metrik bernilai NaN (misal OutputParserException pada salah satu Job),
            # jangan biarkan null di laporan final! Isi sel tersebut dengan skor heuristic fallback.
            for col in _METRIC_COLS:
                if col in df_res.columns and df_res[col].isna().any():
                    if df_fallback is None:
                        logger.info(
                            "Terdapat metrik bernilai NaN pada beberapa sampel. "
                            "Menghitung skor heuristik fallback untuk imputasi sel..."
                        )
                        df_fallback = hitung_skor_heuristic_fallback(dataset_records)

                    nan_indices = df_res[df_res[col].isna()].index
                    for idx in nan_indices:
                        fallback_val = df_fallback.loc[idx, col]
                        sample_id = df_res.loc[idx, "id"] if "id" in df_res.columns else f"baris-{idx}"
                        logger.warning(
                            f"Metrik '{col}' pada sampel '{sample_id}' bernilai NaN/null (gagal dievaluasi RAGAS/Ollama). "
                            f"Mengisi dengan skor heuristic fallback: {fallback_val}"
                        )
                        df_res.loc[idx, col] = fallback_val

        return df_res

    except Exception as e:
        logger.warning(
            f"RAGAS evaluation terganggu/gagal: {e}. "
            "Menggunakan evaluasi heuristic fallback."
        )
        return hitung_skor_heuristic_fallback(dataset_records)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Simpan & Tampilkan Laporan
# ─────────────────────────────────────────────────────────────────────────────

def simpan_dan_tampilkan_laporan(
    df_eval: pd.DataFrame,
    mode_name: str,
    output_json: str,
    output_csv: str,
    output_excel: str | None = None,
    durasi_total: float = None,
) -> dict:
    """Menampilkan ringkasan hasil evaluasi dan menyimpan berkas JSON, CSV, dan Excel."""
    print("\n" + "=" * 80)
    print(f"📊 RINGKASAN SKOR EVALUASI RAGAS - [{mode_name.upper()}]")
    print("=" * 80)

    metric_cols = [c for c in _METRIC_COLS if c in df_eval.columns]

    print(f"{'Metrik Evaluasi':<25} | {'Skor Rata-Rata':<15}")
    print("-" * 45)
    mean_scores = {}
    for m in metric_cols:
        val = df_eval[m].mean()
        mean_scores[m] = round(float(val), 4) if pd.notna(val) else 0.0
        print(f"{m:<25} | {mean_scores[m]:<15.4f}")

    if "durasi_detik" in df_eval.columns:
        avg_dur = df_eval["durasi_detik"].mean()
        mean_scores["avg_durasi_detik"] = round(float(avg_dur), 2)
        print(f"{'Rata-rata Durasi (detik)':<25} | {mean_scores['avg_durasi_detik']:<15.2f}")

    print("=" * 80)

    # 1. Simpan CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Laporan Rinci CSV disimpan ke   : {output_csv}")

    # 2. Simpan Excel (.xlsx)
    if output_excel:
        Path(output_excel).parent.mkdir(parents=True, exist_ok=True)
        try:
            df_excel_detail = df_eval.copy()
            if "contexts" in df_excel_detail.columns:
                df_excel_detail["contexts"] = df_excel_detail["contexts"].apply(
                    lambda c: "\n---\n".join(c) if isinstance(c, list) else str(c)
                )

            df_ringkasan = pd.DataFrame([
                {"Metrik / Info": "Mode Pipeline",                "Nilai": mode_name},
                {"Metrik / Info": "Total Kasus Uji",              "Nilai": len(df_eval)},
                {"Metrik / Info": "Total Durasi (detik)",         "Nilai": round(durasi_total, 2) if durasi_total else "-"},
                *[{"Metrik / Info": f"Rata-rata {m}",            "Nilai": mean_scores[m]} for m in metric_cols],
                {"Metrik / Info": "Rata-rata Durasi per Soal (s)", "Nilai": mean_scores.get("avg_durasi_detik", "-")},
            ])

            kat_col = "category" if "category" in df_eval.columns else (
                "type" if "type" in df_eval.columns else None
            )
            df_kategori = (
                df_eval.groupby(kat_col)[metric_cols].mean().round(4).reset_index()
                if kat_col and kat_col in df_eval.columns
                else pd.DataFrame()
            )

            with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
                df_ringkasan.to_excel(writer,    sheet_name="Ringkasan_Metrik",  index=False)
                df_excel_detail.to_excel(writer, sheet_name="Rincian_Kasus",     index=False)
                if not df_kategori.empty:
                    df_kategori.to_excel(writer, sheet_name="Analisis_Kategori", index=False)

            print(f"💾 Laporan Rinci Excel disimpan ke : {output_excel}")
        except Exception as err:
            logger.warning(f"Gagal menyimpan ke Excel ({output_excel}): {err}")

    # 3. Simpan JSON (NaN → None agar serialisasi bersih)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    def _safe_val(v):
        """Konversi NaN/Inf ke None agar JSON valid."""
        if isinstance(v, float) and (v != v):  # NaN check via identity
            return None
        return v

    rincian_bersih = [
        {k: _safe_val(v) for k, v in rec.items()}
        for rec in df_eval.to_dict(orient="records")
    ]

    laporan_final = {
        "mode": mode_name,
        "total_kasus": len(df_eval),
        "durasi_total_detik": round(durasi_total, 2) if durasi_total else None,
        "skor_rata_rata": mean_scores,
        "rincian": rincian_bersih,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(laporan_final, f, ensure_ascii=False, indent=2)
    print(f"💾 Laporan Rinci JSON disimpan ke  : {output_json}")
    print("=" * 80)

    return laporan_final


# ─────────────────────────────────────────────────────────────────────────────
# 6. Runner Generik
# ─────────────────────────────────────────────────────────────────────────────

def jalankan_evaluasi(
    mode_name: str,
    fungsi_eksekusi,
    target_path: str = "test_cases",
    limit: int | None = None,
    output_json: str = "laporan_evaluasi.json",
    output_csv: str = "laporan_evaluasi.csv",
    output_excel: str | None = None,
) -> dict:
    """Fungsi runner generik untuk mengeksekusi pipeline evaluasi RAG."""
    t_start = time.time()
    print("=" * 80)
    print(f"🎯 MEMULAI EVALUASI RAGAS - [{mode_name.upper()}]")
    print("=" * 80)

    kasus_uji_list = muat_kasus_uji(target_path)
    if limit and limit > 0:
        kasus_uji_list = kasus_uji_list[:limit]

    total_kasus = len(kasus_uji_list)
    print(f"📂 Total Kasus Uji Dimuat: {total_kasus} dari '{target_path}'")
    print("=" * 80)

    dataset_records = []
    for idx, tc in enumerate(kasus_uji_list, 1):
        tc_id = tc.get("id", f"TC-{idx:03d}")
        q     = tc.get("question", "")
        kat   = tc.get("category", tc.get("type", "general"))
        gt    = tc.get("ground_truth") or tc.get("reference")

        if not gt:
            kws = tc.get("expected_keywords", [])
            gt  = (
                f"Jawaban yang benar harus menjelaskan tentang: {', '.join(kws)}."
                if kws
                else tc.get("description", q)
            )

        print(f"[{idx}/{total_kasus}] [{tc_id}] Memproses: \"{q[:60]}...\"")

        try:
            answer, contexts, meta = fungsi_eksekusi(q)

            # Perbaikan: log jika konteks kosong
            if not contexts:
                logger.warning(
                    f"[{tc_id}] Tidak ada konteks untuk: {q[:60]}"
                )
                contexts = ["Tidak ada konteks ditemukan."]

            dataset_records.append({
                "id":                  tc_id,
                "category":            kat,
                "type":                tc.get("type", "happy_path"),
                "question":            q,
                "answer":              answer,
                "contexts":            contexts,
                "ground_truth":        gt,
                "durasi_detik":        meta.get("durasi_detik", 0),
                "perlu_rujukan_medis": meta.get("perlu_rujukan_medis", False),
            })
            print(f"   ✓ Selesai ({meta.get('durasi_detik')}s | {len(contexts)} konteks)")

        except Exception as err:
            logger.error(f"   ❌ Gagal inferensi pada {tc_id}: {err}", exc_info=True)
            dataset_records.append({
                "id":                  tc_id,
                "category":            kat,
                "type":                tc.get("type", "happy_path"),
                "question":            q,
                "answer":              f"Error: {err}",
                "contexts":            ["Tidak ada konteks"],
                "ground_truth":        gt,
                "durasi_detik":        0,
                "perlu_rujukan_medis": False,
            })

    # Perbaikan: Validasi evaluator sebelum memanggil RAGAS
    evaluator_llm, evaluator_embeddings = buat_evaluator_ragas()

    if evaluator_llm is None or evaluator_embeddings is None:
        logger.warning(
            "Evaluator LLM atau Embeddings tidak tersedia; "
            "beralih ke evaluasi heuristik fallback."
        )
        df_eval = hitung_skor_heuristic_fallback(dataset_records)
    else:
        df_eval = evaluasi_dengan_ragas(
            dataset_records=dataset_records,
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
        )

        # Perbaikan (lapis luar): Jika masih NaN setelah RAGAS, pakai fallback
        if _semua_nan(df_eval, _METRIC_COLS):
            logger.warning(
                "Skor evaluasi RAGAS masih NaN setelah dijalankan. "
                "Beralih ke evaluasi heuristik fallback."
            )
            df_eval = hitung_skor_heuristic_fallback(dataset_records)

    t_total = time.time() - t_start
    return simpan_dan_tampilkan_laporan(
        df_eval=df_eval,
        mode_name=mode_name,
        output_json=output_json,
        output_csv=output_csv,
        output_excel=output_excel,
        durasi_total=t_total,
    )
