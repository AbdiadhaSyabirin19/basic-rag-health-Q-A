"""Script Evaluasi RAGAS untuk Pipeline Basic RAG (Vector-only).

Mengevaluasi Basic RAG menggunakan dataset kasus uji pada folder
'test_cases/' (Test Case 1) atau file JSON Test Case 2.

Fitur Utama:
1. Membaca format Test Case 2 (hierarkis) maupun Test Case 1 (flat JSON).
2. Mendukung filter kategori: --category happy-path, edge-cases, dll.
3. Opsi pengujian cepat: --limit N (smoke test).
4. Menyimpan laporan komprehensif (JSON, CSV, Excel) ke 'hasil evaluasi/'.

Penggunaan CLI:
    # Evaluasi semua kategori (smoke test 3 sampel)
    python evaluasi/evaluasi_basic.py --limit 3

    # Evaluasi kategori tertentu
    python evaluasi/evaluasi_basic.py --category happy-path

    # Evaluasi dengan heuristic cepat (tanpa LLM RAGAS)
    python evaluasi/evaluasi_basic.py --fast --limit 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Pengaturan encoding konsol Windows agar karakter UTF-8 aman
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Memastikan root folder basic/ ada di sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Setup folder log
_log_dir = ROOT_DIR / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_dir / "evaluasi_basic.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("evaluasi_basic")

# Import komponen evaluasi
from src.eval_core import (
    _METRIC_COLS,
    _semua_nan,
    buat_evaluator_ragas,
    evaluasi_dengan_ragas,
    hitung_skor_heuristic_fallback,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loader Test Case
# ─────────────────────────────────────────────────────────────────────────────

def muat_test_case(
    file_path: str,
    kategori_filter: Optional[str] = None,
) -> tuple[dict, list[dict]]:
    """Memuat kasus uji dari file JSON (format flat atau hierarkis Test Case 2)."""
    path = Path(file_path)
    if not path.is_absolute():
        # Coba resolve relatif terhadap root project (satu level di atas basic/)
        project_root = ROOT_DIR.parent
        candidates = [
            ROOT_DIR / file_path,
            project_root / file_path,
            Path(file_path),
        ]
        for cand in candidates:
            if cand.is_file():
                path = cand
                break

    if not path.is_file():
        raise FileNotFoundError(f"File test case tidak ditemukan: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Deteksi format hierarkis (Test Case 2) vs flat (Test Case 1)
    if "test_cases" in data and isinstance(data["test_cases"], dict):
        # Format hierarkis Test Case 2
        meta = {
            "dataset_name": data.get("dataset_name", "Test Case Evaluasi"),
            "language": data.get("language", "id"),
        }
        raw_cases = data["test_cases"]
        daftar_kasus = []
        for kat, items in raw_cases.items():
            if kategori_filter and kategori_filter.lower() not in ("all", kat.lower()):
                continue
            for item in items:
                tc = item.copy()
                tc["category"] = kat
                tc["ground_truth"] = tc.get("expected_answer", tc.get("ground_truth", ""))
                daftar_kasus.append(tc)
    else:
        # Format flat Test Case 1 (list of dicts)
        meta = {"dataset_name": "Test Case 1"}
        items = data if isinstance(data, list) else data.get("test_cases", [])
        daftar_kasus = []
        for item in items:
            if kategori_filter and kategori_filter.lower() != "all":
                kat = item.get("category", item.get("type", ""))
                if kat.lower() != kategori_filter.lower():
                    continue
            daftar_kasus.append(item)

    logger.info(
        "Memuat %d kasus uji dari '%s' (filter: %s)",
        len(daftar_kasus), path.as_posix(), kategori_filter or "Semua",
    )
    return meta, daftar_kasus


# ─────────────────────────────────────────────────────────────────────────────
# 2. Eksekutor Basic RAG
# ─────────────────────────────────────────────────────────────────────────────

def buat_eksekutor_basic() -> Callable[[str], tuple[str, list[str], dict]]:
    """Menyiapkan runner eksekusi Basic RAG."""
    from src.basic_rag.rag_chain import dapatkan_dokumen_relevan, tanya as tanya_basic

    def eksekusi_basic_rag(pertanyaan: str) -> tuple[str, list[str], dict]:
        t0 = time.time()
        hasil = tanya_basic(pertanyaan)
        durasi = time.time() - t0

        dokumen_relevan = dapatkan_dokumen_relevan(pertanyaan)
        konteks = (
            [doc.page_content for doc in dokumen_relevan]
            if dokumen_relevan
            else ["Tidak ada konteks ditemukan."]
        )

        meta_info = {
            "durasi_detik": round(durasi, 2),
            "perlu_rujukan_medis": getattr(hasil, "perlu_rujukan_medis", False),
            "jumlah_sitasi": len(getattr(hasil, "jawaban_terkait", [])),
        }
        return hasil.jawaban_utama, konteks, meta_info

    return eksekusi_basic_rag


# ─────────────────────────────────────────────────────────────────────────────
# 3. Simpan Laporan
# ─────────────────────────────────────────────────────────────────────────────

def simpan_laporan(
    df_eval: pd.DataFrame,
    mode_name: str,
    meta_dataset: dict,
    output_dir: Path,
    durasi_total: float,
) -> dict:
    """Menyimpan laporan evaluasi dalam format JSON, CSV, dan Excel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = mode_name.lower().replace(" ", "_")

    file_csv   = output_dir / f"laporan_evaluasi_{slug}.csv"
    file_excel = output_dir / f"laporan_evaluasi_{slug}.xlsx"
    file_json  = output_dir / f"laporan_evaluasi_{slug}.json"

    print("\n" + "=" * 80)
    print(f"📊 RINGKASAN EVALUASI BASIC RAG - [{mode_name.upper()}]")
    print("=" * 80)

    metric_cols = [c for c in _METRIC_COLS if c in df_eval.columns]
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

    # CSV
    df_eval.to_csv(file_csv, index=False, encoding="utf-8-sig")
    print(f"💾 CSV  : {file_csv}")

    # Excel
    try:
        df_excel = df_eval.copy()
        if "contexts" in df_excel.columns:
            df_excel["contexts"] = df_excel["contexts"].apply(
                lambda c: "\n---\n".join(c) if isinstance(c, list) else str(c)
            )
        df_ringkasan = pd.DataFrame([
            {"Metrik / Info": "Pipeline",            "Nilai": mode_name},
            {"Metrik / Info": "Dataset",             "Nilai": meta_dataset.get("dataset_name", "-")},
            {"Metrik / Info": "Total Kasus",         "Nilai": len(df_eval)},
            {"Metrik / Info": "Total Durasi (detik)","Nilai": round(durasi_total, 2)},
            *[{"Metrik / Info": f"Rata-rata {m}", "Nilai": mean_scores[m]} for m in metric_cols],
        ])
        df_kat = pd.DataFrame()
        if "category" in df_eval.columns:
            df_kat = df_eval.groupby("category")[metric_cols].mean().round(4).reset_index()

        with pd.ExcelWriter(file_excel, engine="openpyxl") as writer:
            df_ringkasan.to_excel(writer, sheet_name="Ringkasan_Metrik", index=False)
            if not df_kat.empty:
                df_kat.to_excel(writer, sheet_name="Analisis_Kategori", index=False)
            df_excel.to_excel(writer, sheet_name="Rincian_Kasus", index=False)
        print(f"💾 Excel: {file_excel}")
    except Exception as err:
        logger.warning("Gagal menyimpan Excel: %s", err)

    # JSON
    def _safe(v):
        return None if isinstance(v, float) and v != v else v

    laporan = {
        "mode": mode_name,
        "dataset_name": meta_dataset.get("dataset_name"),
        "total_kasus": len(df_eval),
        "durasi_total_detik": round(durasi_total, 2),
        "skor_rata_rata": mean_scores,
        "rincian": [{k: _safe(v) for k, v in r.items()} for r in df_eval.to_dict(orient="records")],
    }
    with open(file_json, "w", encoding="utf-8") as f:
        json.dump(laporan, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON : {file_json}")
    print("=" * 80)
    return laporan


# ─────────────────────────────────────────────────────────────────────────────
# 4. Runner Utama
# ─────────────────────────────────────────────────────────────────────────────

def jalankan_evaluasi_basic(
    test_case_file: str,
    kategori_filter: Optional[str] = "all",
    limit: Optional[int] = None,
    evaluator_mode: str = "ragas",
    output_dir: str = "hasil evaluasi/basic",
) -> dict:
    """Menjalankan evaluasi lengkap Basic RAG."""
    t_start = time.time()
    mode_name = "Basic RAG"

    print("=" * 80)
    print(f"🚀 MEMULAI EVALUASI BASIC RAG")
    print(f"📁 Test Case : {test_case_file}")
    print(f"🔍 Kategori  : {kategori_filter or 'Semua'}")
    if limit:
        print(f"⚡ Limit     : {limit} soal")
    print("=" * 80)

    meta_dataset, daftar_kasus = muat_test_case(test_case_file, kategori_filter)
    if not daftar_kasus:
        logger.error("Tidak ada kasus uji ditemukan!")
        return {}

    if limit and limit > 0:
        daftar_kasus = daftar_kasus[:limit]

    total_kasus = len(daftar_kasus)
    print(f"📋 Total Kasus: {total_kasus}")

    eksekutor = buat_eksekutor_basic()
    dataset_records = []

    for idx, tc in enumerate(daftar_kasus, 1):
        tc_id = tc.get("id", f"TC-{idx:03d}")
        q = tc.get("question", "")
        kat = tc.get("category", tc.get("type", "general"))
        gt = tc.get("ground_truth") or tc.get("expected_answer", "")
        if not gt:
            kws = tc.get("expected_keywords", [])
            gt = f"Jawaban harus menjelaskan: {', '.join(kws)}." if kws else q

        print(f"[{idx}/{total_kasus}] [{tc_id}] [{kat}] \"{q[:60]}...\"")

        try:
            answer, contexts, meta = eksekutor(q)
            if not contexts:
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
            logger.error("   ❌ Gagal pada %s: %s", tc_id, err, exc_info=True)
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

    # Evaluasi Metrik
    if evaluator_mode == "heuristic":
        df_eval = hitung_skor_heuristic_fallback(dataset_records)
    else:
        evaluator_llm, evaluator_embeddings = buat_evaluator_ragas()
        if evaluator_llm is None or evaluator_embeddings is None:
            logger.warning("Evaluator tidak siap; menggunakan heuristic fallback.")
            df_eval = hitung_skor_heuristic_fallback(dataset_records)
        else:
            df_eval = evaluasi_dengan_ragas(
                dataset_records=dataset_records,
                evaluator_llm=evaluator_llm,
                evaluator_embeddings=evaluator_embeddings,
            )
            if _semua_nan(df_eval, _METRIC_COLS):
                logger.warning("Skor RAGAS NaN; beralih ke heuristic fallback.")
                df_eval = hitung_skor_heuristic_fallback(dataset_records)

    durasi_total = time.time() - t_start

    # Tentukan output dir (relative ke project root di atas basic/)
    out = Path(output_dir)
    if not out.is_absolute():
        out = ROOT_DIR.parent / output_dir

    return simpan_laporan(
        df_eval=df_eval,
        mode_name=mode_name,
        meta_dataset=meta_dataset,
        output_dir=out,
        durasi_total=durasi_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi Basic RAG Kesehatan (Ollama & RAGAS)"
    )
    parser.add_argument(
        "--test-case-file",
        default=str(ROOT_DIR / "test case" / "test_cases_qa_rag_kesehatan.json"),
        help="Path file JSON test case (default: basic/test case/test_cases_qa_rag_kesehatan.json)",
    )
    parser.add_argument(
        "--category",
        default="all",
        help="Filter kategori: all, happy-path, edge-cases, out-of-scope, adversarial",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah soal (contoh: --limit 3)",
    )
    parser.add_argument(
        "--evaluator",
        choices=["ragas", "heuristic"],
        default="ragas",
        help="Metode evaluasi: ragas (default) atau heuristic",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut --evaluator heuristic (tanpa LLM RAGAS)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "hasil evaluasi" / "basic"),
        help="Direktori output laporan evaluasi",
    )

    args = parser.parse_args()
    eval_mode = "heuristic" if args.fast else args.evaluator

    jalankan_evaluasi_basic(
        test_case_file=args.test_case_file,
        kategori_filter=args.category,
        limit=args.limit,
        evaluator_mode=eval_mode,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
