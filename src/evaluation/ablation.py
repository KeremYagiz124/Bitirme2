"""Ablation çalışması, yöntem karşılaştırması ve parametre duyarlılık analizi.

Sentetik değerlendirme veri seti üzerinde:

1. Yöntem/konfigürasyon karşılaştırması: farklı detector ayarları (baseline,
   agresif, muhafazakâr) aynı veri setinde P/R/F1 ile karşılaştırılır.

2. Ablation: detector bileşenleri (kenar tespiti, temporal smoothing, lateral
   split) tek tek kapatılıp F1'e etkisi ölçülür.

3. Parametre duyarlılık analizi: tek bir parametre (ör. min_gap_ratio) taranıp
   F1'in nasıl değiştiği gösterilir.

Tüm sonuçlar CSV + grafik olarak kaydedilir.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation import metrics as M
from src.evaluation import plots as P
from src.evaluation import synthetic


def _eval_config(scenes, config: dict, iou_threshold: float = 0.4):
    """Verilen detector konfigürasyonunu sahnelerde değerlendir → DetectionMetrics."""
    from src.detection.street_parking_detector import StreetParkingDetector
    preds_per_image, gts_per_image = [], []
    for frame, dets, gt_empty in scenes:
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False, **config,
        )
        det.reset_history()
        result = det.analyze(frame, dets)
        preds_per_image.append([tuple(map(float, s)) for s in result["empty_spaces"]])
        gts_per_image.append([tuple(map(float, g)) for g in gt_empty])
    return M.evaluate_detections(preds_per_image, gts_per_image,
                                 iou_threshold=iou_threshold)


# ─── 1. Yöntem/konfigürasyon karşılaştırması ─────────────────────────────────

METHOD_CONFIGS = {
    "Baseline":      {"min_gap_ratio": 0.30, "max_edge_extension_ratio": 0.0,
                      "max_spaces_per_gap": 3},
    "Agresif":       {"min_gap_ratio": 0.15, "max_edge_extension_ratio": 0.20,
                      "max_spaces_per_gap": 4},
    "Muhafazakar":   {"min_gap_ratio": 0.50, "max_edge_extension_ratio": 0.0,
                      "max_spaces_per_gap": 2},
}


def run_method_comparison(scenes, out_dir):
    """Üç konfigürasyonu karşılaştır, CSV + grafik kaydet. Döner: dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cfg in METHOD_CONFIGS.items():
        m = _eval_config(scenes, cfg)
        results[name] = m.as_row()

    with open(out_dir / "method_comparison.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Yontem", "Precision", "Recall", "F1", "AP"])
        for name, row in results.items():
            w.writerow([name, row["Precision"], row["Recall"],
                        row["F1"], row["AP"]])

    P.plot_method_comparison(
        {n: {"Precision": r["Precision"], "Recall": r["Recall"],
             "F1": r["F1"], "AP": r["AP"]} for n, r in results.items()},
        ["Precision", "Recall", "F1", "AP"],
        out_dir / "method_comparison.png")
    return results


# ─── 2. Ablation (bileşen kapat-aç) ──────────────────────────────────────────

ABLATION_VARIANTS = {
    "Tam (hepsi acik)":     {"detect_row_edges": True,  "multi_row": True,
                             "max_edge_extension_ratio": 0.20},
    "Kenar tespiti KAPALI": {"detect_row_edges": False, "multi_row": True,
                             "max_edge_extension_ratio": 0.0},
    "Multi-row KAPALI":     {"detect_row_edges": True,  "multi_row": False,
                             "max_edge_extension_ratio": 0.20},
}


def run_ablation(scenes, out_dir):
    """Bileşenleri tek tek kapatıp F1 etkisini ölç. CSV + grafik. Döner: dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cfg in ABLATION_VARIANTS.items():
        m = _eval_config(scenes, cfg)
        results[name] = m.as_row()

    with open(out_dir / "ablation.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Varyant", "Precision", "Recall", "F1", "AP"])
        for name, row in results.items():
            w.writerow([name, row["Precision"], row["Recall"],
                        row["F1"], row["AP"]])

    P.plot_method_comparison(
        {n: {"F1": r["F1"], "Precision": r["Precision"], "Recall": r["Recall"]}
         for n, r in results.items()},
        ["Precision", "Recall", "F1"],
        out_dir / "ablation.png", title="Ablation — Bilesen Etkisi")
    return results


# ─── 3. Parametre duyarlılık analizi ─────────────────────────────────────────

def run_sensitivity(scenes, param_name, values, out_dir,
                    base_config=None):
    """Tek parametreyi tarayıp F1'in değişimini ölç. CSV + grafik. Döner: list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = dict(base_config or {})
    rows = []
    for v in values:
        cfg = dict(base)
        cfg[param_name] = v
        m = _eval_config(scenes, cfg)
        rows.append((v, m.precision, m.recall, m.f1))

    with open(out_dir / f"sensitivity_{param_name}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([param_name, "Precision", "Recall", "F1"])
        for v, p, r, fp in rows:
            w.writerow([v, round(p, 4), round(r, 4), round(fp, 4)])

    _plot_sensitivity(param_name, rows, out_dir / f"sensitivity_{param_name}.png")
    return rows


def _plot_sensitivity(param_name, rows, out_path):
    """F1/Precision/Recall'ı parametre değerine karşı çizgi grafikle çiz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r[0] for r in rows]
    prec = [r[1] for r in rows]
    rec = [r[2] for r in rows]
    f1 = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(xs, prec, "-o", color="#3b82f6", label="Precision")
    ax.plot(xs, rec, "-o", color="#22c55e", label="Recall")
    ax.plot(xs, f1, "-o", color="#eab308", label="F1")
    ax.set_xlabel(param_name); ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Parametre Duyarliligi: {param_name}")
    ax.tick_params(colors="#e2e8f0")
    for s in ax.spines.values():
        s.set_color("#1e293b")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#e2e8f0")
    leg = ax.legend(facecolor="#0f172a", edgecolor="#1e293b")
    for t in leg.get_texts():
        t.set_color("#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#0f172a")
    plt.close(fig)
    return str(out_path)


# ─── Tümünü çalıştır ─────────────────────────────────────────────────────────

def run_all(n_scenes=120, seed=42, out_dir="outputs/ablation"):
    """Karşılaştırma + ablation + duyarlılık analizini tek seferde çalıştır."""
    scenes = synthetic.make_dataset(n_scenes=n_scenes, seed=seed)
    comparison = run_method_comparison(scenes, out_dir)
    ablation = run_ablation(scenes, out_dir)
    sensitivity = run_sensitivity(
        scenes, "min_gap_ratio",
        [0.15, 0.25, 0.35, 0.45, 0.55, 0.65], out_dir,
        base_config={"max_edge_extension_ratio": 0.0})
    return {"comparison": comparison, "ablation": ablation,
            "sensitivity": sensitivity}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ablation + karşılaştırma + duyarlılık")
    ap.add_argument("--scenes", type=int, default=120)
    ap.add_argument("--out", default="outputs/ablation")
    args = ap.parse_args()
    res = run_all(n_scenes=args.scenes, out_dir=args.out)
    print("Ablation çalışması tamamlandı.")
    print("\nYöntem karşılaştırması:")
    for name, row in res["comparison"].items():
        print(f"  {name:14s} F1={row['F1']:.3f} P={row['Precision']:.3f} R={row['Recall']:.3f}")
    print("\nAblation:")
    for name, row in res["ablation"].items():
        print(f"  {name:24s} F1={row['F1']:.3f}")
    print(f"\nÇıktılar: {args.out}")
