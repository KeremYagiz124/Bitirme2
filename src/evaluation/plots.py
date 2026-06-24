"""Değerlendirme grafikleri — matplotlib ile PNG üretimi.

Ekransız (headless) ortamda çalışır: Agg backend kullanılır, grafikler
dosyaya kaydedilir. İçe aktarım anında backend ayarlanır.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ekransız: dosyaya çiz
import matplotlib.pyplot as plt
import numpy as np


_BG = "#0f172a"
_FG = "#e2e8f0"
_GRID = "#1e293b"


def _style(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


def plot_confusion_matrix(tp, fp, fn, tn, out_path,
                          title="Karisiklik Matrisi"):
    """2x2 karışıklık matrisini ısı haritası olarak kaydet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array([[tp, fp], [fn, tn]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor=_BG)
    im = ax.imshow(mat, cmap="Blues")
    labels = [["TP", "FP"], ["FN", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{int(mat[i, j])}",
                    ha="center", va="center", color="#0f172a",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pozitif", "Negatif"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Pozitif", "Negatif"])
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek")
    ax.set_title(title)
    _style(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    return str(out_path)


def plot_metric_bars(metrics: dict, out_path, title="Metrikler"):
    """Precision/Recall/F1/AP gibi metrikleri çubuk grafikle kaydet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(metrics.keys())
    values = [metrics[k] for k in names]
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 4),
                           facecolor=_BG)
    colors = ["#3b82f6", "#22c55e", "#eab308", "#ef4444", "#a855f7"]
    bars = ax.bar(names, values, color=colors[:len(names)])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", color=_FG, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    return str(out_path)


def plot_per_image_counts(predicted, expected, out_path,
                          title="Goruntu Bazinda Bos Slot"):
    """Görüntü başına tahmin vs beklenen boş slot sayısını çubuk grafikle çiz."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(predicted))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(5, len(predicted) * 0.6), 4),
                           facecolor=_BG)
    ax.bar(x - width / 2, expected, width, label="Beklenen", color="#22c55e")
    ax.bar(x + width / 2, predicted, width, label="Tahmin", color="#3b82f6")
    ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_xlabel("Goruntu"); ax.set_ylabel("Bos slot sayisi")
    ax.set_title(title)
    leg = ax.legend(facecolor=_BG, edgecolor=_GRID)
    for t in leg.get_texts():
        t.set_color(_FG)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    return str(out_path)


def plot_method_comparison(method_metrics: dict, metric_keys, out_path,
                           title="Yontem Karsilastirmasi"):
    """Birden çok yöntemin metriklerini gruplu çubuk grafikle karşılaştır.

    method_metrics: {yontem_adi: {metric: deger, ...}, ...}
    metric_keys: gösterilecek metrik adları (ör. ["Precision","Recall","F1"])
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(method_metrics.keys())
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(5, n_metrics * 1.6), 4.2),
                           facecolor=_BG)
    palette = ["#3b82f6", "#22c55e", "#eab308", "#ef4444", "#a855f7"]
    for mi, method in enumerate(methods):
        vals = [method_metrics[method].get(k, 0.0) for k in metric_keys]
        ax.bar(x + mi * width, vals, width, label=method,
               color=palette[mi % len(palette)])
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    leg = ax.legend(facecolor=_BG, edgecolor=_GRID)
    for t in leg.get_texts():
        t.set_color(_FG)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    return str(out_path)
