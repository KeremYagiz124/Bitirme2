"""
CSV log'undan doluluk zaman grafiği.

Kullanım:
    python scripts/plot_log.py --log outputs/metrics/log_20260421_155805.csv
    python scripts/plot_log.py --log outputs/metrics/log_20260421_155805.csv --save
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",  required=True, help="CSV log dosyası")
    ap.add_argument("--save", action="store_true", help="PNG olarak kaydet")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"HATA: Dosya bulunamadı: {args.log}")
        sys.exit(1)

    df = pd.read_csv(args.log)
    if "timestamp" not in df.columns:
        print("HATA: CSV'de 'timestamp' sütunu yok.")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")

    # Üst: dolu / boş slot
    if "occupied" in df.columns and "available" in df.columns:
        axes[0].plot(df["timestamp"], df["occupied"],  color="#3c3cdc", lw=2, label="Dolu Slot")
        axes[0].plot(df["timestamp"], df["available"], color="#00dc50", lw=2, label="Boş Slot")
        axes[0].fill_between(df["timestamp"], df["occupied"],  alpha=0.15, color="#3c3cdc")
        axes[0].fill_between(df["timestamp"], df["available"], alpha=0.15, color="#00dc50")
        axes[0].set_ylabel("Slot Sayısı", color="#e2e8f0")
        axes[0].set_title("Park Doluluk Durumu", color="#e2e8f0", pad=8)
        axes[0].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    else:
        axes[0].set_title("occupied/available sütunu bulunamadı", color="#e2e8f0")

    # Alt: toplam araç
    if "vehicles" in df.columns:
        axes[1].plot(df["timestamp"], df["vehicles"], color="#f59e0b", lw=2, label="Araç")
        axes[1].fill_between(df["timestamp"], df["vehicles"], alpha=0.15, color="#f59e0b")
        axes[1].set_ylabel("Araç Sayısı", color="#e2e8f0")
        axes[1].set_title("Tespit Edilen Araç Sayısı", color="#e2e8f0", pad=8)
        axes[1].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    plt.tight_layout(pad=2)

    if args.save:
        out_dir = Path("outputs/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (log_path.stem + "_plot.png")
        fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
        print(f"Grafik kaydedildi: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
