"""Plot the QaTa-COV19 computational-efficiency comparison."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = np.array(["LAVT", "LViT", "DMMI", "RefSegformer", "Attr-Mamba"])
GFLOPS = np.array([83.80, 54.10, 63.30, 103.60, 32.97])
FPS = np.array([5.87, 26.46, 14.84, 13.23, 26.88])
OURS = METHODS == "Attr-Mamba"

LABEL_OFFSETS = {
    "LAVT": (4, 4, "left", "bottom"),
    "LViT": (4, -5, "left", "top"),
    "DMMI": (4, 4, "left", "bottom"),
    "RefSegformer": (-4, 4, "right", "bottom"),
    "Attr-Mamba": (5, -5, "left", "top"),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.7,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def make_figure(output_dir: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.42, 2.28), dpi=200)

    baseline_color = "#637A8C"
    accent_color = "#B4494D"

    ax.scatter(
        GFLOPS[~OURS],
        FPS[~OURS],
        s=32,
        marker="o",
        facecolor=baseline_color,
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )
    ax.scatter(
        GFLOPS[OURS],
        FPS[OURS],
        s=38,
        marker="o",
        facecolor=accent_color,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )

    for method, x_value, y_value in zip(METHODS, GFLOPS, FPS):
        dx, dy, ha, va = LABEL_OFFSETS[method]
        is_ours = method == "Attr-Mamba"
        ax.annotate(
            method,
            xy=(x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=accent_color if is_ours else "#252A30",
            fontsize=6.5 if is_ours else 6.3,
            fontweight="bold" if is_ours else "normal",
            zorder=6,
        )

    ax.set_xlabel("Computational cost (GFLOPs)", labelpad=4)
    ax.set_ylabel("Throughput (FPS)", labelpad=4)
    ax.set_xlim(25, 110)
    ax.set_ylim(3, 30)
    ax.set_xticks([30, 50, 70, 90, 110])
    ax.set_yticks([5, 10, 15, 20, 25, 30])
    ax.grid(
        color="#D7DCE0",
        linestyle="-",
        linewidth=0.4,
        alpha=0.55,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#33383E")
    ax.spines["bottom"].set_color("#33383E")
    ax.tick_params(colors="#33383E", pad=2.2)
    fig.subplots_adjust(left=0.155, right=0.98, bottom=0.19, top=0.97)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "efficiency-tradeoff-qata-cov19"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    make_figure(repository_root / "assets" / "figures")
