"""Draw Figure 9 from the supplementary pairwise CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "analysis_outputs" / "pairwise_heldout_advantages.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.45), layout="constrained")
    for ax, dataset in zip(axes, ("CAMELYON17", "FMoW")):
        rows = [row for row in pairs if row["dataset"] == dataset]
        x = [1e6 * float(row["identity_advantage"]) for row in rows]
        y = [1e6 * float(row["label_advantage"]) for row in rows]
        selected, = (i for i, row in enumerate(rows) if int(row["selected"]))
        both = sum(dx > 0 and dy > 0 for dx, dy in zip(x, y))
        ax.set_axisbelow(True)
        ax.grid(alpha=0.23)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.scatter(x, y, s=22, color="tab:blue", alpha=0.65)
        ax.scatter(x[selected], y[selected], s=100, marker="*", color="black",
                   label="Fit-only selected pair", zorder=4)
        ax.set_title(f"{dataset} ({len(rows)} qualifying pairs)", fontsize=11)
        ax.text(0.02, 0.97, f"{both}/{len(rows)} pairs in upper-right quadrant",
                ha="left", va="top", transform=ax.transAxes, fontsize=9)
        ax.margins(y=0.13)
        ax.set_xlabel(r"Identity-role advantage ($10^{-6}$ MSE)")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel(r"Label-role advantage ($10^{-6}$ MSE)")
    dest = ROOT / "figures" / "supp_pairwide_confirmation.pdf"
    dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest, metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
