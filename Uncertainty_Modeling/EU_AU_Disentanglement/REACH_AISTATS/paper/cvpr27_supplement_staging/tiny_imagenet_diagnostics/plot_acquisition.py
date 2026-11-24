"""Draw the Tiny ImageNet acquisition panels in the controlled-study figure style."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, NullLocator
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "TinyImageNet_REACH_Results_v2full"
STYLES = {
    "inputs": ("Deployment inputs", "#0072B2", "o", "-"),
    "references": ("Reference inputs", "#009E73", "s", "--"),
    "identities": ("Source identities", "#D55E00", "^", "-"),
    "labels": ("Target labels", "#8B59A3", "D", ":"),
}
REGIMES = [("image_derived", "Image-derived"), ("confounded", "Confounded")]
TARGETS = [("known_membership", "Known membership"), ("anchored_label", "Anchored label"),
           ("candidate_membership", "Candidate membership")]
BUDGETS = [0, 100, 1000, 10000, 100000]


def main():
    summary = pd.read_csv(RESULTS / "summary.csv").set_index(["regime", "target", "arm"])
    floors = pd.read_csv(RESULTS / "exact_identification_benchmarks.csv").set_index(["regime", "target"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.5,
        "lines.markersize": 3.2, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=.096, right=.985, top=.785, bottom=.15, wspace=.13, hspace=.20)
    handles = [Line2D([], [], label=name, color=color, marker=marker, linestyle=style)
               for name, color, marker, style in STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.002),
               ncol=4, frameon=False, columnspacing=1.3, handlelength=2)
    exact = Line2D([], [], label="Exact input-only identification width", color="#424242",
                   linestyle="--", linewidth=1.2)
    fig.legend(handles=[exact], loc="upper center", bbox_to_anchor=(.5, .946),
               frameon=False, handlelength=2)
    for row, (regime, regime_label) in enumerate(REGIMES):
        for col, (target, target_label) in enumerate(TARGETS):
            ax = axes[row, col]
            ax.grid(axis="y", color="#dce1e5", linewidth=.5)
            ax.set_xscale("symlog", linthresh=100)
            ax.set_xticks(BUDGETS, ["0", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_ylim(-.025, 1.04)
            ax.tick_params(length=2.5, pad=2)
            ax.margins(x=.035)
            for arm, (_, color, marker, style) in STYLES.items():
                values = summary.loc[(regime, target, arm)].sort_values("budget")
                assert values.budget.tolist() == BUDGETS and values.n.eq(6).all()
                ax.fill_between(values.budget, values.min_width, values.max_width,
                                color=color, alpha=.07, linewidth=0)
                ax.plot(values.budget, values.mean_width, color=color, marker=marker, linestyle=style)
            if target != "anchored_label":
                ax.axhline(floors.loc[(regime, target), "width"], color=exact.get_color(),
                           linestyle=exact.get_linestyle(), linewidth=exact.get_linewidth(), zorder=1)
            if row == 0:
                ax.set_title(target_label, weight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"{regime_label}\ninterval width")
    fig.supxlabel("Added budget b (references and labels: b per known source)", y=.025, fontsize=8.5)
    for extension in ["pdf", "png"]:
        fig.savefig(PAPER / "figures" / f"supp_tiny_imagenet_acquisition.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
