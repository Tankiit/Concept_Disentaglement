"""Draw supplementary Figures 5–9 from the records shipped with the paper."""
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

STAGE = Path(__file__).resolve().parents[1]
OUT = STAGE / "figures"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.right": False, "axes.spines.top": False})
INK = "#182d38"
BLUE, ORANGE, PURPLE = "#0072B2", "#C65400", "#79549D"


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def action_values():
    path = STAGE / "workbook/informative_cell_verification_20261117/one_step.csv"
    with path.open() as f:
        values = {(r["world"], r["action"]): float(r["value"]) for r in csv.DictReader(f)}
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 1.98), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.973, top=.77, bottom=.26, wspace=.15)
    for ax, world, title, mechanism in zip(axes, ("M","L"),
            ("M   Uncertain source mixture","L   Uncertain label rates"),
            (r"$S\sim\mathrm{Bern}(p),\quad Y=S$",
             r"$S\sim\mathrm{Bern}(0.5),\quad Y\mid S,p\sim\mathrm{Bern}(p)$")):
        ax.text(0, 1.17, title, transform=ax.transAxes, weight="bold", color=INK, fontsize=9.6)
        ax.text(0, 1.015, mechanism, transform=ax.transAxes, color=INK, fontsize=8.25)
        for y, action, color in zip((2,1,0), ("identity","source0_label","cell"), (ORANGE,PURPLE,BLUE)):
            value = values[world, action]
            if value:
                ax.barh(y,value,height=.32,color=color,zorder=2)
            else:
                ax.scatter(0,y,s=17,facecolor="white",edgecolor=color,zorder=3,clip_on=False)
            ax.text(value+.001,y,f"{value:.4f}" if value else "0",va="center",
                    fontsize=8.4,weight="bold" if value==max(values[world,a] for a in ("identity","source0_label","cell")) else "normal")
        ax.set_xlim(0,.042)
        ax.set_ylim(-.52,2.56)
        ax.set_xticks([0,.01,.02,.03],["0",".01",".02",".03"])
        ax.set_yticks((2,1,0),["Source identity","Source-0 label","Visual cell"])
        ax.grid(axis="x",color="#e0e4e7",lw=.5)
        ax.tick_params(length=0,pad=4)
        ax.spines["left"].set_color("#909da3")
        ax.spines["bottom"].set_visible(False)
    fig.text(.55,.04,"Expected reduction in target variance after one field",ha="center",color=INK)
    save(fig,"supp5_world_actions")


BUDGETS = [0, 100, 1000, 10000, 100000]
ARMS = [
    ("inputs", "Deployment inputs", BLUE, "o", "-"),
    ("references", "Reference inputs", "#008566", "s", "--"),
    ("identities", "Source identities", ORANGE, "^", "-"),
    ("labels", "Target labels", PURPLE, "D", ":"),
]
REGIMES = [("image_derived", "Image-derived"), ("confounded", "Confounded")]
TARGETS = [("candidate_membership", "Candidate membership"),
           ("known_membership", "Known membership"), ("anchored_target", "Anchored label")]
DATASETS = [("mnist", "MNIST"), ("emnist", "EMNIST Letters"),
            ("fashion", "Fashion-MNIST"), ("blood", "BloodMNIST")]


def read(path):
    with (STAGE / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def arm_legend(fig, y):
    handles = [Line2D([], [], label=label, color=color, marker=marker, linestyle=style)
               for _, label, color, marker, style in ARMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, y),
               ncol=4, frameon=False, columnspacing=1.4, handlelength=2)


def width_panel(ax, rows):
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xticks(BUDGETS, ["0", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylim(-.025, 1.04)
    ax.tick_params(length=2.5, pad=2)
    ax.margins(x=.035)
    ax.grid(axis="y", color="#dce1e5", linewidth=.5)
    ax.set_axisbelow(True)
    alias = {"source_ids": "identities", "target_labels": "labels"}
    for arm, _, color, marker, style in ARMS:
        series = sorted((r for r in rows if alias.get(r["arm"], r["arm"]) == arm),
                        key=lambda r: int(r["budget"]))
        assert [int(r["budget"]) for r in series] == BUDGETS
        assert all(int(r.get("n", 6)) == 6 for r in series)
        low, mean, high = (np.array([float(r[k]) for r in series])
                           for k in ("min_width", "mean_width", "max_width"))
        assert np.all(low <= mean + 1e-12) and np.all(mean <= high + 1e-12)
        ax.fill_between(BUDGETS, low, high, color=color, alpha=.09, linewidth=0)
        ax.plot(BUDGETS, mean, color=color, marker=marker, linestyle=style)


def controlled_widths():
    rows = read("figure_data/controlled_summary.csv")
    for num, (regime, title) in enumerate(REGIMES, 6):
        fig, axes = plt.subplots(3, 4, figsize=(7, 4.85), sharex=True, sharey=True)
        fig.subplots_adjust(left=.108, right=.988, bottom=.105, top=.825, wspace=.15, hspace=.17)
        fig.text(.53, .975, f"{title} candidate law", ha="center", va="top",
                 weight="bold", fontsize=10.5, color=INK)
        arm_legend(fig, .931)
        for col, (dataset, name) in enumerate(DATASETS):
            for i, (target, label) in enumerate(TARGETS):
                ax = axes[i, col]
                width_panel(ax, [r for r in rows if (r["dataset"], r["regime"], r["target"])
                                 == (dataset, regime, target)])
                if i == 0:
                    ax.set_title(name, weight="bold")
                if col == 0:
                    ax.set_ylabel(label.replace(" ", "\n"))
        fig.supylabel("Interval width", x=.006, fontsize=8.5)
        fig.supxlabel("Added budget b", y=.015, fontsize=8.5)
        save(fig, f"supp{num}_{regime}_widths")


def tiny_widths():
    run = Path("TinyImageNet_REACH_Results_v2full")
    rows = read(run / "summary.csv")
    floors = {(r["regime"], r["target"]): float(r["width"])
              for r in read(run / "exact_identification_benchmarks.csv")}
    intervals = [r for r in read(run / "intervals.csv") if
                 (r["regime"], r["target"], r["arm"]) == ("confounded", "known_membership", "inputs")]
    for budget in BUDGETS:
        group = [r for r in intervals if int(r["budget"]) == budget]
        fallback = [r for r in group if r["status"] == "undefined_denominator_policy"]
        assert len({r["repeat"] for r in group}) == len(group) == 6 and len(fallback) == 5
        assert all((float(r["lower"]), float(r["upper"])) == (0, 1) for r in fallback)
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.986, top=.78, bottom=.13, wspace=.14, hspace=.18)
    arm_legend(fig, 1.0)
    exact = Line2D([], [], label="Exact input-only identification width", color="#424242",
                   linestyle="--", linewidth=1.0)
    fig.legend(handles=[exact], loc="upper center", bbox_to_anchor=(.52,.94), frameon=False)
    for row, (regime, regime_label) in enumerate(REGIMES):
        for col, (target, label) in enumerate([TARGETS[1], ("anchored_label","Anchored label"), TARGETS[0]]):
            ax = axes[row,col]
            width_panel(ax, [r for r in rows if (r["regime"], r["target"]) == (regime, target)])
            if target != "anchored_label":
                ax.axhline(floors[regime,target], color=exact.get_color(), linestyle="--", linewidth=1, zorder=1.6)
            if row == 0:
                ax.set_title(label, weight="bold")
            if col == 0:
                ax.set_ylabel(regime_label + "\ninterval width")
    axes[1,0].text(.5, .16, "Inputs: 5/6 use [0, 1]\nat every budget", transform=axes[1,0].transAxes,
                   ha="center", va="center", color=BLUE, fontsize=7)
    fig.supxlabel("Added budget b", y=.015, fontsize=8.5)
    save(fig, "supp8_tiny_widths")


def pair_advantages():
    pairs = read("analysis_outputs/pairwise_heldout_advantages.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.95))
    fig.subplots_adjust(left=.115, right=.984, bottom=.22, top=.86, wspace=.28)
    for ax, dataset in zip(axes, ("CAMELYON17", "FMoW")):
        rows = [r for r in pairs if r["dataset"] == dataset]
        x, y = (np.array([float(r[k])*1e6 for r in rows])
                for k in ("identity_advantage", "label_advantage"))
        both = (x > 0) & (y > 0)
        assert np.array_equal(both, [int(r["both_confirmed"]) for r in rows])
        selected, = [i for i,r in enumerate(rows) if int(r["selected"])]
        ax.scatter(x, y, s=18, color=BLUE, alpha=.68, linewidths=0, zorder=2)
        ax.scatter(x[selected], y[selected], s=90, color=INK, marker="*", zorder=4)
        ax.margins(x=.07, y=.12)
        xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
        ax.add_patch(Rectangle((0, 0), xmax, ymax, facecolor="#d7efe4", lw=0, zorder=0))
        ax.axhline(0, color="#727c81", lw=.7)
        ax.axvline(0, color="#727c81", lw=.7)
        ax.grid(color="#87959c", alpha=.23, lw=.45)
        ax.set_title(f"{dataset}   {both.sum()}/{len(rows)} confirm both", weight="bold")
        if dataset == "FMoW":
            ax.annotate(f"Selected ({x[selected]:.2f}, {y[selected]:.2f})",
                        (x[selected], y[selected]), xytext=(.34,.45), textcoords="axes fraction",
                        fontsize=7.5, ha="center", color=INK,
                        arrowprops=dict(arrowstyle="->", color=INK, lw=.7, shrinkB=5))
    axes[0].set_ylabel(r"Label-role advantage ($10^{-6}$ MSE)")
    fig.supxlabel(r"Identity-role advantage ($10^{-6}$ MSE)", y=.045, fontsize=8.5)
    save(fig, "supp9_pair_advantages")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
                         "axes.linewidth": .6, "lines.linewidth": 1.5, "lines.markersize": 3.2})
    action_values()
    controlled_widths()
    tiny_widths()
    pair_advantages()
