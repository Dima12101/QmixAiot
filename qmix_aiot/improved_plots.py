"""Three publication figures computed from the new test records only."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "structured_qmix": "QMIX с оценкой узлов",
    "work_conserving_qmix": "QMIX с обязательным назначением",
    "adapted_qmix": "QMIX с общим выходом",
    "vanilla_qmix": "QMIX базовый",
    "iql": "IQL", "greedy": "Greedy", "round_robin": "Round Robin",
    "structured_no_public": "Без объявлений",
    "structured_optional_reject": "С добровольным отказом",
}


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def paired_diff(rows, a, b, regime, checkpoint):
    groups = defaultdict(lambda: defaultdict(list))
    for record in rows:
        if (record["regime"] == regime and record["method"] in {a, b}
                and int(record["checkpoint"]) == checkpoint):
            groups[int(record["train_seed"])][record["method"]].append(float(record["dcr"]))
    vals = [np.mean(g[a]) - np.mean(g[b]) for g in groups.values() if g[a] and g[b]]
    if len(vals) < 2:
        raise ValueError(f"Need at least two matched training seeds for {a} vs {b}")
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def create(results: str = "results_improved2026", figures: str = "figs_improved"):
    root, dest = Path(results), Path(figures)
    dest.mkdir(parents=True, exist_ok=True)
    summary = _read(root / "summary.csv")
    raw = _read(root / "raw_evaluation.csv")
    s = {(r["method"], int(r["checkpoint"]), r["regime"]): r for r in summary}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "axes.grid": True, "grid.alpha": .16,
        "legend.fontsize": 6.7, "savefig.facecolor": "white",
    })
    methods = ("structured_qmix", "work_conserving_qmix", "adapted_qmix", "greedy", "round_robin")
    styles = [
        ("#111111", "o", "-"), ("#666666", "s", "--"),
        ("#aaaaaa", "^", "-."), ("#333333", "D", ":"),
        ("#555555", "v", "-"),
    ]
    xx = np.array([.06, .12, .24, .36])
    fig, ax = plt.subplots(figsize=(3.31, 3.40))
    for method, style in zip(methods, styles):
        checkpoint = 0 if method in {"greedy", "round_robin"} else 110
        rows = [s[method, checkpoint, f"steady_{int(x * 100):03d}"] for x in xx]
        ax.errorbar(xx, [float(r["dcr_mean"]) for r in rows],
                    yerr=[float(r["dcr_sd"]) for r in rows],
                    color=style[0], marker=style[1], linestyle=style[2],
                    linewidth=1.1, elinewidth=.7, capsize=2, markersize=3.4,
                    label=LABELS[method])
    ax.set(xlabel="Поступления на устройство, 1/с", ylabel="DCR, %", ylim=(0, 103))
    ax.set_xticks(xx, ["0,06", "0,12", "0,24", "0,36"])
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(pad=.6)
    fig.savefig(dest / "01_load_response.png", dpi=320)
    plt.close(fig)

    ablations = (
        ("structured_optional_reject", "Разрешён отказ"),
        ("structured_no_public", "Без объявлений"),
        ("work_conserving_qmix", "Общий выход"),
    )
    values = [paired_diff(raw, "structured_qmix", method, "steady_024", 110)
              for method, _ in ablations]
    fig, ax = plt.subplots(figsize=(3.31, 2.80))
    positions = np.arange(len(values))
    ax.axvline(0, color="black", linewidth=.7)
    ax.barh(positions, [value[0] for value in values],
            xerr=[value[1] for value in values],
            color="white", edgecolor="#222222", hatch="///", capsize=2)
    ax.set_yticks(positions, [label for _, label in ablations])
    ax.invert_yaxis()
    ax.set_xlabel("Разность DCR, п.п.")
    fig.tight_layout(pad=.6)
    fig.savefig(dest / "02_component_ablation.png", dpi=320)
    plt.close(fig)

    checkpoints = (55, 110)
    methods = ("structured_qmix", "structured_no_public",
               "structured_optional_reject", "work_conserving_qmix")
    fig, ax = plt.subplots(figsize=(3.31, 3.0))
    positions = np.arange(len(methods))
    for idx, checkpoint in enumerate(checkpoints):
        rec = [s[method, checkpoint, "steady_024"] for method in methods]
        ax.bar(positions + (idx - .5) * .35,
               [float(r["dcr_mean"]) for r in rec], .35,
               yerr=[float(r["dcr_sd"]) for r in rec],
               color="white" if idx == 0 else "#777777",
               edgecolor="#333333", hatch="///" if idx == 0 else "",
               label=f"{checkpoint} эп.", capsize=2)
    ax.set(ylim=(0, 103), ylabel="DCR, %")
    ax.set_xticks(positions, ["Структ.", "Без\nобъявл.", "С отказом", "Общий\nвыход"])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=.6)
    fig.savefig(dest / "03_training_budget.png", dpi=320)
    plt.close(fig)
    return {"figures": 3, "evaluation_records": len(raw)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_improved2026")
    p.add_argument("--figures", default="figs_improved")
    args = p.parse_args()
    print(create(args.results, args.figures))
