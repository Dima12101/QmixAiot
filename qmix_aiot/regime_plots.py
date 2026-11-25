"""Publication figures for the controlled regime study (CSV sources only)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .regime_study import REGIMES


LABELS = {
    "adapted_qmix": "QMIX адапт.", "no_public": "QMIX без опов.",
    "vanilla_qmix": "QMIX исходн.", "iql": "IQL",
    "greedy": "Greedy", "round_robin": "Round Robin",
}


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _pair_diffs(rows: list[dict], a: str, b: str, regime: str, checkpoint: int):
    groups = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if (x["regime"] == regime and x["method"] in {a, b}
                and int(x["checkpoint"]) == checkpoint):
            groups[int(x["train_seed"])][x["method"]].append(float(x["dcr"]))
    diffs = [np.mean(g[a]) - np.mean(g[b]) for g in groups.values() if g[a] and g[b]]
    if len(diffs) < 2:
        raise ValueError("Need at least two paired training replicates")
    return float(np.mean(diffs)), float(np.std(diffs, ddof=1))


def create_figures(result_path: str = "results_regime2026", output_path: str = "figs_regime"):
    input_dir, output_dir = Path(result_path), Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read(input_dir / "summary.csv")
    raw = _read(input_dir / "raw_evaluation.csv")
    records = {(r["method"], int(r["checkpoint"]), r["regime"]): r for r in summary}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0,
        "axes.labelsize": 9, "legend.fontsize": 7.5,
        "axes.grid": True, "grid.alpha": 0.18,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    methods = ("adapted_qmix", "vanilla_qmix", "iql", "greedy", "round_robin")
    styles = [
        ("#202020", "o", "-"), ("#656565", "s", "--"),
        ("#999999", "^", "-."), ("#333333", "D", ":"),
        ("#aaaaaa", "v", "-"),
    ]
    x = np.array([0.06, 0.12, 0.24, 0.36])
    fig, ax = plt.subplots(figsize=(3.35, 3.55))
    for method, style in zip(methods, styles):
        checkpoint = 0 if method in {"greedy", "round_robin"} else 110
        selected = [records[method, checkpoint, f"steady_{int(rate * 100):03d}"] for rate in x]
        y = [float(r["dcr_mean"]) for r in selected]
        errors = [float(r["dcr_sd"]) for r in selected]
        ax.errorbar(x, y, yerr=errors, label=LABELS[method], color=style[0],
                    marker=style[1], linestyle=style[2], linewidth=1.1,
                    elinewidth=.7, capsize=2, markersize=3.6)
    ax.set(xlabel="Поступления на устройство, 1/с", ylabel="DCR, %", ylim=(0, 103))
    ax.set_xticks(x, ["0,06", "0,12", "0,24", "0,36"])
    ax.legend(loc="lower left", frameon=False, ncol=1)
    fig.tight_layout(pad=.7)
    fig.savefig(output_dir / "01_load_response.png", dpi=300)
    plt.close(fig)

    ablation_regimes = (
        ("steady_012", "Стаб. 0,12"), ("steady_024", "Стаб. 0,24"),
        ("failure_024", "Отказы 0,24"), ("hetero_024", "Неодн. 0,24"),
    )
    values = [_pair_diffs(raw, "adapted_qmix", "no_public", key, 110)
              for key, _ in ablation_regimes]
    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    y = np.arange(len(values))
    ax.axvline(0, color="black", linewidth=.8)
    ax.barh(y, [v[0] for v in values], xerr=[v[1] for v in values],
            color="white", edgecolor="black", hatch="///", capsize=2)
    ax.set_yticks(y, [name for _, name in ablation_regimes])
    ax.invert_yaxis()
    ax.set_xlabel("DCR: с объявлениями − без, п.п.")
    ax.xaxis.grid(True)
    fig.tight_layout(pad=.7)
    fig.savefig(output_dir / "02_information_ablation.png", dpi=300)
    plt.close(fig)

    checkpoints = (55, 110)
    # A focused subset keeps checkpoint comparisons legible in one 82.5-mm
    # column. The remaining component ablations are reported in Table 4.
    methods = ("adapted_qmix", "no_public", "vanilla_qmix", "iql")
    width = 0.33
    pos = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(3.35, 3.10))
    for j, checkpoint in enumerate(checkpoints):
        selected = [records[m, checkpoint, "steady_024"] for m in methods]
        ax.bar(pos + (j - .5) * width, [float(r["dcr_mean"]) for r in selected],
               width, yerr=[float(r["dcr_sd"]) for r in selected],
               label=f"{checkpoint} эп.", color="white" if j == 0 else "#777777",
               edgecolor="#222222", hatch="///" if j == 0 else "", capsize=2)
    ax.set_ylim(0, 103)
    ax.set_ylabel("DCR, %")
    ax.set_xticks(pos, ["Адапт.", "Без\nопов.", "Исходн.", "IQL"])
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=.6)
    fig.savefig(output_dir / "03_training_budget.png", dpi=300)
    plt.close(fig)
    return {"figures": 3, "records": len(raw), "regimes": [r.name for r in REGIMES]}


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_regime2026")
    parser.add_argument("--figures", default="figs_regime")
    a = parser.parse_args()
    print(create_figures(a.results, a.figures))


if __name__ == "__main__":
    cli()
