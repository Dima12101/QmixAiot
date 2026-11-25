"""Create publication figures using only measured CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(path: Path):
    with path.open(encoding="utf-8", newline="") as src:
        records = list(csv.DictReader(src))
    if not records:
        raise ValueError(f"No observations in {path}")
    return records


def create_figures(result_path: str = "results", output_path: str = "figs"):
    result_dir, output_dir = Path(result_path), Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_records(result_dir / "summary.csv")
    train = load_records(result_dir / "training_history.csv")
    scaling = load_records(result_dir / "inference_scaling.csv")
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .22})
    methods = ["adapted_qmix", "vanilla_qmix", "iql", "greedy", "round_robin"]
    labels = {"adapted_qmix": "Адаптированный QMIX", "vanilla_qmix": "Исходный QMIX",
              "iql": "Independent DQN", "greedy": "Greedy", "round_robin": "Round Robin",
              "no_gru": "Без GRU", "no_masks": "Без маски допустимости",
              "no_fairness": "Без штрафа неравномерности", "no_energy": "Без суррогата затрат"}
    scenarios = ["stable", "peaks", "failures", "heterogeneous"]
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = .15
    for i, method in enumerate(methods):
        rows = [next(r for r in summary if r["method"] == method and r["scenario"] == sc) for sc in scenarios]
        heights = [float(r["dcr_mean"]) for r in rows]
        errs = [float(r["dcr_std"]) for r in rows]
        ax.bar(x + (i - 2) * width, heights, width, label=labels[method], yerr=errs, capsize=2)
    ax.set_xticks(x, ["Стабильная", "Пики нагрузки", "Отказы узлов", "Гетерогенность"])
    ax.set_ylabel("Задачи, завершённые в срок, %")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "01_deadline_comparison.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ablations = ["adapted_qmix", "no_gru", "no_masks", "no_fairness", "no_energy"]
    for axis, metric, title in zip(axes, ("dcr", "jain"), ("Соблюдение сроков, %", "Нормированный индекс Джайна")):
        vals = [next(float(r[f"{metric}_mean"]) for r in summary
                     if r["method"] == m and r["scenario"] == "heterogeneous") for m in ablations]
        errs = [next(float(r[f"{metric}_std"]) for r in summary
                     if r["method"] == m and r["scenario"] == "heterogeneous") for m in ablations]
        axis.barh(np.arange(len(ablations)), vals, xerr=errs, capsize=3)
        axis.set_yticks(np.arange(len(ablations)), [labels[m] for m in ablations])
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.set_xlim(0, 100 if metric == "dcr" else 1)
    fig.tight_layout()
    fig.savefig(output_dir / "02_heterogeneity_ablation.png", dpi=240)
    plt.close(fig)

    groups = defaultdict(list)
    for r in train:
        if r["method"] == "adapted_qmix":
            groups[int(r["episode"])].append(float(r["reward"]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ep = sorted(groups)
    avg = np.array([np.mean(groups[i]) for i in ep])
    std = np.array([np.std(groups[i], ddof=1) if len(groups[i]) > 1 else 0 for i in ep])
    curve = np.convolve(avg, np.ones(5) / 5, mode="valid") if len(ep) >= 5 else avg
    position = np.asarray(ep[4:] if len(ep) >= 5 else ep)
    ax.plot(position, curve, label="Адаптированный QMIX, среднее по 5 эпизодам")
    ax.fill_between(np.asarray(ep), avg - std, avg + std, alpha=.17, label="Разброс по начальным условиям")
    ax.set_xlabel("Эпизод обучения")
    ax.set_ylabel("Накопленное вознаграждение за эпизод")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_training_rewards.png", dpi=240)
    plt.close(fig)

    x = np.array([int(r["nodes"]) for r in scaling])
    y = np.array([float(r["median_ms"]) for r in scaling])
    p95 = np.array([float(r["p95_ms"]) for r in scaling])
    fitted_exponent = float(np.polyfit(np.log(x), np.log(y), 1)[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(x, y, "o-", label="Медиана")
    ax.plot(x, p95, "s--", label="95-й процентиль")
    ax.set_xlabel("Число периферийных агентов")
    ax.set_ylabel("Время выбора действий, мс")
    ax.set_title(f"Время исполнения; наклон логарифмической регрессии {fitted_exponent:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_inference_latency.png", dpi=240)
    plt.close(fig)
    return {"count": 4, "fitted_inference_exponent": fitted_exponent}


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figs")
    args = p.parse_args()
    print(create_figures(args.results, args.figures))


if __name__ == "__main__":
    cli()
