import numpy as np

from qmix_aiot.experiments import METRICS
from qmix_aiot.regime_study import REGIMES, summarize


def row(method, train_seed, eval_seed, dcr, checkpoint=110):
    record = {
        "method": method, "train_seed": train_seed, "eval_seed": eval_seed,
        "checkpoint": checkpoint, "regime": "steady_012", "scenario": "stable",
        "arrival_per_device": 0.12,
    }
    return {**record, **{metric: dcr if metric == "dcr" else 1.0 for metric in METRICS}}


def test_learned_summary_aggregates_held_out_episodes_by_model():
    rows = [row("adapted_qmix", 100, 1, 20), row("adapted_qmix", 100, 2, 40),
            row("adapted_qmix", 101, 1, 50), row("adapted_qmix", 101, 2, 70)]
    result = summarize(rows)[0]
    assert result["evaluation_records"] == 4
    assert result["independent_units"] == 2
    assert result["unit_kind"] == "trained_model"
    assert np.isclose(result["dcr_mean"], 45)
    assert np.isclose(result["dcr_sd"], np.sqrt(450))


def test_heuristic_summary_uses_independent_task_streams():
    rows = [row("greedy", -1, 2000, 20, checkpoint=0),
            row("greedy", -1, 2001, 60, checkpoint=0)]
    result = summarize(rows)[0]
    assert result["unit_kind"] == "held_out_stream"
    assert result["independent_units"] == 2
    assert np.isclose(result["dcr_mean"], 40)


def test_regimes_span_underloaded_and_near_capacity_conditions():
    stable_rates = [x.arrival for x in REGIMES if x.scenario == "stable"]
    assert stable_rates == [0.06, 0.12, 0.24, 0.36]
    assert 50 * stable_rates[0] * 44.5 / (25 * 32.5) < 0.2
    assert 50 * stable_rates[-1] * 44.5 / (25 * 32.5) > 0.98
