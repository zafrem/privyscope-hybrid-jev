import pytest

from privyscope_hybrid_jev.artifact import JevMeta
from privyscope_hybrid_jev.calibrate import (CachingScorer, calibrate, config_to_yaml, evaluate, grid,
                                             micro_f1, split)
from privyscope_hybrid_jev.config import JevConfig
from privyscope_hybrid_jev.types import Span

LABELS = ["PER", "LOC"]


class Oracle:
    """PER for 'Alice', NONE otherwise; counts the items it actually receives."""

    meta = JevMeta("m", "rev", LABELS, 1.0)

    def __init__(self):
        self.seen = 0

    def probabilities(self, items):
        self.seen += len(items)
        return [{"PER": 0.98, "LOC": 0.01, "NONE": 0.01} if c == "Alice"
                else {"PER": 0.01, "LOC": 0.01, "NONE": 0.98} for _ctx, c in items]


def rec(i):
    text = f"Hi Alice and Table {i}."
    return text, [Span("PER", 3, 8)], [Span("PER", 3, 8), Span("PER", 13, 18)], []  # 2nd = false positive


def test_micro_f1_strict():
    m = micro_f1([([Span("A", 0, 1)], [Span("A", 0, 1), Span("A", 2, 3)])])
    assert (m["precision"], m["recall"]) == (0.5, 1.0)


def test_split_is_disjoint_interleaved_and_validates():
    recs = [rec(i) for i in range(10)]
    tune, held = split(recs, 0.5)
    assert len(tune) == len(held) == 5 and recs[0] in held and recs[1] in tune
    with pytest.raises(ValueError):
        split(recs, 1.0)


def test_cache_scores_each_distinct_item_once():
    inner = Oracle()
    c = CachingScorer(inner)
    c.probabilities([("x", "a"), ("x", "a"), ("x", "b")])
    c.probabilities([("x", "a")])
    assert inner.seen == 2 and c.misses == 2


def test_grid_sizes():
    base = JevConfig(model="x")
    assert len(grid(base)) == 4 * 2
    assert len(grid(base, modes=("verify", "extend"))) == 8 + 8 * 3


def test_calibrate_finds_config_that_beats_baseline():
    recs = [rec(i) for i in range(20)]
    inner = Oracle()
    rep = calibrate(recs, inner, grid(JevConfig(model="x", skip_verified=False)), 0.5)
    assert rep.baseline_held["f1"] < 1.0 and rep.best_held.metrics["f1"] == 1.0 and rep.helps
    assert inner.seen == rep.server_calls < 100   # cache: not configs x docs


def test_calibrate_reports_no_help_when_model_is_useless():
    class AlwaysNone(Oracle):
        def probabilities(self, items):
            return [{"PER": 0.0, "LOC": 0.0, "NONE": 1.0} for _ in items]

    rep = calibrate([rec(i) for i in range(20)], AlwaysNone(),
                    grid(JevConfig(model="x", skip_verified=False), none_thresholds=(0.5,), skip_verified=(False,)))
    assert not rep.helps


def test_failing_scorer_is_counted_and_keeps_baseline():
    class Down(Oracle):
        def probabilities(self, items):
            raise RuntimeError("down")

    cfg = JevConfig(model="x", skip_verified=False)
    res = evaluate(cfg, Down(), [rec(0), rec(1)])
    assert res.failures == 2
    assert res.metrics == micro_f1((g, m) for _t, g, m, _r in [rec(0), rec(1)])


def test_yaml_roundtrip(tmp_path):
    cfg = JevConfig(model="zafrem/x", mode="extend", none_threshold=0.7)
    p = tmp_path / "c.yaml"
    p.write_text(config_to_yaml(cfg))
    assert JevConfig.from_yaml(str(p)) == cfg
