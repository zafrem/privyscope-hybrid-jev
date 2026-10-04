import pytest

from privyscope_hybrid_jev.artifact import JevMeta
from privyscope_hybrid_jev.config import JevConfig
from privyscope_hybrid_jev.stage import JevStage, _DefaultStage
from privyscope_hybrid_jev.types import Span

LABELS = ["PER", "LOC"]
TEXT = "Contact Alice Kim at Main Street 5. Ask Bob."


class FakeScorer:
    """Answers by candidate text: {candidate: label}; anything else is NONE."""

    def __init__(self, answers, labels=LABELS):
        self.answers, self.meta = answers, JevMeta("m", "rev", list(labels), 1.0)
        self.items = []

    def probabilities(self, items):
        self.items += list(items)
        out = []
        for _ctx, cand in items:
            hit = self.answers.get(cand, "NONE")
            out.append({n: (0.98 if n == hit else 0.01) for n in list(self.meta.labels) + ["NONE"]})
        return out


def make(answers, **cfg):
    sc = FakeScorer(answers)
    return JevStage(JevConfig(model="x", **cfg), sc), sc


def test_verify_drops_false_positive_and_relabels():
    stage, _ = make({"Alice Kim": "PER", "Street 5": "PER"}, skip_verified=False)
    spans = [Span("LOC", 8, 17), Span("PER", 21, 25), Span("LOC", 26, 34)]
    assert stage.refine(TEXT, spans, []) == [Span("PER", 8, 17), Span("PER", 26, 34)]


def test_verify_never_adds_spans():
    stage, sc = make({"Bob": "PER"})
    assert stage.refine(TEXT, [], []) == [] and sc.items == []


def test_skip_verified_bypasses_the_model():
    s = Span("PER", 8, 17)
    stage, sc = make({}, skip_verified=True)
    assert stage.refine(TEXT, [s], [s]) == [s] and sc.items == []
    stage2, sc2 = make({}, skip_verified=False)
    assert stage2.refine(TEXT, [s], [s]) == [] and len(sc2.items) == 1


def test_extend_adds_span_respects_cap_and_prefers_longest_tie():
    stage, _ = make({"Bob": "PER"}, mode="extend", max_ngram=1)
    assert stage.refine(TEXT, [], []) == [Span("PER", 40, 43)]
    stage, _ = make({"Alice": "PER", "Alice Kim": "PER", "Kim": "PER"}, mode="extend", max_ngram=2)
    existing = Span("LOC", 26, 34)
    out = stage.refine(TEXT, [existing], [existing])
    assert existing in out and [s for s in out if s.label == "PER"] == [Span("PER", 8, 17)]
    capped, sc = make({}, mode="extend", max_ngram=2, max_candidates=3)
    capped.refine(TEXT, [], [])
    assert len(sc.items) == 3


def test_artifact_with_labels_the_core_does_not_know_is_rejected():
    with pytest.raises(ValueError, match="unknown label"):
        JevStage(JevConfig(model="x"), FakeScorer({}, labels=["PER", "NAME"]))


def test_unconfigured_default_stage_is_a_noop_and_loads_nothing(monkeypatch):
    for v in ("PRIVYSCOPE_JEV_MODEL", "PRIVYSCOPE_JEV_CONFIG"):
        monkeypatch.delenv(v, raising=False)
    s = [Span("PER", 0, 3)]
    assert _DefaultStage().refine("abc", s, []) == s


def test_importing_and_running_unconfigured_stage_never_imports_torch():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['privyscope'] = None  # block the core: it may pull torch itself\n"
        "from privyscope_hybrid_jev.stage import default_stage\n"
        "from privyscope_hybrid_jev.types import Span\n"
        "s = [Span('PER', 0, 3)]\n"
        "assert default_stage.refine('abc', s, []) == s\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("PRIVYSCOPE_JEV")}
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_spans_with_labels_the_head_was_not_trained_on_are_left_alone():
    """Packs emit URL/IP/CRYPTO/...; the head only knows the base codes, so it must not judge them."""
    stage, sc = make({}, skip_verified=False)  # this scorer would answer NONE for everything
    url = Span("URL", 4, 25)
    assert stage.refine("see https://example.com/x now", [url], []) == [url]
    assert sc.items == []


def test_failed_load_is_attempted_once_and_stays_a_noop(monkeypatch):
    import privyscope_hybrid_jev.stage as st

    calls = []

    class Boom:
        def __init__(self, cfg):
            calls.append(1)
            raise RuntimeError("no network")

    monkeypatch.setattr(st, "JevStage", Boom)
    monkeypatch.setenv("PRIVYSCOPE_JEV_MODEL", "x")
    monkeypatch.delenv("PRIVYSCOPE_JEV_CONFIG", raising=False)
    d, s = st._DefaultStage(), [Span("PER", 0, 3)]
    for _ in range(3):
        assert d.refine("abc", s, []) == s
    assert len(calls) == 1


def test_concurrent_first_calls_load_the_model_once(monkeypatch):
    import threading
    import time

    import privyscope_hybrid_jev.stage as st

    calls = []

    class Slow:
        def __init__(self, cfg):
            calls.append(1)
            time.sleep(0.05)

        def refine(self, text, spans, regex_spans=()):
            return list(spans)

    monkeypatch.setattr(st, "JevStage", Slow)
    monkeypatch.setenv("PRIVYSCOPE_JEV_MODEL", "x")
    monkeypatch.delenv("PRIVYSCOPE_JEV_CONFIG", raising=False)
    d = st._DefaultStage()
    threads = [threading.Thread(target=d.refine, args=("abc", [], [])) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1


def test_langs_gate_runs_the_stage_only_for_listed_languages(monkeypatch):
    pytest.importorskip("privyscope")
    import privyscope._core.plugins as plugins

    monkeypatch.setattr(plugins, "installed_languages", lambda: {"ko": None, "en": None})
    stage, sc = make({}, skip_verified=False, langs=["ko"])
    ko, en = "홍길동에게 연락하세요", "Please contact John Smith"
    stage.refine(ko, [Span("PER", 0, 3)], [])
    assert len(sc.items) == 1                       # Korean text: scored
    out = stage.refine(en, [Span("PER", 15, 25)], [])
    assert len(sc.items) == 1 and out == [Span("PER", 15, 25)]   # English text: untouched, not scored
