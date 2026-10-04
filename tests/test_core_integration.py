"""The stage inside a real privyscope engine (skipped if the core is not importable)."""
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[2] / "privyscope"
pytest.importorskip("privyscope")

from privyscope._api import Privyscope  # noqa: E402
from privyscope._core.regex_filter import RegexFilter  # noqa: E402

from privyscope_hybrid_jev.artifact import JevMeta  # noqa: E402
from privyscope_hybrid_jev.config import JevConfig  # noqa: E402
from privyscope_hybrid_jev.stage import JevStage  # noqa: E402

RULES = CORE / "tests" / "fixtures" / "regex_rules.min.yaml"


class AlwaysNone:
    meta = JevMeta("m", "rev", ["PER", "PHONE"], 1.0)

    def probabilities(self, items):
        return [{"PER": 0.0, "PHONE": 0.0, "NONE": 1.0} for _ in items]


def test_regex_spans_are_trusted_by_default():
    eng = Privyscope(RegexFilter.from_yaml(RULES), stages=[JevStage(JevConfig(model="x"), AlwaysNone())])
    assert eng.redact("call 555-123-4567").redacted_text == "call <PHONE>"


def test_model_can_veto_when_not_skipping_verified():
    stage = JevStage(JevConfig(model="x", skip_verified=False), AlwaysNone())
    eng = Privyscope(RegexFilter.from_yaml(RULES), stages=[stage])
    assert eng.redact("call 555-123-4567").redacted_text == "call 555-123-4567"
