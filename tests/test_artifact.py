import math

import pytest

from privyscope_hybrid_jev.artifact import KNOWN_LABELS, JevMeta, check_base, check_labels

LABELS = ["PER", "LOC"]


def meta(**kw):
    d = dict(base_model="Qwen/x", base_revision="abc123", labels=LABELS, temperature=1.5)
    d.update(kw)
    return JevMeta(**d)


def test_roundtrip(tmp_path):
    m = meta()
    m.save(tmp_path)
    assert JevMeta.load(tmp_path) == m


@pytest.mark.parametrize("bad", [
    dict(temperature=0.0), dict(temperature=-1.0), dict(temperature=math.nan),
    dict(temperature=math.inf), dict(base_revision=""), dict(labels=[]),
    dict(labels=["PER", "PER"]), dict(labels=["PER", "NONE"]), dict(max_length=0),
])
def test_invalid_meta_rejected(bad):
    with pytest.raises(ValueError):
        meta(**bad).validate()


def test_labels_must_be_known_to_the_core():
    check_labels(meta(), KNOWN_LABELS)
    with pytest.raises(ValueError, match="unknown label"):
        check_labels(meta(labels=["PER", "NAME"]), KNOWN_LABELS)


def test_base_revision_must_match_when_requested():
    check_base(meta(), None)
    check_base(meta(), "abc123")
    with pytest.raises(ValueError, match="revision"):
        check_base(meta(), "other")


def test_load_rejects_unknown_schema(tmp_path):
    meta(schema=99).save(tmp_path)
    with pytest.raises(ValueError, match="schema"):
        JevMeta.load(tmp_path)
