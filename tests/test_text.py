import pytest

from privyscope_hybrid_jev.candidates import candidate_windows, context_for
from privyscope_hybrid_jev.config import JevConfig
from privyscope_hybrid_jev.prompt import build_prompt
from privyscope_hybrid_jev.types import Span

TEXT = "First one. Contact Alice Kim at home. Last."


def test_prompt_format_is_fixed():
    assert build_prompt("ctx", "cand") == "Text: ctx\nCandidate: cand\nAnswer:"


def test_context_is_the_sentence_around_the_span():
    assert context_for(TEXT, Span("PER", 19, 28), 200) == "Contact Alice Kim at home."


def test_context_is_clipped_to_chars():
    long = "a" * 500 + " Bob " + "b" * 500
    ctx = context_for(long, Span("PER", 501, 504), 20)
    assert len(ctx) <= 20 + 3 + 20 and "Bob" in ctx


def test_windows_avoid_taken_and_respect_cap():
    text = "Ask Bob or Carol now"
    w = candidate_windows(text, [Span("PER", 4, 7)], 2, 50)
    assert (4, 7) not in w and all(b <= 4 or a >= 7 for a, b in w)
    assert len(candidate_windows(text, [], 3, 5)) == 5


def test_config_validation_and_configured():
    assert not JevConfig().configured and JevConfig(model="repo").configured
    with pytest.raises(ValueError, match="mode"):
        JevConfig(mode="nope")


def test_config_yaml_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: r\nmode: extend\nnone_threshold: 0.7\n")
    cfg = JevConfig.from_yaml(str(p))
    assert (cfg.model, cfg.mode, cfg.none_threshold) == ("r", "extend", 0.7)
    p.write_text("model: r\ntypo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        JevConfig.from_yaml(str(p))


def test_config_env_precedence(monkeypatch, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: from-yaml\n")
    monkeypatch.setenv("PRIVYSCOPE_JEV_MODEL", "from-env")
    monkeypatch.delenv("PRIVYSCOPE_JEV_CONFIG", raising=False)
    assert JevConfig.from_env().model == "from-env"
    monkeypatch.setenv("PRIVYSCOPE_JEV_CONFIG", str(p))
    assert JevConfig.from_env().model == "from-yaml"


def test_config_langs_default_none_and_loads_from_yaml(tmp_path):
    assert JevConfig(model="r").langs is None
    p = tmp_path / "c.yaml"
    p.write_text("model: r\nlangs: [ko, ja]\n")
    assert JevConfig.from_yaml(str(p)).langs == ["ko", "ja"]
