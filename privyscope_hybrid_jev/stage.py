"""The ``privyscope.stages`` refinement stage."""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

from .artifact import KNOWN_LABELS, NONE, check_labels
from .candidates import candidate_windows, context_for
from .config import JevConfig
from .types import Span

log = logging.getLogger(__name__)


class JevStage:
    """Verify (and optionally extend) detections with the trained head.

    ``verify``: scores spans other stages proposed; drops false positives, re-labels.
    ``extend``: also scores token windows so the model can add spans.
    Scoring errors propagate; the core skips a failing stage and keeps the incoming spans.
    """

    def __init__(self, config: JevConfig, scorer=None) -> None:
        self.config = config
        if scorer is None:
            from .loader import load  # lazy: torch/transformers/peft only when really loading

            scorer = load(config.model, device=config.device, cache_dir=config.cache_dir,
                          base_revision=config.base_revision, batch_size=config.batch_size)
        check_labels(scorer.meta, KNOWN_LABELS)
        self.scorer = scorer
        self._detect = None
        self._avail: set = set()
        if config.langs:
            try:
                from privyscope._core.detect import detect_language
                from privyscope._core.plugins import installed_languages
            except ImportError as exc:  # the language gate needs the core's detector
                raise ImportError("JevConfig.langs needs the privyscope core installed") from exc
            try:
                installed = set(installed_languages())
            except Exception:  # noqa: BLE001 - a broken plugin must not disable detection routing
                installed = set()
            # detect among installed + enabled languages, like Privyscope.auto(): detect_language falls
            # back to the sole available code, so detecting among only the enabled ones would never skip.
            self._avail = set(config.langs) | installed
            self._detect = detect_language

    @staticmethod
    def _best(probs) -> str:
        return max((k for k in probs if k != NONE), key=probs.get)

    def refine(self, text: str, spans: Sequence[Span], regex_spans: Sequence[Span] = ()) -> List[Span]:
        cfg = self.config
        spans = list(spans)
        if self._detect is not None and self._detect(text, self._avail) not in cfg.langs:
            return spans  # not one of the enabled languages: leave detections untouched
        trusted = set(regex_spans) if cfg.skip_verified else set()
        known = set(self.scorer.meta.labels)  # the head cannot judge labels it was never trained on (URL, IP, ...)
        to_check = [s for s in spans if s not in trusted and s.label in known]
        windows = (candidate_windows(text, spans, cfg.max_ngram, cfg.max_candidates)
                   if cfg.mode == "extend" else [])
        items = [(context_for(text, s, cfg.context_chars), text[s.start : s.end]) for s in to_check]
        items += [(context_for(text, Span("", a, b), cfg.context_chars), text[a:b]) for a, b in windows]
        if not items:
            return spans
        probs = self.scorer.probabilities(items)

        verdict = {}
        for s, p in zip(to_check, probs):
            verdict[s] = None if p[NONE] >= cfg.none_threshold else Span(self._best(p), s.start, s.end)
        out = [v for v in (verdict.get(s, s) for s in spans) if v is not None]

        added = []
        for (a, b), p in zip(windows, probs[len(to_check):]):
            label = self._best(p)
            if p[NONE] < cfg.none_threshold and p[label] >= cfg.accept_threshold:
                added.append((p[label], Span(label, a, b)))
        taken: List[Span] = []
        for _score, cand in sorted(added, key=lambda t: (-t[0], -(t[1].end - t[1].start))):
            if not any(not (cand.end <= t.start or cand.start >= t.end) for t in taken):
                taken.append(cand)
        return sorted(out + taken, key=lambda s: s.start)


class _DefaultStage:
    """Entry-point object. A no-op until a model is configured via env/YAML."""

    def __init__(self) -> None:
        self._inner: Optional[JevStage] = None
        self._resolved = False
        self._lock = threading.Lock()

    def refine(self, text, spans, regex_spans=()):
        if not self._resolved:
            with self._lock:  # concurrent first calls must not each load a multi-GB model
                if not self._resolved:
                    try:
                        cfg = JevConfig.from_env()
                        self._inner = JevStage(cfg) if cfg.configured else None
                    except Exception as exc:  # noqa: BLE001 - misconfig: say so once, then stay a no-op
                        log.error("privyscope-hybrid-jev disabled, could not load the model: %s", exc)
                        self._inner = None
                    self._resolved = True
        return list(spans) if self._inner is None else self._inner.refine(text, spans, regex_spans)


default_stage = _DefaultStage()
