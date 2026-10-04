"""On-disk artifact: ``jev_meta.json`` + ``head.safetensors`` + ``adapter/`` (PEFT)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

META_FILE = "jev_meta.json"
HEAD_FILE = "head.safetensors"
ADAPTER_DIR = "adapter"
NONE = "NONE"
SCHEMA = 1
KNOWN_LABELS = ("PER", "PHONE", "ID_NUM", "EMAIL", "LOC", "BANK", "DATE", "SECRET")


@dataclass
class JevMeta:
    base_model: str
    base_revision: str
    labels: List[str]            # head class order; class index len(labels) is NONE
    temperature: float           # fitted on val; probabilities = softmax(logits / temperature)
    max_length: int = 512
    schema: int = SCHEMA

    def validate(self) -> "JevMeta":
        if not (isinstance(self.temperature, (int, float)) and math.isfinite(self.temperature)
                and self.temperature > 0):
            raise ValueError(f"temperature must be finite and > 0, got {self.temperature!r}")
        if not self.base_revision:
            raise ValueError("base_revision must be a pinned commit/revision, not empty")
        if not self.labels or len(set(self.labels)) != len(self.labels) or NONE in self.labels:
            raise ValueError(f"labels must be non-empty, unique and exclude {NONE!r}: {self.labels}")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        return self

    def save(self, directory) -> None:
        self.validate()
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / META_FILE).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory) -> "JevMeta":
        data = json.loads((Path(directory) / META_FILE).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported artifact schema {data.get('schema')!r} (expected {SCHEMA})")
        return cls(**data).validate()


def check_labels(meta: JevMeta, allowed: Iterable[str]) -> None:
    """Every trained label must be an entity code the core understands."""
    allowed = set(allowed)
    bad = [x for x in meta.labels if x not in allowed]
    if bad:
        raise ValueError(f"unknown label(s) {bad} in artifact; core entity codes: {sorted(allowed)}")


def check_base(meta: JevMeta, requested_revision: Optional[str]) -> None:
    if requested_revision is not None and requested_revision != meta.base_revision:
        raise ValueError(
            f"base revision mismatch: adapter trained on {meta.base_revision}, requested {requested_revision}")
