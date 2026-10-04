from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import List, Optional


@dataclass
class JevConfig:
    model: str = ""                      # Hub repo id or local artifact dir; empty -> stage is a no-op
    device: Optional[str] = None         # None -> cuda if available else cpu
    cache_dir: Optional[str] = None
    base_revision: Optional[str] = None  # if set, must equal the artifact's pinned revision
    batch_size: int = 8
    mode: str = "verify"                 # "verify" | "extend"
    none_threshold: float = 0.5          # drop a candidate when P(NONE) >= this
    accept_threshold: float = 0.5        # extend: add a window when P(best label) >= this
    skip_verified: bool = True           # regex spans bypass the model
    context_chars: int = 200
    max_ngram: int = 3                   # extend: window size in whitespace tokens
    max_candidates: int = 64             # extend: cap on windows scored per text
    langs: Optional[List[str]] = None    # run only for texts detected as these language codes (None = all)

    def __post_init__(self) -> None:
        if self.mode not in ("verify", "extend"):
            raise ValueError(f"mode must be 'verify' or 'extend', got {self.mode!r}")

    @property
    def configured(self) -> bool:
        return bool(self.model)

    @classmethod
    def from_yaml(cls, path: str) -> "JevConfig":
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_env(cls) -> "JevConfig":
        """``PRIVYSCOPE_JEV_CONFIG`` (YAML path) wins; else ``PRIVYSCOPE_JEV_MODEL`` / ``_MODE``."""
        path = os.environ.get("PRIVYSCOPE_JEV_CONFIG")
        if path:
            return cls.from_yaml(path)
        return cls(model=os.environ.get("PRIVYSCOPE_JEV_MODEL", ""),
                   mode=os.environ.get("PRIVYSCOPE_JEV_MODE", "verify"))
