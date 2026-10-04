"""Use the core's Span when installed so stages interoperate exactly."""
try:
    from privyscope._core.bioes import Span
except ImportError:  # standalone use / tests without the core
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Span:  # type: ignore[no-redef]
        label: str
        start: int
        end: int
