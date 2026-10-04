from __future__ import annotations

from pathlib import Path
from typing import Optional

from .artifact import ADAPTER_DIR, HEAD_FILE, JevMeta, check_base
from .scorer import JevScorer


def load(repo_or_path: str, *, device: Optional[str] = None, cache_dir: Optional[str] = None,
         base_revision: Optional[str] = None, batch_size: int = 8) -> JevScorer:
    """Load base (pinned revision) + LoRA adapter + head from a Hub repo id or local directory."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    from .head import load_head

    path = Path(repo_or_path)
    if not path.is_dir():
        path = Path(snapshot_download(repo_or_path, cache_dir=cache_dir))
    meta = JevMeta.load(path)
    check_base(meta, base_revision)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(meta.base_model, revision=meta.base_revision)
    base = AutoModel.from_pretrained(meta.base_model, revision=meta.base_revision, torch_dtype=dtype)
    backbone = PeftModel.from_pretrained(base, str(path / ADAPTER_DIR)).to(device).eval()
    head = load_head(path / HEAD_FILE).to(device).eval()
    return JevScorer(backbone, tokenizer, head, meta, device=device, batch_size=batch_size)
