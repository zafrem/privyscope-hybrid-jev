import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CORE = HERE.parent / "privyscope"          # sibling working tree (has the stages hook)
if (CORE / "privyscope").is_dir():
    sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE))
