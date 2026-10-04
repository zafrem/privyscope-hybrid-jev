def build_prompt(context: str, candidate: str) -> str:
    """The ONLY prompt format: used by training rows and by inference."""
    return f"Text: {context}\nCandidate: {candidate}\nAnswer:"
