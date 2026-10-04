# privyscope-hybrid-jev

A trained scorer for [privyscope](https://github.com/zafrem/privyscope) PII candidates. A small LoRA
adapter and a linear decision head sit on a pinned open base model; for each candidate span it makes
one forward pass and returns a probability per entity label plus `NONE`. Nothing is generated, so there
is no text to parse and no output to go wrong. It runs as a privyscope refinement stage that can
drop false positives, fix labels and, in `extend` mode, add spans.

```bash
pip install "privyscope-hybrid-jev[torch]"
export PRIVYSCOPE_JEV_MODEL=<hub repo id or local artifact dir>
```

With the model set, every `privyscope` engine runs the stage automatically (it registers under the
`privyscope.stages` entry point). With no model set it does nothing and loads nothing. For more
options use a YAML file via `PRIVYSCOPE_JEV_CONFIG` (see `JevConfig`: `mode`, `none_threshold`,
`skip_verified`, `device`, ...). `python -m privyscope_hybrid_jev calibrate` tunes the thresholds on
your labelled data and refuses to write a config that does not beat the no-stage baseline.

The wheel contains no weights. The adapter, head, labels and temperature are one artifact
(`jev_meta.json`, `head.safetensors`, `adapter/`) hosted on the Hugging Face Hub; the base model is
pinned by revision in the artifact and is downloaded separately. It needs a privyscope core that has
the `privyscope.stages` hook. It is independent of `privyscope-local-llm`.

## Enabling it for some languages only

`langs: [ko, ja]` in the YAML config runs the stage only for texts the core detects as those languages
(codes as in `--lang`). Detection runs among the installed language packs plus the listed ones, exactly
like `Privyscope.auto()`; it needs the privyscope core installed. Spans whose label the head was not
trained on (URL, IP, ...) are never scored, so the stage cannot delete or re-label them.
