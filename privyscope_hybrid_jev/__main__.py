"""``python -m privyscope_hybrid_jev calibrate ...``"""
from __future__ import annotations

import argparse
import sys

from .config import JevConfig


def _collect(args):
    """Run the real pipeline once per document, capturing merged + regex spans."""
    from privyscope import Privyscope
    from privyscope._eval.dataset import read_eval_jsonl

    captured = []

    class Recorder:
        def refine(self, text, spans, regex_spans):
            captured.append((list(spans), list(regex_spans)))
            return list(spans)

    # explicit stages=[...] also stops auto-discovery from applying the installed stage twice
    engine = Privyscope.from_pretrained(lang=args.lang, regex_only=args.regex_only,
                                        regex_rules=args.regex_rules, stages=[Recorder()])
    records = []
    for text, gold in read_eval_jsonl(args.dataset, limit=args.limit):
        engine.redact(text)
        merged, regex_spans = captured.pop()
        records.append((text, gold, merged, regex_spans))
    return records


def _cmd_calibrate(args) -> int:
    from .calibrate import calibrate, config_to_yaml, grid
    from .loader import load

    base = JevConfig.from_yaml(args.config)
    if not base.configured:
        raise SystemExit("config has no `model`")
    records = _collect(args)
    if len(records) < 20:
        print(f"warning: only {len(records)} records; the held-out split will be too small to trust",
              file=sys.stderr)
    scorer = load(base.model, device=base.device, cache_dir=base.cache_dir,
                  base_revision=base.base_revision, batch_size=base.batch_size)
    configs = grid(base, modes=tuple(args.modes.split(",")))
    print(f"{len(records)} records, {len(configs)} configs, tune={args.tune_fraction:.0%}")
    rep = calibrate(records, scorer, configs, args.tune_fraction)

    f = lambda m: f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"  # noqa: E731
    print(f"items scored: {rep.server_calls}")
    print(f"baseline (no stage)   tune: {f(rep.baseline_tune)}   held-out: {f(rep.baseline_held)}")
    if rep.best is None:
        print("no configs evaluated")
        return 1
    c = rep.best.config
    print(f"best on tune: mode={c.mode} none_threshold={c.none_threshold} skip_verified={c.skip_verified} "
          f"accept_threshold={c.accept_threshold}  {f(rep.best.metrics)}")
    print(f"best on held-out: {f(rep.best_held.metrics)}   (failures: {rep.best_held.failures})")
    if not rep.helps:
        print("RESULT: the stage does not beat the baseline on held-out data; don't enable it for this language.")
        return 2
    print("RESULT: improves held-out F1.")
    if args.write_config:
        with open(args.write_config, "w", encoding="utf-8") as fh:
            fh.write(config_to_yaml(c))
        print(f"wrote {args.write_config}  (use via PRIVYSCOPE_JEV_CONFIG)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="privyscope_hybrid_jev")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="sweep thresholds/mode on labelled JSONL")
    c.add_argument("dataset", help='JSONL: {"text": ..., "spans": [{"label","start","end"}]}')
    c.add_argument("--config", required=True, help="JevConfig YAML (model, device, ...)")
    c.add_argument("--lang", default=None)
    c.add_argument("--regex-only", action="store_true", help="skip NER weights")
    c.add_argument("--regex-rules", default=None)
    c.add_argument("--modes", default="verify", help="comma list; 'verify,extend' scores many more windows")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--tune-fraction", type=float, default=0.5)
    c.add_argument("--write-config", default=None)
    return _cmd_calibrate(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
