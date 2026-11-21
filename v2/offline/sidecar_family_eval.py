"""Sidecar scores on the same test rows as long_context_eval.py, split by whether the row's family existed when
probe.json was trained (row id present in out/<model>/scores.json) or was added later (H1 families).
Writes out/<model>/sidecar_by_family.json. Usage: python sidecar_family_eval.py --model qwen0.5b [--n 40]
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, SEED, Run, run_meta  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    from anchor.sidecar import SidecarScorer
    run = Run(a.model, MODELS[a.model]["default_precision"])
    sc = SidecarScorer(run)
    seen = set(json.loads((run.out / "scores.json").read_text())["scores"])
    rows = [json.loads(l) for l in open(run.data / "rows.jsonl")]
    test = [r for r in rows if r["split"] == "test" and not r.get("prior_outputs")]
    rng = random.Random(SEED)  # same pick as long_context_eval.py
    pick = []
    for lab in (0, 1):
        part = [r for r in test if r["label"] == lab]
        rng.shuffle(part)
        pick += part[:a.n]
    out = collections.defaultdict(lambda: {"flagged": 0, "n": 0, "styles": collections.Counter()})
    for r in pick:
        s = sc.raw_score(r["task"], r["output"])["drift_score"]
        k = f"{'injected' if r['label'] else 'clean'}_{'trained_family' if r['id'] in seen else 'new_family'}"
        out[k]["n"] += 1
        if s >= sc.threshold:
            out[k]["flagged"] += 1
            out[k]["styles"][r["style"]] += 1
    res = {"run_meta": run_meta(run, sc.device), "threshold": sc.threshold, "probe_trained_rows": len(seen),
           "dataset_rows": len(rows), "results": {k: {**v, "styles": dict(v["styles"])} for k, v in sorted(out.items())}}
    (run.out / "sidecar_by_family.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["results"], indent=1))


if __name__ == "__main__":
    main()
