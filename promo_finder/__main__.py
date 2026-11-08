"""CLI: python -m promo_finder <folder-or-zip> [--provider gemini --model ... --judge jev --out outputs/run]"""
import argparse
import os
import time
import tomllib
from pathlib import Path

from .parse import load_show
from .pipeline import run

if Path(".env").exists():  # tiny .env loader, no extra dependency
    for line in Path(".env").read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and not k.strip().startswith("#") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())

ap = argparse.ArgumentParser(prog="promo_finder")
ap.add_argument("source", help="folder of episode .txt files, or a .zip")
ap.add_argument("--config", default="config.toml")
ap.add_argument("--provider")
ap.add_argument("--model")
ap.add_argument("--judge", choices=["jev", "llm", "heuristic"])
ap.add_argument("--episodes", help="e.g. 1-5 or 3,7,9")
ap.add_argument("--out")
a = ap.parse_args()

cfg = tomllib.loads(Path(a.config).read_text())
if a.provider:
    cfg["finder"]["provider"] = a.provider
    if not a.model:
        cfg["finder"].pop("model", None)
if a.model:
    cfg["finder"]["model"] = a.model
if a.judge:
    cfg["scorer"]["judge"] = a.judge

eps = load_show(a.source)
if a.episodes:
    want = set()
    for part in a.episodes.split(","):
        lo, _, hi = part.partition("-")
        want |= set(range(int(lo), int(hi or lo) + 1))
    eps = [e for e in eps if e.ep in want]
out = Path(a.out or f"outputs/{time.strftime('%Y%m%d-%H%M%S')}-{cfg['finder']['provider']}")
res = run(eps, cfg, out)
for c in res["ranked"][:10]:
    segs = " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in c["segments_resolved"])
    print(f"{c['rank']:>2}. {c['score']:5.1f}  {c['duration']:4.0f}s  {segs}  | {c['hook_quote'][:60]}")
