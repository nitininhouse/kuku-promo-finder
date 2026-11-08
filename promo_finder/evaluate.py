"""Compare runs against hand-marked gold promos.

eval/gold.csv holds only card IDs (no script text), so it is safe to commit:
    episode,first_card,last_card
    1,E01-007,E01-031
Usage: python -m promo_finder.evaluate data outputs/full_deepseek outputs/baseline
"""
import csv
import json
import sys
from pathlib import Path

from .parse import card_index, load_show


def span(ids, first, last):
    a, b = ids[first], ids[last]
    return a.start, b.end


def iou(x, y):
    inter = max(0.0, min(x[1], y[1]) - max(x[0], y[0]))
    return inter / (max(x[1], y[1]) - min(x[0], y[0]))


def evaluate(source, run_dir, gold_path="eval/gold.csv"):
    ids = card_index(load_show(source))
    gold = list(csv.DictReader(open(gold_path)))
    promos = json.loads((Path(run_dir) / "promos.json").read_text())
    rows = []
    for g in gold:
        ep = int(g["episode"])
        gs = span(ids, g["first_card"], g["last_card"])
        mine = [p for p in promos if p["segments_resolved"][0]["episode"] == ep]
        ious = [max(iou(gs, (s["start"], s["end"])) for s in p["segments_resolved"] if s["episode"] == ep)
                for p in mine]
        best = max(ious, default=0.0)
        top = ious[0] if ious else 0.0  # promos.json is in rank order, so mine[0] is this episode's top pick
        rows.append({"episode": ep, "best_iou": round(best, 2), "hit": best >= 0.5, "top_pick_hit": top >= 0.5,
                     "candidates": len(mine)})
    n = len(rows) or 1
    summary = {"run": str(run_dir), "gold": len(rows),
               "hits@0.5": sum(r["hit"] for r in rows), "top_pick_hits": sum(r["top_pick_hit"] for r in rows),
               "mean_best_iou": round(sum(r["best_iou"] for r in rows) / n, 2)}
    return summary, rows


if __name__ == "__main__":
    src, *runs = sys.argv[1:]
    for r in runs:
        s, rows = evaluate(src, r)
        print(json.dumps(s))
        for row in rows:
            print("   ", row)
