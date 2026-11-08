"""Third answer key: promos that were really posted for this show.

eval/real_promos.csv lists YouTube Shorts posted for the show (a paid clipping
campaign), mapped to card IDs by matching their auto-transcripts to the dialogue.
For each real ad we ask: did the tool pick the same moment (>=50% overlap)?
Views let us weight the result by what real audiences watched.

Usage: python -m promo_finder.eval_real data outputs/<run> [outputs/<other-run> ...]
"""
import csv
import json
import sys
from pathlib import Path

from .parse import card_index, load_show


def overlap(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, inter) / max(0.1, min(a[1] - a[0], b[1] - b[0]))


def main(src, runs):
    ids = card_index(load_show(src))
    real = [r for r in csv.DictReader(open("eval/real_promos.csv")) if r["first_card"]]
    for run in runs:
        promos = json.loads((Path(run) / "promos.json").read_text())
        hits = top_hits = 0
        v_hit = v_all = 0
        detail = []
        for r in real:
            ep = int(r["main_episode"])
            span = (ids[r["first_card"]].start, ids[r["last_card"]].end)
            mine = [(p["rank"], max((overlap(span, (s["start"], s["end"])) for s in p["segments_resolved"]
                                     if s["episode"] == ep), default=0.0)) for p in promos]
            mine = [m for m in mine if m[1] > 0]
            best = max(mine, key=lambda m: m[1], default=(None, 0.0))
            ep_promos = [p for p in promos if any(s["episode"] == ep for s in p["segments_resolved"])]
            top = ep_promos[0] if ep_promos else None
            top_ov = max((overlap(span, (s["start"], s["end"])) for s in top["segments_resolved"]
                          if s["episode"] == ep), default=0.0) if top else 0.0
            hit = best[1] >= 0.5
            hits += hit
            top_hits += top_ov >= 0.5
            views = int(r["views"] or 0)
            v_all += views
            v_hit += views * hit
            detail.append({"video": r["video_id"], "views": views, "ep": ep, "hit": hit,
                           "best_overlap": round(best[1], 2), "our_rank": best[0] if hit else None})
        print(json.dumps({"run": run, "real_ads": len(real), "found_same_moment": hits,
                          "episode_top_pick_matches": top_hits,
                          "view_weighted_recall": round(v_hit / max(1, v_all), 2)}))
        for d in sorted(detail, key=lambda d: -d["views"]):
            print("   ", d)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
