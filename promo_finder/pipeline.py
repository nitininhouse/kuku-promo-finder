"""Glue: parse -> find -> check -> score -> rank -> write outputs.

Every stage's raw output is written to disk so a run can be audited.
"""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .check import Checker
from .find import find_baseline, find_llm
from .parse import Episode
from .score import score_all
from .stitch import stitch
from .tournament import tournament


def _overlap(a: dict, b: dict) -> float:
    best = 0.0
    for s in a["segments_resolved"]:
        for t in b["segments_resolved"]:
            if s["episode"] != t["episode"]:
                continue
            inter = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if inter > 0:
                best = max(best, inter / min(s["seconds"], t["seconds"]))
    return best


def rank(cands: list[dict], per_episode: int = 2, early_bonus: bool = True) -> list[dict]:
    """Dedupe overlapping picks, then build a posting order."""
    for c in cands:
        ep = c["first_episode"]
        c["early_bonus"] = (5 if ep <= 5 else 3 if ep <= 15 else 0) if early_bonus else 0
        c["rank_score"] = c["score"] + c["early_bonus"]
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["rank_score"]):
        dup = next((k for k in kept if _overlap(c, k) > 0.5), None)
        if dup:
            dup.setdefault("duplicates", []).append(c["cid"])
            continue
        kept.append(c)
    # posting order: best first, but no more than `per_episode` from one episode in a row of 10
    order, rest = [], list(kept)
    while rest:
        window = order[-10:]
        pick = next((c for c in rest
                     if sum(o["first_episode"] == c["first_episode"] for o in window) < per_episode), rest[0])
        rest.remove(pick)
        order.append(pick)
    for i, c in enumerate(order, 1):
        c["rank"] = i
        s = c["scores"]
        top = sorted(("hook", "cliffhanger", "standalone"), key=lambda k: -s.get(k, 0))
        c["rank_reason"] = (f"{c['score']:.0f}/100 ({s.get('judge')}); strongest on {top[0]}, "
                            f"weakest on {top[-1]}"
                            + (f"; +{c['early_bonus']} early-episode bonus" if c["early_bonus"] else "")
                            + (f"; won {c['pairwise']:.0f}/100 head-to-head" if "pairwise" in c else "")
                            + (f"; flags: {len(c['flags'])}" if c["flags"] else ""))
    return order


def run(episodes: list[Episode], cfg: dict, out_dir: Path, log=print) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    finder = cfg["finder"]
    n = cfg.get("candidates_per_episode", 5)
    # the judge should come from a different model family than the finder (self-preference bias)
    judge_llm = cfg["scorer"].get("llm") or finder

    def find(ep: Episode) -> dict:
        if finder.get("provider") == "baseline":
            return find_baseline(ep, n)
        try:
            return find_llm(ep, episodes, finder, n)
        except Exception as e:  # keep going; the failure is logged in raw/
            log(f"  ep {ep.ep}: finder failed ({e}); using keyword baseline")
            r = find_baseline(ep, n)
            r["error"] = str(e)
            return r

    log(f"Finding candidates in {len(episodes)} episodes with {finder.get('provider')}:{finder.get('model', '')}")
    with ThreadPoolExecutor(finder.get("workers", 2)) as pool:
        found = list(pool.map(find, episodes))
    for f in found:
        (out_dir / "raw" / f"ep{f['episode']:02d}_finder.json").write_text(json.dumps(f, indent=2))

    # optional extra finders: different models miss different moments, so pool their candidates
    # and let the (single) judge rank them all; overlapping picks are merged by the dedupe in rank()
    for k, extra in enumerate(cfg.get("extra_finders", [])):
        tag = extra.get("model", extra.get("provider", f"x{k}"))
        log(f"Extra finder {tag}")

        def find_extra(ep: Episode, extra=extra) -> dict:
            try:
                return find_llm(ep, episodes, extra, n)
            except Exception as e:
                log(f"  ep {ep.ep}: extra finder failed ({e})")
                return {"episode": ep.ep, "candidates": [], "error": str(e)}

        with ThreadPoolExecutor(extra.get("workers", 4)) as pool:
            more = list(pool.map(find_extra, episodes))
        for f, m in zip(found, more):
            for c in m["candidates"]:
                c["cid"] = f"{c['cid']}-{k + 2}"
                c["source"] = f"llm:{tag}"
            f["candidates"] += m["candidates"]
            (out_dir / "raw" / f"ep{m['episode']:02d}_finder_{k + 2}.json").write_text(json.dumps(m, indent=2))

    checker = Checker(episodes)
    checked = [checker.check(c) for f in found for c in f["candidates"]]
    good = [c for c in checked if c["ok"]]
    bad = [c for c in checked if not c["ok"]]
    with open(out_dir / "rejections.jsonl", "w") as fh:
        for c in bad:
            fh.write(json.dumps({k: c.get(k) for k in ("cid", "source", "title", "segments", "hook_card",
                                                      "hook_quote", "cliffhanger_card", "cliffhanger_quote",
                                                      "reject_reason", "repairs")}) + "\n")
    log(f"Checked {len(checked)} candidates: {len(good)} pass, {len(bad)} rejected")

    log(f"Scoring {len(good)} candidates with {cfg['scorer'].get('judge')}")
    score_all(good, cfg["scorer"] | {"llm": judge_llm}, log=log)

    st_stats = {"skipped": True}
    if cfg.get("stitch", {}).get("enabled", True) and finder.get("provider") != "baseline" and good:
        log("Cross-episode stitch pass")
        try:
            st = stitch(episodes, good, finder, top=cfg.get("stitch", {}).get("pool", 30))
            (out_dir / "raw" / "stitch.json").write_text(json.dumps(st, indent=2))
            xs = [checker.check(c) for c in st["candidates"]]
            x_good = [c for c in xs if c["ok"]]
            score_all(x_good, cfg["scorer"] | {"llm": judge_llm}, log=log)
            good += x_good
            bad += [c for c in xs if not c["ok"]]
            with open(out_dir / "rejections.jsonl", "a") as fh:
                for c in xs:
                    if not c["ok"]:
                        fh.write(json.dumps({k: c.get(k) for k in ("cid", "source", "segments", "hook_quote",
                                                                  "cliffhanger_quote", "reject_reason")}) + "\n")
            st_stats = {"proposed": len(xs), "passed": len(x_good)}
            log(f"  stitch: {len(x_good)}/{len(xs)} cross-episode promos pass the checker")
        except Exception as e:
            st_stats = {"error": str(e)}
            log(f"  stitch failed: {e}")

    ranked = rank(good, cfg.get("max_per_episode", 2), cfg.get("early_bonus", True))
    judge = good[0]["scores"].get("judge", "") if good else ""
    t_cfg = cfg.get("tournament", {})
    t_stats = {"skipped": True}
    if t_cfg.get("k", 24) and judge != "heuristic" and len(ranked) >= 4:
        k = t_cfg.get("k", 24)
        log(f"Pairwise tournament on the top {min(k, len(ranked))}")
        t_stats = tournament(ranked, "jev" if judge.startswith("jev") else "llm",
                             cfg["scorer"] | {"llm": judge_llm}, k=k, rounds=t_cfg.get("rounds", 4), log=log)
        top = rank(ranked[:k], cfg.get("max_per_episode", 2), cfg.get("early_bonus", True))
        ranked = top + ranked[k:]
        for i, c in enumerate(ranked, 1):
            c["rank"] = i
    write_outputs(ranked, out_dir)

    usage = [f.get("usage") or {} for f in found]
    meta = {"episodes": len(episodes), "candidates": len(checked), "passed": len(good),
            "rejected": len(bad), "ranked_after_dedupe": len(ranked),
            "finder": {k: v for k, v in finder.items() if "key" not in k},
            "scorer": {k: v for k, v in cfg["scorer"].items() if "key" not in k},
            "finder_tokens_in": sum(u.get("in", 0) for u in usage),
            "finder_tokens_out": sum(u.get("out", 0) for u in usage),
            "judge_tokens_in": sum(c["scores"].get("tokens_in", 0) for c in good),
            "tournament": t_stats,
            "stitch": st_stats,
            "seconds": round(time.time() - t0, 1)}
    price = cfg.get("prices", {})
    if price:
        meta["finder_cost_usd"] = round(meta["finder_tokens_in"] / 1e6 * price.get("finder_in", 0)
                                        + meta["finder_tokens_out"] / 1e6 * price.get("finder_out", 0), 4)
        meta["judge_cost_usd"] = round(meta["judge_tokens_in"] / 1e6 * price.get("jev_in", 0.042), 4)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Done in {meta['seconds']}s -> {out_dir}")
    return {"ranked": ranked, "rejected": bad, "meta": meta}


def _segments_str(c: dict) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in c["segments_resolved"])


def write_outputs(ranked: list[dict], out_dir: Path) -> None:
    keep = ("rank", "cid", "score", "rank_reason", "title", "duration", "segments_resolved", "hook_card",
            "hook_quote", "cliffhanger_card", "cliffhanger_quote", "why", "needs_context", "scores",
            "flags", "repairs", "duplicates", "source", "abs_score", "pairwise", "text")
    (out_dir / "promos.json").write_text(json.dumps([{k: c.get(k) for k in keep} for c in ranked], indent=2))
    with open(out_dir / "promos.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "score", "duration_s", "segments", "hook", "cliffhanger", "why",
                    "hook_score", "cliff_score", "standalone_score", "boundary_score", "hook_type", "flags"])
        for c in ranked:
            s = c["scores"]
            w.writerow([c["rank"], c["score"], c["duration"], _segments_str(c), c["hook_quote"],
                        c["cliffhanger_quote"], c["why"], s.get("hook"), s.get("cliffhanger"),
                        s.get("standalone"), s.get("boundary"), s.get("hook_type"), "; ".join(c["flags"])])
    md = ["# Ranked promos\n", "| # | Score | Len | Segments | Hook | Cliffhanger |", "|---|---|---|---|---|---|"]
    for c in ranked:
        md.append(f"| {c['rank']} | {c['score']:.0f} | {c['duration']:.0f}s | {_segments_str(c)} | "
                  f"{c['hook_quote']} | {c['cliffhanger_quote']} |")
    for c in ranked:
        md += [f"\n## {c['rank']}. {c.get('title', '')} ({c['score']:.0f}/100)",
               f"- **Segments:** {_segments_str(c)} ({c['duration']}s)",
               f"- **Hook:** {c['hook_quote']}", f"- **Cliffhanger:** {c['cliffhanger_quote']}",
               f"- **Why it works:** {c['why']}", f"- **Why this rank:** {c['rank_reason']}"]
        if c.get("needs_context"):
            md.append(f"- **New viewer may miss:** {c['needs_context']}")
        if c["flags"] or c["repairs"]:
            md.append(f"- **Checks:** {'; '.join(c['flags'] + c['repairs'])}")
    (out_dir / "report.md").write_text("\n".join(md))
