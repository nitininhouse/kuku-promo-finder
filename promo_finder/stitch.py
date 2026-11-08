"""Stage 2b (LLM, one call per show): cross-episode promos.

Per-episode finding can't join a hook in episode 12 to a cliffhanger in episode 13.
This pass shows the model the season's loglines plus the best single-episode clips
(as cards) and asks for promos stitched from 2-3 of them. Output uses the same
schema as find.py, so the same checker and judge apply.
"""
from __future__ import annotations

from .find import SCHEMA, SYSTEM
from .models import call_model
from .parse import Episode

PROMPT = """You are building CROSS-EPISODE promos: one ad stitched from 2-3 moments in DIFFERENT episodes.
Good stitches: a setup or threat in one episode + its peak in a later one; a humiliation + the
revenge that answers it, cut just before the payoff lands; the same character escalating.
Every promo still needs a hook in its first 3 seconds, a cliffhanger at the very end, 20-60s total,
and must make sense to a stranger. Keep segments in story order (earlier episode first).

Season so far (loglines):
{season}

Clips you may cut from (cards with IDs; times are relative to each clip's start):
{clips}

Propose up to {n} cross-episode promos. Each segment is a range of card IDs ("first".."last",
inclusive) taken from ONE clip above. Quotes must be copied word for word from the named card.
Only propose a stitch if it is better than either clip alone. Return JSON only, same format:
{{"promos": [{{"title": "...", "segments": [{{"first": "E12-032", "last": "E12-045"}},
{{"first": "E13-010", "last": "E13-022"}}], "hook_card": "E12-032", "hook_quote": "...",
"cliffhanger_card": "E13-022", "cliffhanger_quote": "...", "why": "...", "needs_context": ""}}]}}"""


def stitch(episodes: list[Episode], scored: list[dict], cfg: dict, top: int = 30, n: int = 8) -> dict:
    pool = sorted(scored, key=lambda c: -c["score"])[:top]
    if len({c["first_episode"] for c in pool}) < 2:
        return {"candidates": [], "skipped": True}
    season = "\n".join(f"- Ep {e.ep}: {e.logline}" for e in episodes)
    clips = "\n\n".join(f"=== CLIP from Ep {c['first_episode']} ({c['duration']}s) ===\n{c['text']}"
                        for c in sorted(pool, key=lambda c: c["first_episode"]))
    res = call_model(PROMPT.format(season=season, clips=clips, n=n), SCHEMA, cfg, system=SYSTEM)
    cands = res["json"].get("promos", []) if isinstance(res["json"], dict) else []
    multi = []
    for k, c in enumerate(cands):
        eps = {s.get("first", "")[:3] for s in c.get("segments", [])}
        c.update(source="stitch", cid=f"X-P{k + 1}")
        if len(eps) > 1:
            multi.append(c)
    return {"raw": res["raw"], "usage": res.get("usage"), "candidates": multi,
            "dropped_single_episode": len(cands) - len(multi)}
