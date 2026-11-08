"""Stage 4: score each checked candidate on the brief's rubric.

Default judge is Jev (TypeSafe System One): it returns a probability over each
rubric level instead of free text, so a score of 2.6/4 with confidence 0.4 is
visibly less sure than 2.6/4 with confidence 0.9. Fallbacks: the same rubric as
an LLM prompt via call_model, or a keyword heuristic when no key is available.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .models import call_model

RUBRIC = {
    "hook": ("In the FIRST 3 SECONDS of this clip, would someone scrolling TikTok stop? "
             "Judge only the opening lines (00:00-00:03).",
             ["Nothing happens: establishing shot, walking, small talk",
              "Mild interest: a tense look or a raised voice",
              "Clear tension: an argument, a threat or a surprising line",
              "Strong hook: a confrontation, humiliation or reveal starts immediately",
              "Thumb-stopping: a slap, violence, a shocking line or a huge reveal in the first seconds"]),
    "cliffhanger": ("Does the clip END on an unresolved moment that makes the viewer need the next part? "
                    "Judge only the last 5 seconds. 'after_the_cut_NOT_SHOWN' is what happens next in the show: "
                    "if it simply answers the question, the clip ends on a weak cliffhanger; if the clip "
                    "cuts just before a payoff, that is a strong one.",
                    ["Ends resolved or on nothing",
                     "Ends on a quiet beat with weak tension",
                     "Ends on a clear open question",
                     "Ends mid-confrontation or on a threat about to land",
                     "Ends on a shocking reveal, a cut-off line or imminent danger"]),
    "standalone": ("Would a stranger who has NEVER seen the show understand who these people are "
                   "and what is at stake, from this clip alone? Universal tropes (betrayal, revenge, hidden identity, "
                   "a bully getting their due) need no explanation; callbacks, unexplained names and objects do.",
                   ["Confusing without the earlier episodes",
                    "Needs a lot of backstory",
                    "Mostly clear, a few references are unexplained",
                    "Clear: roles and conflict are obvious",
                    "Instantly clear: a universal situation anyone gets in seconds"]),
    "boundary": ("Are the start and end points clean cuts for an ad: starting at the beginning of a "
                 "moment and stopping at a natural cut, not mid-sentence or mid-action?",
                 ["Starts or ends mid-sentence", "Awkward cut", "Acceptable", "Clean", "Perfect cut points"]),
}
HOOK_TYPES = {
    "physical": "A slap, a strike, a grab, a fall or other physical action",
    "reveal": "A secret, identity or truth is revealed",
    "shocking_line": "A cruel, outrageous or shocking line of dialogue",
    "humiliation": "Someone is publicly humiliated, rejected or betrayed",
    "romance": "A kiss, flirtation or romantic tension",
    "threat": "Danger, a threat or a looming punishment",
    "other": None,
}
WEIGHTS = {"hook": 0.35, "cliffhanger": 0.30, "standalone": 0.25, "boundary": 0.10}
JEV_URL = "https://api.typesafe.ai/v1/systemone"


def _state(c: dict) -> dict:
    return {"clip_length_seconds": c["duration"],
            "format": "Screenplay of a vertical microdrama ad clip. Times are relative to the clip start.",
            "clip": c["text"],
            "after_the_cut_NOT_SHOWN": c.get("after_cut", "")}


def jev_score(c: dict, cfg: dict) -> dict:
    key = cfg.get("api_key") or os.environ.get(cfg.get("key_env", "TYPESAFE_API_KEY"), "")
    questions = {k: {"type": "score", "instructions": q, "criteria": lv} for k, (q, lv) in RUBRIC.items()}
    questions["hook_type"] = {"type": "choice", "instructions": "What kind of hook opens the clip?",
                              "criteria": HOOK_TYPES}
    body = {"model": cfg.get("model", "jev-latest"), "state": _state(c), "questions": questions}
    for attempt in range(5):
        r = requests.post(JEV_URL, json=body, timeout=60,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
            continue
        r.raise_for_status()
        data = r.json()
        a = data["answers"]
        return {"judge": data.get("model", "jev"),
                **{k: round(a[k]["score"], 2) for k in RUBRIC},
                **{f"{k}_conf": round(a[k]["confidence"], 2) for k in RUBRIC},
                "hook_type": a["hook_type"]["choice"],
                "tokens_in": data.get("usage", {}).get("input_tokens", 0)}
    raise RuntimeError(f"Jev kept failing: {r.status_code} {r.text[:200]}")


LLM_JUDGE = """Score each promo clip below on this rubric. Be strict: most clips are NOT a 4.
Use the full 0-4 range and compare clips against each other.

{rubric}

hook_type: one of {types}

{clips}

Return JSON only: {{"scores": [{{"cid": "...", "hook": 0-4, "cliffhanger": 0-4, "standalone": 0-4,
"boundary": 0-4, "hook_type": "...", "reason": "one short line"}}]}}"""


def llm_scores(cands: list[dict], cfg: dict, batch: int = 6) -> dict[str, dict]:
    rubric = "\n".join(f"{k} (0-4): {q}\n  " + " | ".join(f"{i}={l}" for i, l in enumerate(lv))
                       for k, (q, lv) in RUBRIC.items())
    def one(part):
        clips = "\n\n".join(f"=== CLIP {c['cid']} ({c['duration']}s) ===\n{c['text']}\n"
                            f"--- after the cut, NOT shown in the clip (use only for cliffhanger) ---\n"
                            f"{c.get('after_cut', '')}" for c in part)
        try:
            return call_model(LLM_JUDGE.format(rubric=rubric, types=", ".join(HOOK_TYPES), clips=clips),
                              None, cfg, system="You are a strict promo editor.")
        except Exception as e:  # one failed batch should not sink the run: those clips fall back
            return {"json": {"scores": []}, "error": str(e)}

    out = {}
    parts = [cands[i:i + batch] for i in range(0, len(cands), batch)]
    with ThreadPoolExecutor(cfg.get("workers", 8)) as pool:
        for res in pool.map(one, parts):
            for s in (res["json"] or {}).get("scores", []):
                out[s.get("cid")] = {"judge": res.get("model", "llm"), "hook_type": s.get("hook_type", ""),
                                     "judge_reason": s.get("reason", ""),
                                     **{k: float(s.get(k, 0)) for k in RUBRIC}}
    return out


HOOK_RX = re.compile(r"slap|strike|struck|scream|gasp|reveal|blood|knife|sword|kneel|grab|"
                     r"throw|shove|crash|kiss|faint|how dare|liar|traitor|dead|kill|betray", re.I)


def heuristic_score(c: dict) -> dict:
    lines = c["text"].splitlines()
    head = " ".join(l for l in lines if l.split("] ", 1)[-1][:5] < "00:03")
    tail = " ".join(lines[-4:])
    return {"judge": "heuristic",
            "hook": min(4.0, 1.0 + 1.5 * len(HOOK_RX.findall(head))),
            "cliffhanger": 3.0 if re.search(r"\?|ominous|cliffhanger|cut to black|freeze", tail, re.I) else 1.0,
            "standalone": max(0.0, 3.0 - len(c.get("unseen_names", []))),
            "boundary": 2.0 if not any("inside line" in f for f in c["flags"]) else 1.0,
            "hook_type": "other"}


def total(s: dict, c: dict) -> float:
    base = sum(WEIGHTS[k] * s.get(k, 0) for k in WEIGHTS) / 4 * 100
    pen = 4 * min(2, len(c.get("unseen_names", []))) + 2 * sum("inside line" in f for f in c["flags"])
    return round(max(0.0, base - pen), 1)


def panel_scores(cands: list[dict], judges) -> dict[str, dict]:
    """One judge config, or a list of them. A panel averages each rubric score across judges,
    which cancels part of each model's own bias (and its preference for its own picks)."""
    if isinstance(judges, dict):
        return llm_scores(cands, judges)
    runs = [llm_scores(cands, j) for j in judges]
    out = {}
    for c in cands:
        got = [r[c["cid"]] for r in runs if c["cid"] in r]
        if not got:
            continue
        avg = {k: round(sum(g[k] for g in got) / len(got), 2) for k in RUBRIC}
        out[c["cid"]] = {"judge": "panel(" + "+".join(g["judge"] for g in got) + ")", **avg,
                         "hook_type": got[0].get("hook_type", ""),
                         "judge_reason": " | ".join(g.get("judge_reason", "") for g in got),
                         "panel": {g["judge"]: {k: g[k] for k in RUBRIC} for g in got}}
    return out


def score_all(cands: list[dict], cfg: dict, log=print) -> list[dict]:
    judge = cfg.get("judge", "jev")
    if judge == "jev" and not (cfg.get("api_key") or os.environ.get("TYPESAFE_API_KEY")):
        log("No TYPESAFE_API_KEY: falling back to the LLM rubric judge")
        judge = "llm"
    if judge == "jev":
        with ThreadPoolExecutor(cfg.get("workers", 4)) as pool:
            results = list(pool.map(lambda c: jev_score(c, cfg), cands))
        scores = {c["cid"]: s for c, s in zip(cands, results)}
    elif judge == "llm":
        scores = panel_scores(cands, cfg["llm"])
    else:
        scores = {c["cid"]: heuristic_score(c) for c in cands}
    for c in cands:
        s = scores.get(c["cid"]) or heuristic_score(c)
        c["scores"] = s
        c["score"] = total(s, c)
    return cands
