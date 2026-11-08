"""Stage 5: pairwise re-rank of the top clips.

Absolute 0-4 scores bunch at the top (judges are lenient), so the top K are also
compared head to head ("which would you post first?") over a few Swiss rounds and
fitted with Bradley-Terry. Every pair is asked in both A/B orders; a verdict
that flips with the order is a tie, and the flip rate is reported.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
import requests

from .models import call_model
from .score import JEV_URL

PAIR_Q = ("Both clips are candidate ads for the same vertical drama, posted to TikTok to get new viewers "
          "to install the app. Which clip would you post first? Consider: does it stop the scroll in the "
          "first 3 seconds, does it end on a cliffhanger, and would a stranger understand it?")


def _jev_pair(a: dict, b: dict, key: str) -> float:
    body = {"model": "jev-latest",
            "state": {"clip_A": a["text"], "clip_B": b["text"]},
            "questions": {"pick": {"type": "choice", "instructions": PAIR_Q,
                                   "criteria": {"A": "Post clip A first", "B": "Post clip B first"}}}}
    r = requests.post(JEV_URL, json=body, timeout=60, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return float(r.json()["answers"]["pick"]["probabilities"]["A"])


LLM_PAIRS = PAIR_Q + """

{pairs}

Return JSON only: {{"winners": [{{"pair": 1, "winner": "A" or "B"}}, ...]}}"""


def _llm_pairs(pairs: list[tuple[dict, dict]], cfg: dict) -> list[float]:
    try:
        return _llm_pairs_once(pairs, cfg)
    except Exception:  # a failed call counts as ties, not a crash
        return [0.5] * len(pairs)


def _llm_pairs_once(pairs: list[tuple[dict, dict]], cfg: dict) -> list[float]:
    body = "\n\n".join(f"##### PAIR {i + 1}\n--- CLIP A ---\n{a['text']}\n--- CLIP B ---\n{b['text']}"
                       for i, (a, b) in enumerate(pairs))
    res = call_model(LLM_PAIRS.format(pairs=body), None, cfg, system="You are a strict promo editor.")
    won = {w.get("pair"): w.get("winner") for w in res["json"].get("winners", [])}
    return [1.0 if won.get(i + 1) == "A" else 0.0 if won.get(i + 1) == "B" else 0.5 for i in range(len(pairs))]


def judge_pairs(pairs, judge: str, cfg: dict) -> list[float]:
    """P(A beats B) for each pair."""
    if judge == "jev":
        key = cfg.get("api_key") or os.environ.get("TYPESAFE_API_KEY", "")
        return [_jev_pair(a, b, key) for a, b in pairs]
    judges = cfg["llm"] if isinstance(cfg["llm"], list) else [cfg["llm"]]
    chunks = [pairs[i:i + 6] for i in range(0, len(pairs), 6)]
    per_judge = []
    with ThreadPoolExecutor(8) as pool:
        for j in judges:  # a panel averages P(A wins) across judges
            per_judge.append([p for res in pool.map(lambda ch: _llm_pairs(ch, j), chunks) for p in res])
    return [sum(ps) / len(ps) for ps in zip(*per_judge)]


def bradley_terry(n: int, games: list[tuple[int, int, float]], iters: int = 200) -> list[float]:
    """MM fit; games are (i, j, p_i_wins). A small prior keeps unbeaten clips finite."""
    w = [0.5] * n
    g = [0.0] * n
    for i, j, p in games:
        w[i] += p
        w[j] += 1 - p
    s = [1.0] * n
    for _ in range(iters):
        new = []
        for i in range(n):
            denom = sum(1 / (s[i] + s[j]) for a, b, _ in games for (x, j) in ((a, b), (b, a)) if x == i)
            new.append(w[i] / (denom + 1 / (s[i] + 1)))
        tot = sum(new)
        s = [x * n / tot for x in new]
    return s


def tournament(ranked: list[dict], judge: str, cfg: dict, k: int = 24, rounds: int = 4,
               log=print) -> dict:
    top = ranked[:k]
    n = len(top)
    if n < 4:
        return {"skipped": True}
    strength = [c["score"] for c in top]
    games, seen = [], set()
    flips = asked = first_won = 0
    for r in range(rounds):
        order = sorted(range(n), key=lambda i: -strength[i])
        pairs = []
        while len(order) > 1:
            i = order.pop(0)
            j = next((x for x in order if (min(i, x), max(i, x)) not in seen), order[0])
            order.remove(j)
            seen.add((min(i, j), max(i, j)))
            pairs.append((i, j))
        # ask every pair in both orders; a verdict that flips with the order counts as a tie
        fwd = judge_pairs([(top[a], top[b]) for a, b in pairs], judge, cfg)
        rev = judge_pairs([(top[b], top[a]) for a, b in pairs], judge, cfg)
        for (a, b), p, q in zip(pairs, fwd, rev):
            asked += 1
            first_wins = (p > 0.5) and (q > 0.5) or (p < 0.5) and (q < 0.5)
            flips += first_wins
            first_won += (p > 0.5) + (q > 0.5)
            games.append((a, b, (p + (1 - q)) / 2))
        strength = bradley_terry(n, games)
        log(f"  tournament round {r + 1}/{rounds}: {len(pairs)} pairs x 2 orders")
    flip_rate = flips / max(1, asked)

    lo, hi = min(strength), max(strength)
    for c, s in zip(top, strength):
        c["abs_score"] = c["score"]
        c["pairwise"] = round(100 * (s - lo) / (hi - lo or 1), 1)
        c["score"] = round(0.7 * c["pairwise"] + 0.3 * c["abs_score"], 1)
    stats = {"clips": n, "pairs": len(games), "order_flip_rate": round(flip_rate, 2),
             "first_shown_win_rate": round(first_won / max(1, 2 * asked), 2), "judge": judge}
    log(f"  tournament: {flip_rate:.0%} of verdicts flipped with A/B order (counted as ties)")
    return stats
