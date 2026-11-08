"""Stage 2 (generative model): propose promo candidates as card IDs + verbatim quotes.

The model never writes a timestamp. Times come from the cards in stage 3.
A keyword baseline (no model, no key) exists for comparison and offline runs.
"""
from __future__ import annotations

import re

from .models import call_model
from .parse import Episode

SYSTEM = """You are a promo producer for a vertical microdrama app (ReelShort / DramaBox style).
You read timestamped screenplays and pick moments to cut into 20-60 second social ads
for TikTok, Instagram Reels and YouTube Shorts. You only pick moments; no editing or overlays."""

PROMPT = """A good promo:
1. HOOK: something in the FIRST 3 SECONDS that stops a scrolling thumb - a slap, a reveal,
   a shocking line, a threat, a humiliation. Not an establishing shot, not a slow walk-in.
2. CLIFFHANGER: it CUTS at an unresolved moment - a question unanswered, a threat about to land,
   a reveal half-made. Never end after the tension resolves.
3. STANDALONE: a stranger who has never seen the show understands who wants what within seconds.
   Avoid moments that depend on names, backstory or callbacks the clip does not explain.
4. Total length 20-60 seconds (aim for 25-45). Usually one continuous stretch; you MAY join
   2-3 stretches from this episode (e.g. hook moment + later cliffhanger) if each stretch is clean.

Story context (what happened before; the viewer of the promo has NOT seen this):
{context}

THIS EPISODE: {title}
Logline: {logline}
Continuity notes: {notes}

Screenplay cards. Format: [CARD_ID] start-end TYPE/SPEAKER: text
{cards}

Propose up to {n} distinct promo candidates from this episode, best first. Rules:
- A segment is a range of cards: "first" and "last" card IDs, inclusive, in screenplay order.
- The hook card must be a line of dialogue, an ACTION or a SOUND FX - not a CAMERA description.
- The hook card must be inside the first ~3 seconds of the first segment, so start the
  segment ON the hook (or at most one short card before it).
- The cliffhanger card must be at the very end of the last segment.
- hook_quote and cliffhanger_quote must be copied WORD FOR WORD from those cards' text (5-15 words).
- Do not invent card IDs. Do not write timestamps.
- If the episode has fewer good promos, return fewer. Quality over quantity.

Return JSON only:
{{"promos": [{{"title": "short label",
  "segments": [{{"first": "E01-010", "last": "E01-034"}}],
  "hook_card": "E01-010", "hook_quote": "...",
  "cliffhanger_card": "E01-034", "cliffhanger_quote": "...",
  "why": "one line on why this works as an ad",
  "needs_context": "what a new viewer might not get, or empty string"}}]}}"""

SCHEMA = {
    "type": "object",
    "properties": {"promos": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "segments": {"type": "array", "items": {
                "type": "object",
                "properties": {"first": {"type": "string"}, "last": {"type": "string"}},
                "required": ["first", "last"]}},
            "hook_card": {"type": "string"}, "hook_quote": {"type": "string"},
            "cliffhanger_card": {"type": "string"}, "cliffhanger_quote": {"type": "string"},
            "why": {"type": "string"}, "needs_context": {"type": "string"}},
        "required": ["segments", "hook_card", "hook_quote", "cliffhanger_card",
                     "cliffhanger_quote", "why"]}}},
    "required": ["promos"],
}


def story_context(episodes: list[Episode], upto: int, max_eps: int = 4) -> str:
    prev = [e for e in episodes if e.ep < upto][-max_eps:]
    if not prev:
        return "(first episode)"
    return "\n".join(f"- Ep {e.ep}: {e.logline}" for e in prev)


def build_prompt(ep: Episode, episodes: list[Episode], n: int) -> str:
    return PROMPT.format(context=story_context(episodes, ep.ep), title=ep.title,
                         logline=ep.logline, notes=ep.notes, n=n,
                         cards="\n".join(c.render() for c in ep.cards))


def find_llm(ep: Episode, episodes: list[Episode], cfg: dict, n: int = 5) -> dict:
    prompt = build_prompt(ep, episodes, n)
    res = call_model(prompt, SCHEMA, cfg, system=SYSTEM)
    cands = res["json"].get("promos", []) if isinstance(res["json"], dict) else []
    for k, c in enumerate(cands):
        c["source"] = "llm"
        c["cid"] = f"E{ep.ep:02d}-P{k + 1}"
    return {"episode": ep.ep, "prompt_chars": len(prompt), "raw": res["raw"],
            "usage": res.get("usage"), "model": res.get("model"),
            "cached": res.get("cached", False), "candidates": cands}


# ---------- keyword baseline (no model) ----------

HOOK_WORDS = re.compile(r"slap|strike|struck|scream|gasp|shock|reveal|blood|knife|sword|gun|"
                        r"kneel|grab|throw|shove|tear|crash|kiss|faint|collapse|how dare|liar|"
                        r"traitor|divorce|pregnant|dead|kill|betray", re.I)
CLIFF_WORDS = re.compile(r"cliffhanger|freeze|cut to black|ominous|swell|stinger|sting|"
                         r"\?$|who are you|what\?|no\.\.\.", re.I)


def find_baseline(ep: Episode, n: int = 5) -> dict:
    cards = ep.cards
    cliffs = [i for i, c in enumerate(cards) if CLIFF_WORDS.search(c.text)] or [len(cards) - 1]
    cands = []
    for ci in reversed(cliffs):
        end = cards[ci].end
        # earliest hook-word card that gives a 20-60s span
        for hi, h in enumerate(cards[:ci]):
            if 20 <= end - h.start <= 60 and HOOK_WORDS.search(h.text):
                cands.append({
                    "source": "baseline", "cid": f"E{ep.ep:02d}-B{len(cands) + 1}",
                    "title": "keyword baseline",
                    "segments": [{"first": h.id, "last": cards[ci].id}],
                    "hook_card": h.id, "hook_quote": " ".join(h.text.split()[:10]),
                    "cliffhanger_card": cards[ci].id,
                    "cliffhanger_quote": " ".join(cards[ci].text.split()[:10]),
                    "why": f"keyword hook '{HOOK_WORDS.search(h.text).group()}' + "
                           f"cliff cue '{CLIFF_WORDS.search(cards[ci].text).group() if CLIFF_WORDS.search(cards[ci].text) else 'episode end'}'",
                    "needs_context": ""})
                break
        if len(cands) >= n:
            break
    return {"episode": ep.ep, "candidates": cands}
