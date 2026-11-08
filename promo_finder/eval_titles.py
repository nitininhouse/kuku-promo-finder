"""Second answer key: Kuku's own episode titles.

Kuku's editors name each released episode after its key moment ("Ceiling Is Falling",
"Save My Husband"). For each episode we locate the card that the title points at
(model proposes a card + verbatim quote, code verifies the quote), then ask:
does our top-ranked promo for that episode contain the titled moment?

Usage: python -m promo_finder.eval_titles data outputs/<run> [outputs/<other-run> ...]
Needs eval/kuku_public_episodes.csv (public titles from kukutv.app).
"""
import csv
import json
import os
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .check import quote_match
from .models import call_model
from .parse import card_index, load_show

PROMPT = """The released episode below is titled "{title}" by the show's editors.
Find the ONE card where the moment the title refers to actually happens on screen
(the line is spoken, or the action happens). Return JSON only:
{{"card": "E01-012", "quote": "5-12 words copied word for word from that card"}}

{cards}"""


def locate(ep, title, cfg):
    res = call_model(PROMPT.format(title=title, cards="\n".join(c.render() for c in ep.cards)), None, cfg)
    j = res["json"]
    ids = {c.id: c for c in ep.cards}
    c = ids.get(j.get("card"))
    if c and quote_match(j.get("quote", ""), c.text) >= 0.85:
        return c
    # quote may sit on a neighbour card
    best = max(ep.cards, key=lambda x: quote_match(j.get("quote", ""), x.text))
    return best if quote_match(j.get("quote", ""), best.text) >= 0.85 else None


def contains(promo, card) -> bool:
    return any(s["episode"] == card.ep and s["start"] - 0.5 <= card.start <= s["end"]
               for s in promo["segments_resolved"])


def main(src, runs):
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())
    cfg = tomllib.loads(Path("config.toml").read_text())["finder"]
    eps = {e.ep: e for e in load_show(src)}
    titles = {int(r["episode"]): r["official_title"] for r in csv.DictReader(open("eval/kuku_public_episodes.csv"))}
    todo = [n for n in titles if n in eps]
    with ThreadPoolExecutor(8) as pool:
        anchors = dict(zip(todo, pool.map(lambda n: locate(eps[n], titles[n], cfg), todo)))
    found = {n: a for n, a in anchors.items() if a}
    print(f"title moment located in {len(found)}/{len(todo)} episodes")
    Path("eval/title_anchors.json").write_text(json.dumps({n: a.id for n, a in found.items()}, indent=1))
    for run in runs:
        promos = json.loads((Path(run) / "promos.json").read_text())
        top_hit = any_hit = 0
        for n, card in found.items():
            mine = [p for p in promos if p["segments_resolved"][0]["episode"] == n]
            top_hit += bool(mine) and contains(mine[0], card)
            any_hit += any(contains(p, card) for p in mine)
        print(json.dumps({"run": run, "episodes": len(found),
                          "top_pick_contains_title_moment": top_hit,
                          "any_pick_contains_title_moment": any_hit}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
