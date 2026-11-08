"""Stage 3 (code only): turn card IDs into timestamps and enforce the brief's rules.

Hard rules (reject):  IDs exist, segments in order, 20-60s total, hook in first 3s,
                      cliffhanger at the very end, quotes really appear in the script.
Repairs (logged):     trim a segment so it starts on the hook / ends on the cliffhanger.
Soft flags (shown):   cuts into a line of dialogue, scene jumps, names a new viewer won't know.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .parse import Card, Episode, fmt

MIN_S, MAX_S, HOOK_S, CLIFF_S = 20.0, 60.0, 3.0, 5.0
GENERIC = {"LADY", "LORD", "THE", "SIR", "MAID", "GUARD", "MAN", "WOMAN", "OLD", "YOUNG", "ELDER",
           "BLOND", "MYSTERIOUS", "BEARDED", "FORMER", "HIGH", "FATHER", "MOTHER", "JUDGE", "DUKE",
           "DUCHESS", "DOWAGER", "KING", "QUEEN", "PRIEST", "NOBLEMAN", "NOBLEWOMAN", "STRANGER",
           "GROOM", "BRIDE", "IN", "GOLD", "GOWN", "OF", "AND", "MATRIARCH", "VOICE", "CROWD", "ALL"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", s.lower().replace("’", "'"))).strip()


def quote_match(quote: str, text: str) -> float:
    q, t = norm(quote), norm(text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    sm = SequenceMatcher(None, q, t)
    blocks = sum(b.size for b in sm.get_matching_blocks())
    return blocks / len(q)


def character_names(episodes: list[Episode]) -> set[str]:
    names = set()
    for e in episodes:
        for c in e.cards:
            if c.kind == "DIALOGUE":
                for w in re.findall(r"[A-Z][A-Z'\-]+", c.speaker.upper()):
                    w = w.strip("'")
                    if w.endswith("'S"):
                        w = w[:-2]
                    if w not in GENERIC and len(w) > 2:
                        names.add(w.title())
    return names


class Checker:
    def __init__(self, episodes: list[Episode]):
        self.by_ep = {e.ep: e for e in episodes}
        self.pos = {c.id: (e.ep, i) for e in episodes for i, c in enumerate(e.cards)}
        self.names = character_names(episodes)

    def _span(self, ep: int, a: int, b: int) -> list[Card]:
        return self.by_ep[ep].cards[a:b + 1]

    def check(self, cand: dict) -> dict:
        out = dict(cand)
        out.update(ok=False, flags=[], repairs=[], reject_reason="")

        def reject(why: str) -> dict:
            out["reject_reason"] = why
            return out

        # 1. resolve segments to (ep, first_idx, last_idx)
        segs = []
        for s in cand.get("segments") or []:
            a, b = s.get("first"), s.get("last")
            if a not in self.pos or b not in self.pos:
                return reject(f"unknown card id {a if a not in self.pos else b}")
            (ea, ia), (eb, ib) = self.pos[a], self.pos[b]
            if ea != eb or ib < ia:
                return reject(f"bad segment {a}..{b}")
            segs.append([ea, ia, ib])
        if not segs:
            return reject("no segments")

        # 2. quotes must really be in the cards the model named
        for role in ("hook", "cliffhanger"):
            cid, q = cand.get(f"{role}_card"), cand.get(f"{role}_quote", "")
            if cid not in self.pos:
                return reject(f"{role} card {cid} does not exist")
            ep, i = self.pos[cid]
            card = self.by_ep[ep].cards[i]
            score = quote_match(q, card.text)
            if score < 0.85:
                # the model often quotes the right line but names the neighbouring card
                near = [(quote_match(q, c.text), c) for c in self.by_ep[ep].cards[max(0, i - 3):i + 4]]
                best, bc = max(near, key=lambda x: x[0])
                if best >= 0.85:
                    out["repairs"].append(f"{role} quote found on {bc.id}, not {cid}")
                    out[f"{role}_card"] = bc.id
                    continue
                return reject(f"{role} quote not found in script (match {score:.0%}): {q!r}")
            if score < 1.0:
                out["flags"].append(f"{role} quote paraphrased ({score:.0%} match)")

        # 3. hook must open the promo: trim the first segment to start on it
        hep, hi = self.pos[out["hook_card"]]
        if not (segs[0][0] == hep and segs[0][1] <= hi <= segs[0][2]):
            return reject("hook card is not in the first segment")
        hook = self.by_ep[hep].cards[hi]
        seg0_start = min(c.start for c in self._span(*segs[0]))
        if hook.start - seg0_start > HOOK_S:
            segs[0][1] = hi
            out["repairs"].append(f"trimmed start to hook card ({hook.start - seg0_start:.1f}s of lead-in cut)")

        # 4. cliffhanger must close the promo: trim the last segment to end on it
        cep, ci = self.pos[out["cliffhanger_card"]]
        if not (segs[-1][0] == cep and segs[-1][1] <= ci <= segs[-1][2]):
            return reject("cliffhanger card is not in the last segment")
        if ci < segs[-1][2]:
            cliff = self.by_ep[cep].cards[ci]
            # keep cards that overlap the cliffhanger moment (camera on the reaction etc.)
            j = ci
            while j + 1 <= segs[-1][2] and self.by_ep[cep].cards[j + 1].start < cliff.end:
                j += 1
            if j < segs[-1][2]:
                tail = self.by_ep[cep].cards[segs[-1][2]].end - self.by_ep[cep].cards[j].end
                if tail > CLIFF_S:
                    out["repairs"].append(f"trimmed end to cliffhanger card ({tail:.1f}s of resolution cut)")
                    segs[-1][2] = j

        # 4b. never cut inside a line: extend each segment over a line that is still being spoken
        for seg in segs:
            ep_cards = self.by_ep[seg[0]].cards
            end = max(c.end for c in self._span(*seg))
            while seg[2] + 1 < len(ep_cards) and ep_cards[seg[2] + 1].start < end - 0.3:
                nxt = ep_cards[seg[2] + 1]
                if nxt.kind == "DIALOGUE" and nxt.end - end > 4:
                    break  # a long new line: better to stop before it than swallow it
                seg[2] += 1
                if nxt.end > end:
                    out["repairs"].append(f"extended end over {nxt.id} to avoid a mid-line cut")
                end = max(end, nxt.end)

        # 5. timestamps from cards
        resolved, total, prev_end = [], 0.0, None
        for ep, a, b in segs:
            span = self._span(ep, a, b)
            st, en = min(c.start for c in span), max(c.end for c in span)
            if prev_end and prev_end[0] == ep and st < prev_end[1] - 0.05:
                return reject("segments overlap or are out of order")
            prev_end = (ep, en)
            total += en - st
            resolved.append({"episode": ep, "first": span[0].id, "last": span[-1].id,
                             "start": round(st, 2), "end": round(en, 2),
                             "start_ts": fmt(st), "end_ts": fmt(en), "seconds": round(en - st, 1)})
            # soft: does the cut land inside a line of dialogue we did not include?
            ids = {c.id for c in span}
            for c in self.by_ep[ep].cards:
                if c.kind == "DIALOGUE" and c.id not in ids:
                    if c.start < st - 0.3 < c.end or c.start < en - 0.3 < c.end:
                        out["flags"].append(f"cut lands inside line {c.id}")
        out["segments_resolved"] = resolved
        out["duration"] = round(total, 1)
        if total < MIN_S:
            return reject(f"too short after repairs ({total:.1f}s)")
        if total > MAX_S:
            return reject(f"too long ({total:.1f}s)")

        # 6. soft standalone checks
        cards = [c for ep, a, b in segs for c in self._span(ep, a, b)]
        out["text"] = "\n".join(c.render(offset=cards[0].start) for c in cards)
        last_ep, _, last_i = segs[-1]
        after = self.by_ep[last_ep].cards[last_i + 1:last_i + 4]
        out["after_cut"] = "\n".join(c.render(offset=cards[0].start) for c in after) or "(episode ends)"
        scenes = {(c.ep, c.scene) for c in cards}
        if len(scenes) > 1:
            out["flags"].append(f"spans {len(scenes)} scenes")
        on_screen = " ".join(c.speaker + " " + (c.text if c.kind != "DIALOGUE" else "") for c in cards).lower()
        spoken = " ".join(c.text for c in cards if c.kind == "DIALOGUE")
        unseen = sorted(n for n in self.names
                        if re.search(rf"\b{n}\b", spoken) and n.lower() not in on_screen)
        if unseen:
            out["flags"].append("mentions characters never shown: " + ", ".join(unseen))
        out["unseen_names"] = unseen
        out["first_episode"] = segs[0][0]
        out["ok"] = True
        return out
