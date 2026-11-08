"""Stage 1 (code only): turn screenplay files into numbered cards.

A card is one timestamped block (CAMERA, ACTION, SOUND FX or a line of
dialogue). Cards get stable IDs like E07-031 so the model can point at them
without ever writing a timestamp.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME = r"(\d+):(\d{2}(?:\.\d+)?)"
BLOCK_HEAD = re.compile(r"^\*\*(.+?):\*\*\s*$")
BLOCK_TIME = re.compile(rf"^Start:\s*\({TIME}\)\s*\|\s*End:\s*\({TIME}\)")
SCENE = re.compile(r"^SCENE\s+(\d+)")
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "SFX", "VO", "MUSIC", "TEXT", "TITLE"}


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


@dataclass
class Card:
    id: str
    ep: int
    scene: int
    kind: str           # CAMERA | ACTION | SOUND FX | DIALOGUE
    speaker: str        # only for DIALOGUE
    direction: str      # e.g. "Cold, Commanding"
    text: str
    start: float
    end: float

    def render(self, offset: float = 0.0) -> str:
        t = f"{fmt(self.start - offset)}-{fmt(self.end - offset)}"
        if self.kind == "DIALOGUE":
            d = f" ({self.direction})" if self.direction else ""
            return f"[{self.id}] {t} {self.speaker}{d}: {self.text}"
        return f"[{self.id}] {t} {self.kind}: {self.text}"


@dataclass
class Episode:
    ep: int
    title: str
    logline: str
    notes: str
    cards: list[Card] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((c.end for c in self.cards), default=0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _header(text: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def parse_episode(text: str, fallback_ep: int) -> Episode:
    num = re.search(r"\d+", _header(text, "EPISODE NUMBER"))
    ep_no = int(num.group()) if num else fallback_ep
    ep = Episode(ep_no, _header(text, "EPISODE NUMBER") or f"Episode {ep_no}",
                 _header(text, "EPISODE LOGLINE"), _header(text, "CONTINUITY NOTES"))

    lines = text.splitlines()
    scene, i = 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if m := SCENE.match(line):
            scene = int(m.group(1))
        elif (h := BLOCK_HEAD.match(line)) and i + 1 < len(lines):
            t = BLOCK_TIME.match(lines[i + 1].strip())
            if t:
                head = h.group(1).strip()
                name, direction = head, ""
                if p := re.match(r"^(.*?)\s*\((.*)\)$", head):
                    name, direction = p.group(1).strip(), p.group(2).strip()
                body = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not BLOCK_HEAD.match(lines[j].strip()):
                    body.append(lines[j].strip())
                    j += 1
                is_dlg = name.upper() not in NON_DIALOGUE
                ep.cards.append(Card(
                    id=f"E{ep_no:02d}-{len(ep.cards) + 1:03d}", ep=ep_no, scene=scene,
                    kind="DIALOGUE" if is_dlg else name.upper(),
                    speaker=name if is_dlg else "", direction=direction,
                    text=" ".join(body).strip(),
                    start=to_sec(*t.group(1, 2)), end=to_sec(*t.group(3, 4))))
                i = j
                continue
        i += 1
    return ep


def _episode_no_from_name(name: str, default: int) -> int:
    m = re.search(r"episode[-_ ]?(\d+)", Path(name).name, re.I)
    return int(m.group(1)) if m else default


def load_show(source) -> list[Episode]:
    """Load from a folder, a .zip path, or zip bytes (Streamlit upload)."""
    files: list[tuple[str, str]] = []
    if isinstance(source, (bytes, bytearray)) or str(source).lower().endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
        for n in zf.namelist():
            if n.lower().endswith(".txt") and not Path(n).name.startswith("._"):
                files.append((n, zf.read(n).decode("utf-8", "replace")))
    else:
        for p in sorted(Path(source).glob("*.txt")):
            files.append((p.name, p.read_text("utf-8", "replace")))
    eps = [parse_episode(t, _episode_no_from_name(n, k + 1)) for k, (n, t) in enumerate(sorted(files))]
    return sorted(eps, key=lambda e: e.ep)


def card_index(episodes: list[Episode]) -> dict[str, Card]:
    return {c.id: c for e in episodes for c in e.cards}
