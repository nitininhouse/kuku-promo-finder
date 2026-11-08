"""Optional stage 6: watch the top clips on the real footage with a video model.

The text pipeline can pick promos before an episode is even rendered. Once video
exists, this stage cuts each top promo out of the episode files (ffmpeg), sends the
clip to Gemini (which sees frames and hears audio), and checks what text can't:
is the hook visible in the first 3 seconds, does the acting sell it, does the cut land,
and does the footage match the script.

Usage: python -m promo_finder.verify_video outputs/<run> path/to/videos --top 20
Videos are matched by episode number in the filename (e.g. episode-012*.mp4).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .models import extract_json

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """You are a promo editor. Watch this vertical-drama ad clip (with sound).
It was cut from the show using the screenplay below. Judge the FOOTAGE, not the script.

Screenplay of the clip (times relative to clip start):
{script}

Return JSON only:
{{"visual_hook_0_4": 0-4,         // does something gripping SHOW or SOUND in the first 3 seconds?
  "first_gripping_second": number, // when the first gripping moment actually happens
  "performance_0_4": 0-4,          // do faces, acting and music sell the emotion?
  "cut_lands_0_4": 0-4,            // does the last second land on an unresolved beat, not mid-word?
  "matches_script": true/false,    // does the footage show what the screenplay says?
  "note": "one line on what the video shows that the script did not"}}"""


def find_video(videos: Path, ep: int) -> Path | None:
    for f in sorted(videos.iterdir()):
        m = re.search(r"(?:episode|ep)[-_ ]?0*(\d+)", f.name, re.I)
        if m and int(m.group(1)) == ep and f.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            return f
    return None


def cut_clip(promo: dict, videos: Path, out: Path) -> Path:
    """Cut every segment, scale down to 480p, and join them into one mp4."""
    parts = []
    for i, s in enumerate(promo["segments_resolved"]):
        src = find_video(videos, s["episode"])
        if not src:
            raise FileNotFoundError(f"no video for episode {s['episode']}")
        part = out.with_name(f"{out.stem}_{i}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(s["start"]), "-to", str(s["end"]),
                        "-i", str(src), "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "30", "-c:a", "aac", "-b:a", "64k", str(part)], check=True)
        parts.append(part)
    if len(parts) == 1:
        parts[0].rename(out)
    else:
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(out)], check=True)
    return out


def ask_gemini(clip: Path, script: str, model: str, key: str) -> dict:
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(clip.read_bytes()).decode()}},
        {"text": PROMPT.format(script=script)}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    r = requests.post(URL.format(model=model), json=body, timeout=180, headers={"x-goog-api-key": key})
    r.raise_for_status()
    text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    return extract_json(text)


def verify(run_dir: Path, videos: Path, top: int = 20, model: str = "gemini-3.8-flash") -> list[dict]:
    key = os.environ["GEMINI_API_KEY"]
    promos = json.loads((run_dir / "promos.json").read_text())[:top]
    tmp = Path(tempfile.mkdtemp())

    def one(p):
        try:
            clip = cut_clip(p, videos, tmp / f"{p['cid']}.mp4")
            v = ask_gemini(clip, p["text"], model, key)
            text_hook = p["scores"].get("hook", 0)
            v["hook_gap"] = round(text_hook - v.get("visual_hook_0_4", 0), 1)  # script promised more than video shows
            return {"rank": p["rank"], "cid": p["cid"], **v}
        except Exception as e:
            return {"rank": p["rank"], "cid": p["cid"], "error": str(e)[:200]}

    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(one, promos))
    (run_dir / "video_check.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("videos")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--model", default="gemini-3.8-flash")
    a = ap.parse_args()
    for r in verify(Path(a.run_dir), Path(a.videos), a.top, a.model):
        print(r)
