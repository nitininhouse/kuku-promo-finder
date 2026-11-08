"""Streamlit front end: upload a screenplay ZIP, pick a model, get ranked promos."""
import io
import json
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

import streamlit as st

from promo_finder.models import PRESETS
from promo_finder.parse import load_show
from promo_finder.pipeline import run

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Reads a microdrama screenplay and finds 20-60s promo clips: hook in 3s, cliffhanger ending, "
           "makes sense to a stranger. The model picks card IDs; code sets every timestamp.")

cfg = tomllib.loads(Path("config.toml").read_text())


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Finder", list(PRESETS) + ["baseline"],
                            help="baseline = keyword rules, no model, no key")
    if provider != "baseline":
        p = PRESETS[provider]
        model = st.text_input("Model name", p["model"])
        base_url = st.text_input("Base URL", p["base_url"],
                                 help="Any OpenAI-compatible endpoint works here")
        key = st.text_input(f"API key ({p['key_env'] or 'none'})", type="password",
                            value=secret(p["key_env"]) if p["key_env"] else "")
        cfg["finder"] = {"provider": provider, "model": model, "base_url": base_url,
                         "api_key": key, "workers": 2}
    else:
        cfg["finder"] = {"provider": "baseline"}
    st.header("Judge")
    judge = st.selectbox("Scorer", ["jev", "llm", "heuristic"],
                         help="jev = TypeSafe Jev probability scores; llm = same rubric via the finder model")
    cfg["scorer"]["judge"] = judge
    if judge == "jev":
        cfg["scorer"]["api_key"] = st.text_input("TypeSafe API key", type="password",
                                                 value=secret("TYPESAFE_API_KEY"))
    st.caption("Keys stay in this browser session and are never written to disk.")

src = st.radio("Input", ["Upload a ZIP of episode .txt files", "Demo show (original sample)"], horizontal=True)
data = None
if src.startswith("Upload"):
    up = st.file_uploader("Screenplay ZIP", type="zip")
    if up:
        data = up.getvalue()
else:
    data = "samples/demo_show"

if data:
    eps = load_show(data)
    st.write(f"**{len(eps)} episodes**, {sum(len(e.cards) for e in eps)} timestamped blocks, "
             f"{sum(e.duration for e in eps) / 60:.1f} min")
    nums = [e.ep for e in eps]
    lo, hi = st.select_slider("Episodes", options=nums, value=(nums[0], nums[-1])) if len(nums) > 1 else (nums[0], nums[0])
    cfg["candidates_per_episode"] = st.slider("Candidates per episode", 1, 8, cfg["candidates_per_episode"])
    if st.button("Find promos", type="primary"):
        out = Path(tempfile.mkdtemp()) / time.strftime("%Y%m%d-%H%M%S")
        logbox = st.empty()
        lines = []

        def log(m):
            lines.append(m)
            logbox.code("\n".join(lines[-12:]))

        with st.spinner("Running..."):
            st.session_state.result = run([e for e in eps if lo <= e.ep <= hi], cfg, out, log=log)
            st.session_state.out = out

res = st.session_state.get("result")
if res:
    m = res["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", m["candidates"])
    c2.metric("Passed checks", m["passed"])
    c3.metric("Rejected", m["rejected"])
    c4.metric("Ranked (deduped)", m["ranked_after_dedupe"])

    rows = [{"#": c["rank"], "score": c["score"], "len (s)": c["duration"],
             "segments": " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in c["segments_resolved"]),
             "hook": c["hook_quote"], "cliffhanger": c["cliffhanger_quote"], "why": c["why"]}
            for c in res["ranked"]]
    st.dataframe(rows, width='stretch', hide_index=True)

    for c in res["ranked"][:20]:
        s = c["scores"]
        with st.expander(f"#{c['rank']}  {c['score']:.0f}/100  {c.get('title', '')}"):
            st.write(f"**Why it works:** {c['why']}")
            st.write(f"**Why this rank:** {c['rank_reason']}")
            st.write({k: s.get(k) for k in ("hook", "cliffhanger", "standalone", "boundary", "hook_type")}
                     | {k: v for k, v in s.items() if k.endswith("_conf")})
            if c["flags"] or c["repairs"]:
                st.warning("; ".join(c["flags"] + c["repairs"]))
            if c.get("needs_context"):
                st.info(f"New viewer may miss: {c['needs_context']}")
            st.code(c["text"], language=None)

    if res["rejected"]:
        with st.expander(f"Rejected by the checker ({len(res['rejected'])})"):
            st.dataframe([{"cid": c["cid"], "reason": c["reject_reason"], "hook": c.get("hook_quote")}
                          for c in res["rejected"]], width='stretch', hide_index=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in Path(st.session_state.out).rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(st.session_state.out))
    st.download_button("Download all outputs (zip)", buf.getvalue(), "promo_outputs.zip")
