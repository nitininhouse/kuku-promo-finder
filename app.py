"""CliffCut: upload a screenplay ZIP, get ranked promo clips with exact cut points."""
import csv
import html
import io
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

import streamlit as st

from promo_finder.models import PRESETS
from promo_finder.parse import load_show
from promo_finder.pipeline import run

st.set_page_config(page_title="CliffCut · promos from the screenplay", page_icon="✂️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
.cc-hero {display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap;
  padding:22px 26px; border-radius:18px; margin-bottom:18px;
  background: radial-gradient(120% 140% at 0% 0%, #3a1220 0%, #1a1620 55%, #0e0c11 100%);
  border:1px solid #2c2433;}
.cc-logo {font-size:44px; font-weight:800; letter-spacing:-1.5px; line-height:1; color:#fff;}
.cc-logo span {color:#ff4d5a;}
.cc-tag {color:#c9c1d3; font-size:15px; margin-top:8px; max-width:560px;}
.cc-steps {display:flex; gap:8px; flex-wrap:wrap;}
.cc-step {font-size:12px; padding:6px 10px; border-radius:999px; background:#241d2b; color:#d8d0e2; border:1px solid #352b3e;}
.cc-step b {color:#ffb347;}
.cc-card {border:1px solid #2c2433; background:#16121b; border-radius:16px; padding:16px 18px; margin-bottom:14px;}
.cc-top {display:flex; align-items:center; gap:12px; margin-bottom:10px;}
.cc-rank {font-weight:800; font-size:13px; background:#ff4d5a; color:#fff; border-radius:8px; padding:3px 8px;}
.cc-title {font-weight:700; font-size:16px; color:#fff; flex:1;}
.cc-score {font-weight:800; font-size:20px; color:#ffb347;}
.cc-score small {font-size:11px; color:#9a90a6; font-weight:500;}
.cc-chips {display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px;}
.cc-chip {font: 12px/1.2 ui-monospace, Menlo, monospace; background:#221b29; border:1px solid #3a2f44; color:#e6def0;
  padding:4px 8px; border-radius:8px;}
.cc-q {border-left:3px solid; padding:6px 10px; margin:6px 0; border-radius:0 8px 8px 0; font-size:14px; color:#efe9f5;}
.cc-q .lbl {display:block; font-size:10px; letter-spacing:1.2px; font-weight:700; margin-bottom:2px;}
.cc-hook {border-color:#ffb347; background:#2a2016;} .cc-hook .lbl {color:#ffb347;}
.cc-cliff {border-color:#ff4d5a; background:#2a161b;} .cc-cliff .lbl {color:#ff4d5a;}
.cc-why {color:#bdb3c8; font-size:13px; margin:8px 0 10px;}
.cc-bars {display:grid; grid-template-columns: repeat(4, 1fr); gap:10px;}
.cc-bar .n {font-size:11px; color:#9a90a6; display:flex; justify-content:space-between;}
.cc-bar .t {height:6px; background:#2a2231; border-radius:6px; overflow:hidden; margin-top:3px;}
.cc-bar .f {height:100%; background:linear-gradient(90deg,#ffb347,#ff4d5a);}
.cc-flags {font-size:12px; color:#e0a35a; margin-top:8px;}
.cc-map {display:grid; grid-template-columns: repeat(auto-fill, minmax(34px, 1fr)); gap:5px;}
.cc-ep {height:58px; border-radius:7px; background:#1c1722; border:1px solid #2c2433; position:relative;
  display:flex; align-items:flex-end; justify-content:center; font-size:10px; color:#7f7589; padding-bottom:3px;}
.cc-ep .h {position:absolute; left:3px; right:3px; bottom:16px; border-radius:4px; background:#ff4d5a;}
.cc-legend {font-size:12px; color:#9a90a6; margin:6px 0 14px;}
.cc-stat {border:1px solid #2c2433; background:#16121b; border-radius:14px; padding:14px 16px;}
.cc-stat .v {font-size:26px; font-weight:800; color:#fff;} .cc-stat .k {font-size:12px; color:#9a90a6;}
.cc-proof td, .cc-proof th {padding:7px 10px; border-bottom:1px solid #2c2433; font-size:14px; text-align:left;}
.cc-proof th {color:#9a90a6; font-weight:600; font-size:12px;}
.cc-proof b {color:#7ee0a1;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="cc-hero">
  <div><div class="cc-logo">Cliff<span>Cut</span></div>
  <div class="cc-tag">Reads a microdrama screenplay and returns ranked 20-60 s promo clips with exact cut points:
  a hook in the first 3 seconds, a cliffhanger at the cut, and a story a stranger can follow.</div></div>
  <div class="cc-steps">
    <div class="cc-step"><b>1</b> code numbers every line</div>
    <div class="cc-step"><b>2</b> AI proposes by line ID</div>
    <div class="cc-step"><b>3</b> code checks quotes &amp; times</div>
    <div class="cc-step"><b>4</b> a second AI judges, head-to-head</div>
  </div>
</div>""", unsafe_allow_html=True)

cfg = tomllib.loads(Path("config.toml").read_text())
cfg.pop("extra_finders", None)  # the app runs one finder and one judge; the CLI config can pool several
cfg["scorer"].pop("panel", None)


MAX_SHARED_EPS = 10  # cap per run when the owner's keys are used, so a public link can't drain them


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


with st.sidebar:
    st.markdown("### ⚙️ Engine")
    has_keys = bool(secret("GEMINI_API_KEY"))
    modes = (["AI (ready to use)"] if has_keys else []) + ["Free (no key)", "AI (bring your key)"]
    mode = st.radio("Mode", modes, help="AI (ready to use) runs the full pipeline on the owner's keys, which stay on "
                                         "the server. Free mode uses keyword rules only.")
    if mode.startswith("AI (ready"):
        cfg["finder"] = {"provider": "gemini", "model": "gemini-3.8-flash",
                         "api_key": secret("GEMINI_API_KEY"), "workers": 6}
        judge_cfg = {"provider": "gemini", "model": "gemini-3.8-flash", "api_key": secret("GEMINI_API_KEY"),
                     "workers": 6}
        if secret("DEEPSEEK_API_KEY"):  # a judge from another model family when available
            judge_cfg = {"provider": "deepseek", "model": "deepseek-flash",
                         "api_key": secret("DEEPSEEK_API_KEY"), "workers": 4}
        cfg["scorer"].update(judge="llm", llm=judge_cfg)
        st.success(f"Gemini finds, {judge_cfg['provider'].title()} judges. Up to {MAX_SHARED_EPS} episodes per run.")
    elif mode.startswith("Free"):
        cfg["finder"] = {"provider": "baseline"}
        cfg["scorer"]["judge"] = "heuristic"
    else:
        names = ["deepseek", "gemini", "anthropic", "groq", "openrouter", "ollama"]
        provider = st.selectbox("Finder model", names)
        p = PRESETS[provider]
        with st.expander("Finder settings", expanded=True):
            model = st.text_input("Model", p["model"])
            base_url = st.text_input("Base URL", p["base_url"], help="Any OpenAI-compatible endpoint works")
            key = st.text_input("API key", type="password", placeholder=p["key_env"] or "not needed")
        cfg["finder"] = {"provider": provider, "model": model, "base_url": base_url, "api_key": key, "workers": 4}
        judge = st.selectbox("Judge", ["Different model (recommended)", "Same model as finder", "TypeSafe Jev"])
        if judge.startswith("Different"):
            jp = st.selectbox("Judge model", [n for n in names if n != provider])
            with st.expander("Judge settings", expanded=True):
                jm = st.text_input("Judge model name", PRESETS[jp]["model"])
                jk = st.text_input("Judge API key", type="password", placeholder=PRESETS[jp]["key_env"])
            cfg["scorer"].update(judge="llm", llm={"provider": jp, "model": jm, "api_key": jk, "workers": 4})
        elif judge.startswith("Same"):
            cfg["scorer"].update(judge="llm", llm=cfg["finder"])
        else:
            cfg["scorer"].update(judge="jev", api_key=st.text_input("TypeSafe API key", type="password"))
    st.caption("🔒 Keys never reach the browser: the owner's stay on the server, yours stay in this session.")
    st.divider()
    cfg["candidates_per_episode"] = st.slider("Candidates per episode", 1, 8, cfg["candidates_per_episode"])

# ---------- input ----------
left, right = st.columns([3, 2])
with left:
    src = st.radio("Screenplay", ["Use the demo episode", "Upload a ZIP of episode .txt files"], horizontal=True,
                   label_visibility="collapsed")
    data = "samples/demo_show" if src.startswith("Use") else None
    if data is None:
        up = st.file_uploader("Screenplay ZIP", type="zip", label_visibility="collapsed")
        data = up.getvalue() if up else None
eps = load_show(data) if data else []
with right:
    if eps:
        a, b, c = st.columns(3)
        a.markdown(f'<div class="cc-stat"><div class="v">{len(eps)}</div><div class="k">episodes</div></div>',
                   unsafe_allow_html=True)
        b.markdown(f'<div class="cc-stat"><div class="v">{sum(len(e.cards) for e in eps):,}</div>'
                   f'<div class="k">timed lines</div></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="cc-stat"><div class="v">{sum(e.duration for e in eps) / 60:.0f}m</div>'
                   f'<div class="k">runtime</div></div>', unsafe_allow_html=True)

if eps:
    nums = [e.ep for e in eps]
    lo, hi = (st.select_slider("Episodes", options=nums, value=(nums[0], nums[-1])) if len(nums) > 1
              else (nums[0], nums[0]))
    if st.button("✂️  Find promos", type="primary", use_container_width=False):
        out = Path(tempfile.mkdtemp()) / time.strftime("%Y%m%d-%H%M%S")
        with st.status("Cutting...", expanded=True) as status:
            chosen = [e for e in eps if lo <= e.ep <= hi]
            if mode.startswith("AI (ready") and len(chosen) > MAX_SHARED_EPS:
                st.write(f"Shared keys: running the first {MAX_SHARED_EPS} of {len(chosen)} episodes.")
                chosen = chosen[:MAX_SHARED_EPS]
            st.session_state.result = run(chosen, cfg, out, log=st.write)
            st.session_state.out = out
            st.session_state.n_eps = (lo, hi)
            status.update(label="Done", state="complete", expanded=False)


def bar(name, v):
    v = max(0.0, min(4.0, float(v or 0)))
    return (f'<div class="cc-bar"><div class="n"><span>{name}</span><span>{v:.1f}/4</span></div>'
            f'<div class="t"><div class="f" style="width:{v / 4 * 100:.0f}%"></div></div></div>')


def card(c):
    e = html.escape
    s = c["scores"]
    chips = "".join(f'<span class="cc-chip">Ep {x["episode"]} · {x["start_ts"]} → {x["end_ts"]}</span>'
                    for x in c["segments_resolved"])
    chips += f'<span class="cc-chip">{c["duration"]:.0f} s</span>'
    if s.get("hook_type"):
        chips += f'<span class="cc-chip">{e(str(s["hook_type"]))}</span>'
    flags = c["flags"] + c["repairs"]
    return f"""<div class="cc-card">
 <div class="cc-top"><span class="cc-rank">#{c['rank']}</span><span class="cc-title">{e(c.get('title') or 'Promo')}</span>
 <span class="cc-score">{c['score']:.0f}<small>/100</small></span></div>
 <div class="cc-chips">{chips}</div>
 <div class="cc-q cc-hook"><span class="lbl">HOOK · FIRST 3 SECONDS</span>{e(c['hook_quote'])}</div>
 <div class="cc-q cc-cliff"><span class="lbl">CLIFFHANGER · THE CUT</span>{e(c['cliffhanger_quote'])}</div>
 <div class="cc-why">{e(c.get('why', ''))}</div>
 <div class="cc-bars">{bar('Hook', s.get('hook'))}{bar('Cliffhanger', s.get('cliffhanger'))}
 {bar('Standalone', s.get('standalone'))}{bar('Clean cut', s.get('boundary'))}</div>
 {f'<div class="cc-flags">⚑ {e("; ".join(flags))}</div>' if flags else ''}
</div>"""


res = st.session_state.get("result")
tabs = st.tabs(["🎬 Promos", "🗺️ Season map", "🚫 Rejected", "📊 Proof", "⬇️ Export"])

with tabs[0]:
    if not res:
        st.info("Pick the demo episode (or upload a ZIP) and press **Find promos**. Free mode needs no key.")
    else:
        m = res["meta"]
        st.caption(f"{m['candidates']} candidates · {m['passed']} passed the checker · {m['rejected']} rejected · "
                   f"{m['ranked_after_dedupe']} after removing overlaps · {m['seconds']} s")
        cols = st.columns(2)
        for i, c in enumerate(res["ranked"][:30]):
            with cols[i % 2]:
                st.markdown(card(c), unsafe_allow_html=True)
                with st.expander("Screenplay of this cut"):
                    st.code(c["text"], language=None)

with tabs[1]:
    if not res:
        st.info("After a run, every episode appears here with its best promo score, so you can see where the "
                "strong moments sit in the season.")
    else:
        best = {}
        for c in res["ranked"]:
            for x in c["segments_resolved"]:
                best[x["episode"]] = max(best.get(x["episode"], 0), c["score"])
        lo, hi = st.session_state.n_eps
        cells = "".join(f'<div class="cc-ep" title="Episode {n}: best {best.get(n, 0):.0f}/100">'
                        f'<div class="h" style="height:{max(2, best.get(n, 0) * 0.38):.0f}px;'
                        f'opacity:{0.25 + best.get(n, 0) / 140:.2f}"></div>{n}</div>' for n in range(lo, hi + 1))
        st.markdown(f'<div class="cc-legend">Bar height = best promo score in that episode.</div>'
                    f'<div class="cc-map">{cells}</div>', unsafe_allow_html=True)

with tabs[2]:
    if not res:
        st.info("Everything the checker throws out (invented quotes, wrong lengths, overlapping cuts) is listed here.")
    elif not res["rejected"]:
        st.success("Nothing rejected in this run.")
    else:
        st.dataframe([{"candidate": c["cid"], "why rejected": c["reject_reason"], "hook": c.get("hook_quote")}
                      for c in res["rejected"]], width="stretch", hide_index=True)

with tabs[3]:
    st.markdown("#### Tested on the full 50-episode show against moments real people chose")
    st.markdown("""
<table class="cc-proof">
<tr><th>Answer key</th><th>Who picked the moments</th><th>Keyword rules</th><th>CliffCut</th></tr>
<tr><td>8 promos marked blind</td><td>hand-marked before seeing output</td><td>1 / 8</td><td><b>8 / 8</b></td></tr>
<tr><td>Kuku's own episode titles</td><td>Kuku editors</td><td>20 / 45</td><td><b>42 / 45</b></td></tr>
<tr><td>Real YouTube Shorts promos of the show</td><td>clippers + real audiences</td><td>10 / 21</td><td><b>21 / 21</b></td></tr>
<tr><td>Same, weighted by views</td><td>real audiences</td><td>27%</td><td><b>100%</b></td></tr>
</table>""", unsafe_allow_html=True)
    st.markdown("&nbsp;")
    st.markdown("**Paid ads agree.** 20 Meta Ad Library creatives for the show open most often on the episode-18 fire "
                "and the episode-3 wedding expulsion: the #1 picks of CliffCut's two main configurations.")
    p = Path("eval/meta_ads.csv")
    if p.exists():
        rows = [r for r in csv.DictReader(open(p)) if r["show_on_screen"]]
        st.dataframe([{"ad": r["url"], "running": r["run_dates"], "sec": r["seconds"], "versions": r["versions"],
                       "opens on": r["opening_3s"][:110]} for r in rows],
                     width="stretch", hide_index=True, column_config={"ad": st.column_config.LinkColumn("ad")})
    st.caption("Honest limits: it finds the right moments more reliably than it ranks them (the most-viewed real "
               "promo was found but ranked low), and the AI judge flips 23-27% of A/B verdicts when the order is "
               "swapped, so every pair is asked both ways.")

with tabs[4]:
    if not res:
        st.info("Run first, then download every output: ranked CSV/JSON/report, raw model replies, rejections.")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in Path(st.session_state.out).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(st.session_state.out))
        st.download_button("Download all outputs (zip)", buf.getvalue(), "cliffcut_outputs.zip", type="primary")
        st.download_button("Ranked promos (CSV)", (Path(st.session_state.out) / "promos.csv").read_bytes(),
                           "promos.csv")
