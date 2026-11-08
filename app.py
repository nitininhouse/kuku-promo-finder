"""CliffCut: upload a screenplay ZIP, get an edit list of promo clips with exact in and out points."""
import csv
import html
import io
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

import streamlit as st

from promo_finder.parse import fmt, load_show
from promo_finder.pipeline import run

st.set_page_config(page_title="CliffCut", page_icon="✂", layout="wide", initial_sidebar_state="collapsed")

MAX_EPS = 10  # per run: the app runs on the owner's keys, so one run can't drain them

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, input, label {font-family:'IBM Plex Sans', sans-serif;}
[data-testid="stSidebar"], [data-testid="collapsedControl"], footer, #MainMenu {display:none;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.4rem; max-width:1240px;}
.bar {display:flex; align-items:baseline; gap:18px; border-bottom:1px solid #d6d6d6; padding-bottom:10px; margin-bottom:18px;}
.bar .name {font-weight:600; font-size:20px; letter-spacing:-0.2px;}
.bar .name i {font-style:normal; color:#c8102e;}
.bar .what {color:#5c5c5c; font-size:14px;}
.bar .right {margin-left:auto; color:#5c5c5c; font:12px 'IBM Plex Mono', monospace;}
.lead {max-width:760px; color:#2b2b2b; font-size:15px; line-height:1.55; margin:0 0 14px;}
.k {font:12px 'IBM Plex Mono', monospace; color:#5c5c5c; text-transform:uppercase; letter-spacing:.6px;}
.meta {font:13px 'IBM Plex Mono', monospace; color:#3a3a3a; margin:6px 0 14px;}
table.edl {border-collapse:collapse; width:100%; font-size:14px;}
table.edl th {text-align:left; font:500 11px 'IBM Plex Mono', monospace; color:#5c5c5c; text-transform:uppercase;
  letter-spacing:.5px; border-bottom:1px solid #141414; padding:6px 8px;}
table.edl td {border-bottom:1px solid #e1e1e1; padding:9px 8px; vertical-align:top;}
table.edl td.tc {font:13px 'IBM Plex Mono', monospace; white-space:nowrap;}
table.edl td.n {font:500 14px 'IBM Plex Mono', monospace; width:34px;}
table.edl td.sc {font:500 14px 'IBM Plex Mono', monospace; text-align:right; white-space:nowrap;}
table.edl .cut {color:#c8102e;}
table.edl .why {color:#5c5c5c; font-size:13px; margin-top:4px;}
.tl {margin:8px 0 4px;}
.tl .row {display:grid; grid-template-columns:64px 1fr 70px; align-items:center; gap:10px; margin:5px 0;}
.tl .ep {font:12px 'IBM Plex Mono', monospace; color:#3a3a3a;}
.tl .track {position:relative; height:22px; background:#e9e9e9; border:1px solid #d6d6d6;}
.tl .clip {position:absolute; top:2px; bottom:2px; background:#141414; color:#fff; font:10px 'IBM Plex Mono', monospace;
  display:flex; align-items:center; padding-left:3px; overflow:hidden;}
.tl .clip.top {background:#c8102e; z-index:2; outline:1px solid #fafafa;}
.tl .len {font:11px 'IBM Plex Mono', monospace; color:#5c5c5c; text-align:right;}
.legend {font-size:12px; color:#5c5c5c; margin-bottom:6px;}
.legend b {display:inline-block; width:10px; height:10px; vertical-align:-1px; margin:0 4px 0 12px;}
.method td, .method th {padding:6px 10px; border-bottom:1px solid #e1e1e1; font-size:14px; text-align:left;}
.method th {font:500 11px 'IBM Plex Mono', monospace; color:#5c5c5c; text-transform:uppercase; border-bottom:1px solid #141414;}
.method td.v {font-family:'IBM Plex Mono', monospace;}
.stButton button[kind="primary"] {border-radius:2px; background:#141414; border-color:#141414;}
.stButton button[kind="primary"]:hover {background:#c8102e; border-color:#c8102e;}
.stTabs [data-baseweb="tab"] {font-family:'IBM Plex Sans', sans-serif; font-size:14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="bar"><span class="name">Cliff<i>Cut</i></span>'
            '<span class="what">promo finder for microdrama screenplays</span>'
            '<span class="right">screenplay in · edit list out</span></div>', unsafe_allow_html=True)
st.markdown('<p class="lead">Upload a show\'s episode screenplays. CliffCut returns an edit list of 20-60 second '
            'promo cuts: each opens on a hook in the first three seconds, ends on an unresolved beat, and plays for '
            'someone who has never seen the show. A model proposes cuts by line number, code checks every quote and '
            'sets every timecode, and a second model ranks them.</p>', unsafe_allow_html=True)


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


cfg = tomllib.loads(Path("config.toml").read_text())
cfg.pop("extra_finders", None)
cfg["scorer"].pop("panel", None)
# speed for an interactive page: the full CLI run also does cross-episode stitching and a head-to-head
# tournament; here each episode's top slot comes from the finder's own #1 and the judge orders the rest
cfg["tournament"] = {"k": 0}
cfg["stitch"] = {"enabled": False}
cfg["finder"] = {"provider": "gemini", "model": "gemini-3.8-flash", "api_key": secret("GEMINI_API_KEY"), "workers": 6}
judge = {"provider": "gemini", "model": "gemini-3.8-flash", "api_key": secret("GEMINI_API_KEY"), "workers": 6}
if secret("DEEPSEEK_API_KEY"):  # judge from a different model family than the finder
    judge = {"provider": "deepseek", "model": "deepseek-flash", "api_key": secret("DEEPSEEK_API_KEY"), "workers": 4}
cfg["scorer"].update(judge="llm", llm=judge)
ready = bool(secret("GEMINI_API_KEY"))

# ---------- input ----------
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.markdown('<div class="k">Screenplays</div>', unsafe_allow_html=True)
    up = st.file_uploader("ZIP of episode .txt files", type="zip", label_visibility="collapsed")
eps = load_show(up.getvalue()) if up else []
with c2:
    if eps:
        st.markdown('<div class="k">Loaded</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta">{len(eps)} episodes · {sum(len(e.cards) for e in eps):,} timed lines · '
                    f'{fmt(sum(e.duration for e in eps)).split(".")[0]} runtime</div>', unsafe_allow_html=True)
        nums = [e.ep for e in eps]
        default_hi = nums[min(len(nums), MAX_EPS) - 1]
        lo, hi = (st.select_slider(f"Episodes (up to {MAX_EPS} per run)", options=nums, value=(nums[0], default_hi))
                  if len(nums) > 1 else (nums[0], nums[0]))
        if st.button("Find promos", type="primary", disabled=not ready):
            chosen = [e for e in eps if lo <= e.ep <= hi][:MAX_EPS]
            out = Path(tempfile.mkdtemp()) / time.strftime("%Y%m%d-%H%M%S")
            with st.status(f"Working on episodes {chosen[0].ep}-{chosen[-1].ep}", expanded=True) as status:
                st.session_state.result = run(chosen, cfg, out, log=st.write)
                st.session_state.out = out
                st.session_state.eps = {e.ep: e.duration for e in chosen}
                status.update(label="Done", state="complete", expanded=False)
        if not ready:
            st.error("The service keys are not configured on this deployment.")


def segs_text(c):
    return "<br>".join(f'Ep {s["episode"]:>2}  {s["start_ts"]} → <span class="cut">{s["end_ts"]}</span>'
                       for s in c["segments_resolved"])


res = st.session_state.get("result")
tabs = st.tabs(["Edit list", "Timeline", "Rejected", "Method and evidence", "Export"])

with tabs[0]:
    if not res:
        st.markdown('<p class="lead">Upload a ZIP of episode screenplays (one .txt per episode, with timed '
                    'SCENE / CAMERA / ACTION / dialogue blocks), choose up to ten episodes, and press '
                    '<b>Find promos</b>. A run takes about two minutes.</p>', unsafe_allow_html=True)
    else:
        m = res["meta"]
        st.markdown(f'<div class="meta">{m["candidates"]} proposed · {m["rejected"]} rejected by the checker · '
                    f'{m["ranked_after_dedupe"]} after removing overlaps · {m["seconds"]:.0f} s</div>',
                    unsafe_allow_html=True)
        e = html.escape
        rows = []
        for c in res["ranked"][:40]:
            s = c["scores"]
            rows.append(
                f'<tr><td class="n">{c["rank"]:02d}</td><td class="tc">{segs_text(c)}</td>'
                f'<td class="tc">{c["duration"]:.1f}s</td>'
                f'<td>{e(c["hook_quote"])}</td><td class="cut">{e(c["cliffhanger_quote"])}'
                f'<div class="why">{e(c.get("why", ""))}</div></td>'
                f'<td class="sc">{c["score"]:.0f}<div class="why">H{s.get("hook", 0):.0f} C{s.get("cliffhanger", 0):.0f} '
                f'S{s.get("standalone", 0):.0f} B{s.get("boundary", 0):.0f}</div></td></tr>')
        st.markdown('<table class="edl"><tr><th>#</th><th>In → out</th><th>Dur</th><th>Opens on (hook)</th>'
                    '<th>Cuts on (cliffhanger)</th><th>Score</th></tr>' + "".join(rows) + "</table>",
                    unsafe_allow_html=True)
        st.caption("Score out of 100. H / C / S / B: hook, cliffhanger, standalone, clean cut, each 0-4.")
        pick = st.selectbox("Read the screenplay of a cut", [f'{c["rank"]:02d}  {c.get("title") or ""}'
                                                              for c in res["ranked"][:40]])
        if pick:
            c = res["ranked"][int(pick.split()[0]) - 1]
            flags = c["flags"] + c["repairs"]
            if flags:
                st.caption("Checker notes: " + "; ".join(flags))
            st.code(c["text"], language=None, wrap_lines=True)

with tabs[1]:
    if not res:
        st.markdown('<p class="lead">After a run, each episode is drawn to scale with its promo cuts on it.</p>',
                    unsafe_allow_html=True)
    else:
        durs = st.session_state.eps
        best = {}
        for c in res["ranked"]:
            best.setdefault(c["segments_resolved"][0]["episode"], c["rank"])
        rows = []
        for ep, d in sorted(durs.items()):
            clips = ""
            for c in res["ranked"]:
                for s in c["segments_resolved"]:
                    if s["episode"] == ep and d:
                        left, width = s["start"] / d * 100, max(0.8, (s["end"] - s["start"]) / d * 100)
                        top = " top" if best.get(ep) == c["rank"] else ""
                        clips += (f'<div class="clip{top}" style="left:{left:.2f}%;width:{width:.2f}%" '
                                  f'title="#{c["rank"]} {s["start_ts"]}-{s["end_ts"]}">{c["rank"]}</div>')
            rows.append(f'<div class="row"><span class="ep">Ep {ep}</span><div class="track">{clips}</div>'
                        f'<span class="len">{fmt(d).split(".")[0]}</span></div>')
        st.markdown('<div class="legend">Cuts drawn to scale<b style="background:#c8102e"></b>episode\'s best'
                    '<b style="background:#141414"></b>other cuts (number = rank)</div>'
                    f'<div class="tl">{"".join(rows)}</div>', unsafe_allow_html=True)

with tabs[2]:
    if not res:
        st.markdown('<p class="lead">Every proposal the checker throws out is listed here with the reason: '
                    'a quote that is not in the script, a cut outside 20-60 seconds, overlapping segments.</p>',
                    unsafe_allow_html=True)
    elif not res["rejected"]:
        st.markdown('<p class="lead">Nothing was rejected in this run.</p>', unsafe_allow_html=True)
    else:
        st.dataframe([{"proposal": c["cid"], "reason": c["reject_reason"], "hook": c.get("hook_quote")}
                      for c in res["rejected"]], width="stretch", hide_index=True)

with tabs[3]:
    st.markdown("""
<table class="method">
<tr><th>Step</th><th>Done by</th><th>What happens</th></tr>
<tr><td>1 Parse</td><td>code</td><td>Every timed block becomes a numbered line (E12-032).</td></tr>
<tr><td>2 Find</td><td>Gemini</td><td>Proposes cuts as line ranges plus the exact hook and cliffhanger words. It never writes a time.</td></tr>
<tr><td>3 Check</td><td>code</td><td>Verifies every quote, sets timecodes, trims to the hook and the cliffhanger, never cuts mid-line, enforces 20-60 s.</td></tr>
<tr><td>4 Judge</td><td>DeepSeek</td><td>A model from a different family scores hook, cliffhanger, standalone and clean cut.</td></tr>
<tr><td>5 Rank</td><td>code</td><td>Each episode's top slot goes to the finder's own first pick (it read the whole episode); the judge's scores order everything else.</td></tr>
</table>""", unsafe_allow_html=True)
    st.caption("This page runs the fast path. The full command-line run behind the numbers below also stitches "
               "cross-episode promos and re-ranks the top cuts head to head, asking each pair in both orders.")
    st.markdown("&nbsp;")
    st.markdown('<div class="k">Tested on a full 50-episode show</div>', unsafe_allow_html=True)
    st.markdown("""
<table class="method">
<tr><th>Answer key: found / as the episode's first pick</th><th>Keyword rules</th><th>Rival A</th><th>Rival B</th><th>CliffCut</th></tr>
<tr><td>8 promos marked blind</td><td class="v">1 / 1</td><td class="v">4 / 4</td><td class="v">1 / 1</td><td class="v"><b>8 / 5</b></td></tr>
<tr><td>Kuku's own episode titles (45)</td><td class="v">20 / 19</td><td class="v">34 / 28</td><td class="v">9 / 8</td><td class="v"><b>41 / 29</b></td></tr>
<tr><td>Real YouTube Shorts promos of the show (21)</td><td class="v">10 / 8</td><td class="v">21 / 17</td><td class="v">8 / 8</td><td class="v"><b>21 / 19</b></td></tr>
</table>""", unsafe_allow_html=True)
    st.caption("Rivals A and B are public solutions to the same assignment, scored from their published 50-episode "
               "rankings with the same evaluator. Known limits: the most-viewed real promo was found but ranked low; "
               "the judge changes 10-35% of head-to-head verdicts when the order is swapped, so both orders are asked.")
    p = Path("eval/meta_ads.csv")
    if p.exists():
        st.markdown('<div class="k" style="margin-top:14px">Paid Meta ads for the show, for comparison</div>',
                    unsafe_allow_html=True)
        rows = [r for r in csv.DictReader(open(p)) if r["show_on_screen"]]
        st.dataframe([{"ad": r["url"], "running": r["run_dates"], "sec": r["seconds"], "opens on": r["opening_3s"]}
                      for r in rows], width="stretch", hide_index=True,
                     column_config={"ad": st.column_config.LinkColumn("ad")})

with tabs[4]:
    if not res:
        st.markdown('<p class="lead">After a run: the edit list as CSV, and every output (unedited model replies, '
                    'rejections, an HTML report) as a ZIP.</p>', unsafe_allow_html=True)
    else:
        out = Path(st.session_state.out)
        st.download_button("Edit list (CSV)", (out / "promos.csv").read_bytes(), "cliffcut_edit_list.csv",
                           type="primary")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in out.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        st.download_button("All outputs (ZIP)", buf.getvalue(), "cliffcut_outputs.zip")
