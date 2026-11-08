"""A standalone HTML report for every run: open report.html in any browser, no app needed."""
from __future__ import annotations

import html
from pathlib import Path

CSS = """
body{margin:0;background:#0e0c11;color:#ece8f1;font:15px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 18px 60px}
.hero{padding:22px 26px;border-radius:18px;border:1px solid #2c2433;margin-bottom:18px;
 background:radial-gradient(120% 140% at 0% 0%,#3a1220 0%,#1a1620 55%,#0e0c11 100%)}
.logo{font-size:38px;font-weight:800;letter-spacing:-1.3px}.logo span{color:#ff4d5a}
.sub{color:#c9c1d3;margin-top:6px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0 22px}
.stat{border:1px solid #2c2433;background:#16121b;border-radius:14px;padding:12px 14px}
.stat b{display:block;font-size:24px}.stat span{font-size:12px;color:#9a90a6}
h2{font-size:18px;margin:26px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:14px}
@media(max-width:560px){.grid{grid-template-columns:1fr}}
.card{border:1px solid #2c2433;background:#16121b;border-radius:16px;padding:16px 18px}
.top{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.rank{font-weight:800;font-size:13px;background:#ff4d5a;color:#fff;border-radius:8px;padding:3px 8px}
.title{font-weight:700;flex:1}.score{font-weight:800;font-size:20px;color:#ffb347}.score small{font-size:11px;color:#9a90a6}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.chip{font:12px ui-monospace,Menlo,monospace;background:#221b29;border:1px solid #3a2f44;padding:4px 8px;border-radius:8px}
.q{border-left:3px solid;padding:6px 10px;margin:6px 0;border-radius:0 8px 8px 0}
.q i{display:block;font-style:normal;font-size:10px;letter-spacing:1.2px;font-weight:700}
.hook{border-color:#ffb347;background:#2a2016}.hook i{color:#ffb347}
.cliff{border-color:#ff4d5a;background:#2a161b}.cliff i{color:#ff4d5a}
.why{color:#bdb3c8;font-size:13px;margin:8px 0}.reason{color:#8f859b;font-size:12px}
.flags{font-size:12px;color:#e0a35a;margin-top:6px}
details{margin-top:8px}summary{cursor:pointer;color:#9a90a6;font-size:12px}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,Menlo,monospace;color:#cfc6da;background:#120f16;padding:10px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px 8px;border-bottom:1px solid #2c2433;text-align:left}
th{color:#9a90a6;font-weight:600}
"""


def _card(c: dict) -> str:
    e = html.escape
    s = c.get("scores", {})
    chips = "".join(f'<span class="chip">Ep {x["episode"]} · {x["start_ts"]} → {x["end_ts"]}</span>'
                    for x in c["segments_resolved"]) + f'<span class="chip">{c["duration"]:.0f} s</span>'
    bars = " · ".join(f"{k} {s.get(k, 0):.1f}/4" for k in ("hook", "cliffhanger", "standalone", "boundary"))
    flags = c.get("flags", []) + c.get("repairs", [])
    return f"""<div class="card"><div class="top"><span class="rank">#{c['rank']}</span>
<span class="title">{e(c.get('title') or 'Promo')}</span><span class="score">{c['score']:.0f}<small>/100</small></span></div>
<div class="chips">{chips}</div>
<div class="q hook"><i>HOOK · FIRST 3 SECONDS</i>{e(c['hook_quote'])}</div>
<div class="q cliff"><i>CLIFFHANGER · THE CUT</i>{e(c['cliffhanger_quote'])}</div>
<div class="why">{e(c.get('why', ''))}</div><div class="reason">{e(bars)} · {e(c.get('rank_reason', ''))}</div>
{f'<div class="flags">⚑ {e("; ".join(flags))}</div>' if flags else ''}
<details><summary>Screenplay of this cut</summary><pre>{e(c.get('text', ''))}</pre></details></div>"""


def write_html(ranked: list[dict], rejected: list[dict], meta: dict, out_dir: Path, top: int = 40) -> Path:
    e = html.escape
    t = meta.get("tournament", {})
    stats = [(meta.get("episodes"), "episodes"), (meta.get("candidates"), "candidates proposed"),
             (meta.get("rejected"), "rejected by the checker"), (meta.get("ranked_after_dedupe"), "promos ranked"),
             (f'{meta.get("seconds", 0):.0f}s', "run time")]
    if t.get("order_flip_rate") is not None:
        stats.append((f'{t["order_flip_rate"]:.0%}', "judge A/B flips (ties)"))
    rej = "".join(f"<tr><td>{e(str(c.get('cid')))}</td><td>{e(c.get('reject_reason', ''))}</td>"
                  f"<td>{e(str(c.get('hook_quote', ''))[:90])}</td></tr>" for c in rejected)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CliffCut report</title><style>{CSS}</style></head><body><div class="wrap">
<div class="hero"><div class="logo">Cliff<span>Cut</span></div>
<div class="sub">Ranked promo clips · finder {e(str(meta.get('finder', {}).get('model', '')))} ·
judge {e(str(meta.get('judge_name', '')))}</div></div>
<div class="stats">{''.join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)}</div>
<h2>Top {min(top, len(ranked))} promos, in posting order</h2>
<div class="grid">{''.join(_card(c) for c in ranked[:top])}</div>
<h2>Rejected by the checker ({len(rejected)})</h2>
<table><tr><th>candidate</th><th>why</th><th>hook</th></tr>{rej}</table>
</div></body></html>"""
    path = Path(out_dir) / "report.html"
    path.write_text(page)
    return path
