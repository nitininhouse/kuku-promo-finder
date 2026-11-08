"""Unit tests for every stage, on synthetic data only (no Kuku content, no network)."""
import io
import json
import zipfile
from pathlib import Path

import pytest

from promo_finder import check, find, models, pipeline, score, stitch, tournament
from promo_finder.check import Checker, norm, quote_match
from promo_finder.evaluate import iou
from promo_finder.parse import fmt, load_show, parse_episode
from promo_finder.report import write_html

DEMO = Path(__file__).parent.parent / "samples" / "demo_show"
SHOW = load_show(DEMO)
CHK = Checker(SHOW)

TWO_SCENES = """EPISODE NUMBER: Episode 2
EPISODE LOGLINE: test
CONTINUITY NOTES: test

SCENE 01
**ACTION:**
Start: (00:00.00) | End: (00:05.00)
Ana slams the door.

**ANA (Angry):**
Start: (00:05.00) | End: (00:15.00)
"Where is Boris? Tell me where Boris went!"

**ACTION:**
Start: (00:15.00) | End: (00:30.00)
Ana searches the room.

SCENE 02
**CAMERA:**
Start: (00:30.00) | End: (00:40.00)
Wide shot of a dark street.

**ANA (Whisper):**
Start: (00:38.00) | End: (00:50.00)
"Someone is following me. Who is there?"

**ACTION:**
Start: (00:50.00) | End: (01:20.00)
A stranger steps out of the shadows and explains everything calmly.
"""
EP2 = parse_episode(TWO_SCENES, 2)


def cand(**kw):
    base = dict(cid="t", segments=[{"first": "E01-003", "last": "E01-016"}],
                hook_card="E01-003", hook_quote="a hand slaps Mara across the face",
                cliffhanger_card="E01-016", cliffhanger_quote="Husband? Who are you?", why="x")
    return base | kw


# ---------- parse ----------
def test_fmt_pads_minutes_and_seconds():
    assert fmt(65.5) == "01:05.50" and fmt(0) == "00:00.00"


def test_render_dialogue_and_action():
    c = SHOW[0].cards[3]
    assert c.render().startswith("[E01-004] 00:07.00-00:09.50 VIVIAN (Furious):")
    assert SHOW[0].cards[0].render().split("] ")[1].startswith("00:00.00-00:04.00 CAMERA:")


def test_render_with_offset_starts_at_zero():
    assert SHOW[0].cards[2].render(offset=6.0).split("] ")[1].startswith("00:00.00")


def test_scenes_are_tracked():
    assert [c.scene for c in EP2.cards] == [1, 1, 1, 2, 2, 2]


def test_load_from_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(next(DEMO.glob("*.txt")), "show/episode-001-Episode 1.txt")
        z.writestr("__MACOSX/._junk.txt", "x")
    eps = load_show(buf.getvalue())
    assert len(eps) == 1 and len(eps[0].cards) == 18


def test_episode_number_falls_back_to_filename():
    ep = parse_episode("no header\n", 7)
    assert ep.ep == 7 and ep.cards == []


# ---------- check ----------
def test_norm_and_quote_match():
    assert norm("Who ARE you?!") == "who are you"
    assert quote_match("who are you", "Husband? Who are you?") == 1.0
    assert quote_match("completely different words here", "Husband? Who are you?") < 0.85


def test_unknown_hook_card_rejected():
    assert "does not exist" in CHK.check(cand(hook_card="E01-999"))["reject_reason"]


def test_hook_outside_first_segment_rejected():
    c = cand(segments=[{"first": "E01-011", "last": "E01-016"}], hook_card="E01-003")
    assert "hook card is not in the first segment" in CHK.check(c)["reject_reason"]


def test_cliffhanger_outside_last_segment_rejected():
    c = cand(segments=[{"first": "E01-003", "last": "E01-010"}])
    assert "cliffhanger card is not in the last segment" in CHK.check(c)["reject_reason"]


def test_resolution_after_cliffhanger_is_trimmed():
    c = CHK.check(cand(segments=[{"first": "E01-003", "last": "E01-018"}],
                       cliffhanger_card="E01-012", cliffhanger_quote="And who are you to give orders here?"))
    assert c["ok"] and any("trimmed end" in r for r in c["repairs"])
    assert c["segments_resolved"][0]["end"] <= 40.0


def test_overlapping_segments_rejected():
    c = cand(segments=[{"first": "E01-003", "last": "E01-010"}, {"first": "E01-008", "last": "E01-016"}])
    assert "overlap" in CHK.check(c)["reject_reason"]


def test_reversed_segment_rejected():
    assert "bad segment" in CHK.check(cand(segments=[{"first": "E01-016", "last": "E01-003"}]))["reject_reason"]


def test_too_long_rejected():
    chk = Checker([EP2])
    c = dict(cid="x", segments=[{"first": "E02-001", "last": "E02-006"}], hook_card="E02-001",
             hook_quote="Ana slams the door", cliffhanger_card="E02-006",
             cliffhanger_quote="A stranger steps out of the shadows")
    assert "too long" in chk.check(c)["reject_reason"]


def test_flags_names_never_shown_and_scene_jumps():
    chk = Checker([EP2])
    c = chk.check(dict(cid="x", segments=[{"first": "E02-002", "last": "E02-005"}], hook_card="E02-002",
                       hook_quote="Where is Boris? Tell me where Boris went!", cliffhanger_card="E02-005",
                       cliffhanger_quote="Someone is following me. Who is there?"))
    assert c["ok"]
    assert any("spans 2 scenes" in f for f in c["flags"])
    assert c["unseen_names"] == []  # Boris never speaks, so he is not a known character name


def test_cut_never_lands_inside_a_line():
    # ending on the camera card at 40s would cut 2s into ANA's 12s line (38-50s): pull the cut back to 38s
    chk = Checker([EP2])
    c = chk.check(dict(cid="x", segments=[{"first": "E02-003", "last": "E02-004"}], hook_card="E02-003",
                       hook_quote="Ana searches the room", cliffhanger_card="E02-004",
                       cliffhanger_quote="Wide shot of a dark street"))
    assert c["segments_resolved"][-1]["end"] == 38.0
    assert any("pulled the cut back" in r for r in c["repairs"])
    assert not any("inside line" in f for f in c["flags"])


def test_short_overlapping_line_is_included():
    # the demo slap SFX overlaps VIVIAN's line by a fraction; short overlaps are absorbed, not cut
    c = CHK.check(cand(segments=[{"first": "E01-003", "last": "E01-016"}]))
    assert not any("inside line" in f for f in c["flags"])


def test_after_cut_marks_episode_end():
    chk = Checker([EP2])
    c = chk.check(dict(cid="x", segments=[{"first": "E02-005", "last": "E02-006"}], hook_card="E02-005",
                       hook_quote="Someone is following me", cliffhanger_card="E02-006",
                       cliffhanger_quote="A stranger steps out of the shadows"))
    assert c["after_cut"] == "(episode ends)"


# ---------- find ----------
def test_prompt_lists_every_card_and_forbids_timestamps():
    p = find.build_prompt(SHOW[0], SHOW, 3)
    assert all(c.id in p for c in SHOW[0].cards)
    assert "Do not write timestamps" in p and "(first episode)" in p


def test_baseline_candidates_pass_the_checker():
    out = find.find_baseline(SHOW[0], 3)
    assert out["candidates"] and all(CHK.check(c)["ok"] for c in out["candidates"])


def test_find_llm_uses_model_reply(monkeypatch):
    reply = {"promos": [cand()]}
    monkeypatch.setattr(find, "call_model", lambda *a, **k: {"json": reply, "raw": "r", "usage": {}})
    out = find.find_llm(SHOW[0], SHOW, {}, 3)
    assert out["candidates"][0]["cid"] == "E01-P1" and out["candidates"][0]["source"] == "llm"


# ---------- models ----------
def test_extract_json_variants():
    assert models.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert models.extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(models.ModelError):
        models.extract_json("no json here")


def test_resolve_merges_preset_and_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k123")
    r = models.resolve({"provider": "gemini", "model": "custom-model"})
    assert r["model"] == "custom-model" and r["api_key"] == "k123" and r["api"] == "openai"


def test_call_model_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "CACHE", tmp_path)
    calls = []
    monkeypatch.setattr(models, "_call_once", lambda *a: calls.append(1) or ('{"x": 1}', {"in": 1, "out": 1}))
    cfg = {"provider": "ollama"}
    assert models.call_model("p", None, cfg)["json"] == {"x": 1}
    assert models.call_model("p", None, cfg)["cached"] is True and len(calls) == 1


def test_call_model_does_not_retry_bad_key(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "CACHE", tmp_path)
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    calls = []

    def boom(*a):
        calls.append(1)
        raise models.ModelError("HTTP 401: bad key")
    monkeypatch.setattr(models, "_call_once", boom)
    with pytest.raises(models.ModelError):
        models.call_model("p", None, {"provider": "ollama"}, use_cache=False)
    assert len(calls) == 1


# ---------- score ----------
def test_total_weights_and_penalties():
    perfect = {k: 4 for k in score.WEIGHTS}
    assert score.total(perfect, {"flags": [], "unseen_names": []}) == 100
    assert score.total(perfect, {"flags": ["cut lands inside line X"], "unseen_names": ["Bo"]}) == 94


def test_heuristic_scores_in_range():
    s = score.heuristic_score(CHK.check(cand()))
    assert all(0 <= s[k] <= 4 for k in score.WEIGHTS)


def test_panel_averages_judges(monkeypatch):
    def fake(cands, cfg):
        v = 4.0 if cfg["model"] == "a" else 2.0
        return {c["cid"]: {"judge": cfg["model"], **{k: v for k in score.RUBRIC}} for c in cands}
    monkeypatch.setattr(score, "llm_scores", fake)
    out = score.panel_scores([{"cid": "x"}], [{"model": "a"}, {"model": "b"}])
    assert out["x"]["hook"] == 3.0 and out["x"]["judge"] == "panel(a+b)"


def test_jev_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(score, "panel_scores", lambda c, j: {x["cid"]: {"judge": "llm", **{k: 2.0 for k in score.RUBRIC}}
                                                             for x in c})
    c = CHK.check(cand())
    score.score_all([c], {"judge": "jev", "llm": {}}, log=lambda m: None)
    assert c["scores"]["judge"] == "llm" and c["score"] == 50.0


# ---------- tournament ----------
def test_bradley_terry_symmetric_when_even():
    s = tournament.bradley_terry(2, [(0, 1, 0.5), (1, 0, 0.5)])
    assert abs(s[0] - s[1]) < 1e-6


def test_panel_pairs_average(monkeypatch):
    monkeypatch.setattr(tournament, "_llm_pairs", lambda pairs, cfg: [1.0 if cfg["m"] == "a" else 0.0] * len(pairs))
    assert tournament.judge_pairs([({}, {})] * 3, "llm", {"llm": [{"m": "a"}, {"m": "b"}]}) == [0.5, 0.5, 0.5]


def _clips(n):
    return [{"cid": f"c{i}", "score": 50.0 + i, "text": f"clip {i}"} for i in range(n)]


def test_tournament_detects_first_position_bias(monkeypatch):
    monkeypatch.setattr(tournament, "judge_pairs", lambda pairs, j, c: [1.0] * len(pairs))  # always picks A
    stats = tournament.tournament(_clips(6), "llm", {}, k=6, rounds=2, log=lambda m: None)
    assert stats["order_flip_rate"] == 1.0 and stats["first_shown_win_rate"] == 1.0


def test_tournament_consistent_judge_has_no_flips(monkeypatch):
    def by_index(pairs, j, c):
        return [1.0 if int(a["cid"][1:]) > int(b["cid"][1:]) else 0.0 for a, b in pairs]
    monkeypatch.setattr(tournament, "judge_pairs", by_index)
    clips = _clips(6)
    stats = tournament.tournament(clips, "llm", {}, k=6, rounds=3, log=lambda m: None)
    assert stats["order_flip_rate"] == 0.0
    assert clips[5]["pairwise"] == 100.0 and clips[0]["pairwise"] == 0.0


# ---------- stitch ----------
def test_stitch_keeps_only_cross_episode(monkeypatch):
    reply = {"promos": [cand(), {**cand(), "segments": [{"first": "E01-003", "last": "E01-005"},
                                                         {"first": "E02-002", "last": "E02-005"}]}]}
    monkeypatch.setattr(stitch, "call_model", lambda *a, **k: {"json": reply, "raw": "r", "usage": {}})
    pool = [{"score": 50, "first_episode": 1, "duration": 30, "text": "a"},
            {"score": 40, "first_episode": 2, "duration": 30, "text": "b"}]
    out = stitch.stitch(SHOW + [EP2], pool, {})
    assert len(out["candidates"]) == 1 and out["dropped_single_episode"] == 1


# ---------- rank / pipeline ----------
def _ranked(cid, ep, start, end, sc):
    return {"cid": cid, "score": sc, "first_episode": ep, "flags": [], "scores": {"judge": "t"},
            "segments_resolved": [{"episode": ep, "start": start, "end": end, "seconds": end - start}]}


def test_rank_merges_overlapping_picks():
    out = pipeline.rank([_ranked("a", 1, 0, 30, 80), _ranked("b", 1, 5, 35, 70), _ranked("c", 1, 60, 90, 60)])
    assert [c["cid"] for c in out] == ["a", "c"] and out[0]["duplicates"] == ["b"]


def test_rank_spaces_out_episodes():
    picks = [_ranked(f"e1-{i}", 1, i * 100, i * 100 + 30, 90 - i) for i in range(3)] + [_ranked("e2", 2, 0, 30, 10)]
    order = [c["cid"] for c in pipeline.rank(picks, per_episode=2, early_bonus=False)]
    assert order.index("e2") == 2


def test_full_pipeline_with_fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(find, "call_model", lambda *a, **k: {"json": {"promos": [cand(), cand(hook_quote="nope nope")]},
                                                            "raw": "r", "usage": {"in": 10, "out": 5}})
    monkeypatch.setattr(pipeline, "find_llm", find.find_llm)
    cfg = {"finder": {"provider": "fake"}, "scorer": {"judge": "heuristic"}, "candidates_per_episode": 2,
           "stitch": {"enabled": False}, "tournament": {"k": 0}}
    res = pipeline.run(SHOW, cfg, tmp_path, log=lambda m: None)
    assert res["meta"]["passed"] == 1 and res["meta"]["rejected"] == 1
    for f in ("promos.csv", "promos.json", "report.md", "report.html", "rejections.jsonl", "run_meta.json"):
        assert (tmp_path / f).exists(), f
    assert "quote not found" in (tmp_path / "rejections.jsonl").read_text()


# ---------- evaluate / report ----------
def test_iou():
    assert iou((0, 10), (0, 10)) == 1.0 and iou((0, 10), (5, 15)) == pytest.approx(1 / 3) and iou((0, 1), (2, 3)) == 0


def test_report_escapes_html(tmp_path):
    c = CHK.check(cand(title="<script>alert(1)</script>"))
    c.update(rank=1, score=50.0, scores={"hook": 2}, rank_reason="r")
    page = write_html([c], [], {"episodes": 1}, tmp_path).read_text()
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_failed_judge_batch_falls_back_instead_of_crashing(monkeypatch):
    def boom(*a, **k):
        raise models.ModelError("429 balance")
    monkeypatch.setattr(score, "call_model", boom)
    c = CHK.check(cand())
    score.score_all([c], {"judge": "llm", "llm": {"model": "x"}}, log=lambda m: None)
    assert c["scores"]["judge"] == "heuristic" and c["score"] > 0


def test_failed_pair_call_counts_as_tie(monkeypatch):
    def boom(*a, **k):
        raise models.ModelError("timeout")
    monkeypatch.setattr(tournament, "call_model", boom)
    assert tournament._llm_pairs([({"text": "a"}, {"text": "b"})], {}) == [0.5]


def test_lead_with_finder_pick_puts_each_episodes_p1_first():
    ranked = [{"cid": "E02-P2"}, {"cid": "E01-P1"}, {"cid": "E01-P2"}, {"cid": "E02-P1"}, {"cid": "E01-P1-2"}]
    out = [c["cid"] for c in pipeline.lead_with_finder_pick(ranked)]
    assert out == ["E01-P1", "E02-P1", "E02-P2", "E01-P2", "E01-P1-2"] and ranked[0]["rank"] == 3
