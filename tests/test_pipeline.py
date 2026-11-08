from pathlib import Path

from promo_finder.check import Checker
from promo_finder.parse import load_show
from promo_finder.pipeline import rank
from promo_finder.score import score_all

SHOW = load_show(Path(__file__).parent.parent / "samples" / "demo_show")
CHK = Checker(SHOW)


def cand(**kw):
    base = dict(cid="t", segments=[{"first": "E01-003", "last": "E01-016"}],
                hook_card="E01-003", hook_quote="a hand slaps Mara across the face",
                cliffhanger_card="E01-016", cliffhanger_quote="Husband? Who are you?", why="x")
    return base | kw


def test_parse():
    ep = SHOW[0]
    assert ep.ep == 1 and len(ep.cards) == 18
    assert ep.cards[3].kind == "DIALOGUE" and ep.cards[3].speaker == "VIVIAN"
    assert ep.cards[3].direction == "Furious" and ep.cards[3].start == 7.0


def test_good_candidate_gets_times_from_cards():
    c = CHK.check(cand())
    assert c["ok"], c["reject_reason"]
    s = c["segments_resolved"][0]
    assert (s["start"], s["end"]) == (6.0, 48.0) and c["duration"] == 42.0


def test_invented_id_rejected():
    assert "unknown card" in CHK.check(cand(segments=[{"first": "E01-003", "last": "E01-099"}]))["reject_reason"]


def test_invented_quote_rejected():
    c = CHK.check(cand(hook_quote="She throws wine in his face"))
    assert not c["ok"] and "quote not found" in c["reject_reason"]


def test_quote_on_neighbour_card_is_repaired():
    c = CHK.check(cand(hook_card="E01-004"))  # quote is really on E01-003
    assert c["ok"] and c["hook_card"] == "E01-003" and c["repairs"]


def test_slow_open_is_trimmed_to_hook():
    c = CHK.check(cand(segments=[{"first": "E01-001", "last": "E01-016"}]))
    assert c["ok"] and c["segments_resolved"][0]["start"] == 6.0
    assert any("trimmed start" in r for r in c["repairs"])


def test_too_short():
    short = cand(segments=[{"first": "E01-011", "last": "E01-016"}], hook_card="E01-011",
                 hook_quote="Let go of her. Right now.")
    assert "too short" in CHK.check(short)["reject_reason"]


def test_multi_segment_promo():
    c = CHK.check(cand(segments=[{"first": "E01-003", "last": "E01-005"},
                                 {"first": "E01-011", "last": "E01-016"}]))
    assert c["ok"] and len(c["segments_resolved"]) == 2 and c["duration"] == 24.0


def test_heuristic_score_and_rank():
    c = CHK.check(cand())
    out = rank(score_all([c], {"judge": "heuristic"}))
    assert out[0]["rank"] == 1 and 0 < out[0]["score"] <= 100


def test_bradley_terry_orders_by_wins():
    from promo_finder.tournament import bradley_terry
    games = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (0, 1, 0.8)]
    s = bradley_terry(3, games)
    assert s[0] > s[1] > s[2]


def test_after_cut_context_is_attached():
    c = CHK.check(cand())
    assert c["after_cut"].startswith("[E01-017]")
