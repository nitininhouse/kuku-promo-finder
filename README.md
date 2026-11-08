# Promo Finder

Reads a microdrama screenplay and finds 20-60 second promo clips: a hook in the first 3 seconds,
a cliffhanger at the cut, and a story a stranger can follow. Built for Kuku's AI Generalist
assignment, Problem 1.

**The model proposes, code checks, a separate judge scores, a human picks.**

```
episodes/*.txt ─► 1 parse (code) ─► 2 find (LLM) ─► 3 check (code) ─► 4 score (Jev) ─► 5 pairwise re-rank ─► ranked promos
                  numbered cards     card IDs +       times from IDs,    probability     head-to-head on      + raw outputs
                  E07-031 …          exact quotes     20-60s, hook/cliff  rubric          the top 24           + rejection log
```

| Stage | Who | What it does |
|---|---|---|
| 1 Parse | code | Every timestamped block (camera, action, sound, dialogue) becomes a card with an ID. |
| 2 Find | LLM, one call per episode | Proposes promos as card-ID ranges plus a **word-for-word** hook and cliffhanger quote. Never writes a timestamp. |
| 3 Check | code | Rejects made-up IDs and quotes; trims the start onto the hook and the end onto the cliffhanger; never cuts mid-line; enforces 20-60s; flags names a new viewer never sees. Every rejection goes to `rejections.jsonl`. |
| 4 Score | Jev (TypeSafe), or the same rubric as an LLM prompt | Hook, cliffhanger, standalone and boundary scores. The cliffhanger question also sees what happens *after* the cut, so it can tell an open ending from a resolved one. |
| 5 Re-rank | same judge | Swiss tournament + Bradley-Terry on the top 24, because absolute scores bunch at the top. Reports position bias. |
| Rank | code | Removes overlapping picks, spaces episodes out, gives early episodes a small boost (new viewers start at episode 1). |

## Results on the full 50-episode pack

Three answer keys nobody tuned on (`eval/`):

| Answer key | Keyword baseline | Promo Finder (default) |
|---|---|---|
| 8 promos marked blind (card IDs) | 1/8 | **8/8** |
| Kuku's own episode titles: titled moment inside a pick | 20/45 | **42/45** |
| 21 real YouTube Shorts promos of the show (by transcript) | 10/21 | **21/21** (100% of views) |

Plus 20 paid Meta Ad Library creatives for the show (`eval/meta_ads.csv`): their most-used openings
(episode-18 fire, episode-3 wedding expulsion) are the #1 picks of our DeepSeek-only and default runs.

Scoring: `python -m promo_finder.evaluate`, `python -m promo_finder.eval_titles`, `python -m promo_finder.eval_real`.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # add DEEPSEEK_API_KEY (or GEMINI_API_KEY, ...) and TYPESAFE_API_KEY
.venv/bin/python -m promo_finder path/to/episodes_folder_or.zip           # all episodes
.venv/bin/python -m promo_finder show.zip --episodes 1-5 --provider gemini  # swap model from the CLI
.venv/bin/python -m promo_finder show.zip --provider baseline --judge heuristic  # no key, no model
.venv/bin/streamlit run app.py                                              # web UI
.venv/bin/python -m pytest -q                                               # tests (synthetic show only)
```

## Swapping the model

One block in `config.toml`:

```toml
[finder]
provider = "deepseek"        # deepseek | gemini | groq | openrouter | ollama | anthropic | baseline
model = "deepseek-flash"

[[extra_finders]]            # optional: pool a second model's candidates
provider = "gemini"
model = "gemini-3.8-flash"

[scorer.llm]                 # the judge: a different family from the finder
provider = "gemini"
model = "gemini-3.8-flash"
# base_url = "https://your-gateway/v1"   # any OpenAI-compatible API
```

All model calls go through one function, `call_model()` in `promo_finder/models.py`. Responses are cached
on disk, so re-running the checker or ranking costs nothing.

## Outputs (`outputs/<run>/`)

`promos.csv` / `promos.json` / `report.md` (ranked promos), `raw/epNN_finder.json` (unedited model replies),
`rejections.jsonl` (everything the checker threw out, and why), `run_meta.json` (tokens, cost, timing,
tournament bias stats).

## Evaluate

Mark promos by hand in `eval/gold.csv` (card IDs only, no script text), then:

```bash
.venv/bin/python -m promo_finder.evaluate data outputs/<run> outputs/<baseline-run>
```

## Confidentiality

The input pack is not in this repo: `data`, `outputs/`, `.cache/` and `.env` are git-ignored.
`samples/demo_show` is an original made-up episode used for tests and the hosted demo.
