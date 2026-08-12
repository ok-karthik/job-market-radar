# LinkedIn Jobs ETL & Semantic Matcher

## 🎯 Project Goal
An automated pipeline that scrapes LinkedIn for highly specific technical roles in Germany (Platform Engineering, SRE, DevOps, Cloud Engineering, AI Infrastructure, MLOps), forcefully filters out non-relevant or non-English roles, and uses local AI/NLP to semantically rank the remaining jobs against a specific candidate's CV.

## 🏗 Architecture & Pipeline Flow
The project is orchestrated by `main.py` and relies on a strict, three-stage ETL pipeline, plus two standalone analytics scripts that only ever read already-processed output.

### Stage 1: Extraction (`src/scraper/apify_replica.py`)
- **Functionality**: Bypasses LinkedIn authentication to scrape job search results from the public guest API, in two phases.
- **Phase A — ID harvest (`extract_job_ids`, serial `httpx.Client`)**: Iterates `TARGET_PROFILES` — 18 search pools, built from 3 keyword groups (`_KW_CORE` = Platform Engineer/SRE/DevOps/Infrastructure Engineer, `_KW_AI` = AI Infrastructure/MLOps/AI Platform/Agentic/GenAI, `_KW_CLOUD` = Cloud Engineer/Cloud Infrastructure Engineer/Kubernetes Engineer&Administrator/DevSecOps) × 6 geo targets (Germany-remote, Berlin, Munich, Frankfurt, Cologne/Düsseldorf, Hamburg). Each pool is a single `geoId`; the per-profile cap (150 for the default 24h range) is gated on that profile's own distinct-job count, not global cross-pool novelty. Detects LinkedIn's IP soft-block (a full page of duplicate results replayed at HTTP 200) versus genuine end-of-stream, backing off and re-probing once before aborting the whole run with rotation instructions if the block persists. Retries 429/503 with full-jitter capped exponential backoff.
- **Phase B — detail hydration (`populate_job_details`, async, `HYDRATION_CONCURRENCY=2`)**: Pre-filters obviously irrelevant titles before the HTTP round-trip, then fetches each job's full HTML detail page. Injects bullet points (`• `) into `<li>` HTML tags before extracting raw text to preserve formatting, and extracts `seniorityLevel`/`employmentType`/`postedAt`.
- **Output**: Raw JSON in `jobs_output/`, named `jobs_[TIMERANGE]_[YYYYMMDD]_[HHMM].json` (e.g. `jobs_24h_20260715_1700.json`; `TIMERANGE` is `24h`/`7d`/`1m` depending on `TIME_RANGE`). Job schema: `id, link, title, companyName, location, postedAt, descriptionText, seniorityLevel, employmentType`.
- **Legacy note**: `src/scraper/apify_replica_old.py` is a pre-refactor version, unreferenced anywhere in the codebase — dead code, not a fallback.

### Stage 2: Filtering (`src/scraper/filter_jobs.py`)
- **Functionality**: Cleans the raw data and drops jobs that do not meet strict criteria. Considered **stable** — don't change its core filter logic without explicit instruction (see `.agents/AGENTS.md` rule 3).
- **Logic**:
  - Drops jobs requiring German (`langdetect` NLP detection on the description, plus regex patterns for phrases like "fluent German" — excludes "nice to have" phrasing).
  - Drops Contract, Freelance, Part-time, Temporary, or Internship roles.
  - Drops jobs located outside Germany (German cities/regions allowed outright; EU/UK locations allowed only if explicitly remote; US state abbreviations explicitly rejected).
  - Cleans up UI artifacts from the scraper (e.g. "Show more"/"Show less").
- **Output**: `.csv` and `.json` files appended with `_filtered` (e.g. `jobs_24h_20260715_1700_filtered.json`).

### Stage 3: Semantic Analysis & Analytics (`src/scraper/semantic_job_analyzer.py`)
- **Functionality**: Ranks the filtered jobs against the user's CV using local models — no data leaves the machine.
- **Models**: `BAAI/bge-small-en-v1.5` for embeddings (512-token context, chosen over `all-MiniLM-L6-v2`'s 256-token limit which silently truncated most descriptions) and `urchade/gliner_small-v2.1` (GLiNER) for zero-shot NER, extending skill extraction beyond the canonical regex list.
- **CV source**: `CV_URL` constant (a Google Docs export link) — downloaded fresh to `CV.pdf` (gitignored) on every run so edits to the Google Doc are picked up automatically. `--cv-file` overrides with a local `.txt`/`.pdf`/`.docx`.
- **Logic**:
  - **Categorization**: Rule-based title overrides for high-confidence cases (e.g. "backend" in title → `Backend Engineering` immediately), falling back to cosine similarity against 17 semantic category buckets (Platform Engineering, SRE, DevOps Engineering, Cloud Engineering, Security Engineering/DevSecOps, AI Infrastructure, MLOps, Data Engineering, Data Science & ML Engineering, AI Solutions Architecture, Solutions Architecture, Staff/Principal Engineering, Engineering Leadership, Backend Engineering, Frontend & Fullstack Engineering, Technical Sales & Pre-sales, Product Management).
  - **Role-family rejection (`classify_role_family`)**: adds `RoleFamily` (`on-target` | `unclear` | `off-target`) and `RoleFamilyReason`. Categorization above is **forced-choice with no "none of the above"**, so a job whose true role is not one of the 17 (an FPGA engineer, a QA lead, an account executive) is still labelled and lands in the nearest bucket — measured at 58% of the platform/SRE/DevOps/cloud categories having no infra term in the title. This is the separate reject signal. Off-target rows are **dropped from the written output** (`--keep-off-target` retains them; the input `_filtered.json` always keeps every row). Evaluated title-first (hard off-discipline → explicit infra term → app-dev title needing strong infra evidence → web-stack dominance), defaulting to `unclear` = keep. Deliberately uncached (pure regex) so pattern fixes apply immediately.
  - **Skill Extraction**: ~70 canonical technologies via word-boundary regex (`TECH_SKILLS_PATTERNS`), extended by GLiNER's dynamic zero-shot extraction (confidence-filtered at `score >= 0.6`, with a soft-skill/business-term blocklist).
  - **Salary Extraction**: Regex-parsed EUR min/max from the description (full-number, `k`-format, hybrid ranges), sanity-clamped to €30k–€500k.
  - **Ranking**: `SemanticMatchScore` — raw cosine similarity between job description and CV, min-max normalized across the whole batch so the top match reads ~100 and the weakest ~0.
  - **Analytics**: Generates two text files per run summarising the job market (Raw Market Insights across the full scrape vs. Filtered Market Insights for the shortlist, split Germany vs. EU/UK-remote).
- **Output**: Ranked files appended with `_semantic` (e.g. `jobs_24h_20260715_1700_filtered_semantic.json`) and insights files appended with `_insights.txt`.

### `jobs_analytics/skill_gap_report.py`
- **Functionality**: weekly skill-gap check — the most-demanded NOW skills vs the CV's tools, gaps split NOW/NEXT by **Req%**, and a z-tested trend over the scrape window, ending in an explicit VERDICT.
- **Read-only**: parses existing `*_filtered_semantic.json`. No models, no network. Wired into `main.py` as step 5/6, non-fatal.
- **Baseline**: `jobs_analytics/skill_gap_baseline.json`, committed, giving a git-diffable record of market movement. `main.py` runs *without* `--baseline` on purpose — re-baselining every run would zero the week-over-week diff.
- **The alert threshold is Req%, not prevalence** (`GAP_ALERT_REQ_PCT`), matching the project's standing rule that prevalence overstates what actually gates an application.

### Orchestration (`main.py`)
A wrapper script that sequentially: (1) runs the scraper (`python -m scraper.apify_replica`), (2) finds the freshest raw JSON in `jobs_output/` by ctime (excluding `_filtered`/`_semantic` derivatives), (3) passes it to the filter (`python -m scraper.filter_jobs`), (4) passes the filtered result to the semantic analyzer (`python -m scraper.semantic_job_analyzer`), (5) runs `jobs_analytics/update_learning_plan.py`, (6) runs `jobs_analytics/skill_gap_report.py`, then (7) publishes to Notion. Sets `PYTHONPATH=src` itself. Scheduled via macOS `launchd` on weekdays at 18:30 (see `README.md`).

### Standalone Analytics — NOT wired into the ETL, never called from `apify_replica.py`
- **`jobs_analytics/update_learning_plan.py`**: Aggregates all `jobs_output/*_filtered_semantic.json` data, buckets extracted skills into tool "umbrellas" (cloud, Kubernetes, IaC, CI/CD, observability, LLMs, RAG, GPU, agents, etc.), groups the 17 categories into 3 career tracks (`TRACKS`: 🎯 NOW Core / 📈 NEXT Growth-AI / 🧭 LATER Staff-IC), and dynamically overwrites the auto-generated sections inside `your learning plan (`profile.plan_markdown` in config.json)` in the repo root — per-**category** skill demand with a recent-vs-prior trend column, per-track ROI, a title-based Senior→Staff gap table + Staff-bar readiness themes, per-track salary bands, an emerging-skills radar, and a qualitative leadership-signal section. The emerging-skills radar is gated on per-job fit (>=50) and requires >=3 distinct employers with no single employer over 50% of mentions, so it reports tool trends rather than one company's vocabulary.
- **`jobs_analytics/market_insights.py`**: Aggregates **all historical** `*_semantic.json` files in a directory (default `jobs_output/`), deduplicating by `(title, companyName)` across all runs, into `jobs_output/global_market_insights.txt`. Only ever run against `*_semantic.json` — never raw data.

## 📂 Key File Paths
- `<repo>` (Root)
  - `main.py` — pipeline orchestrator
  - `pyproject.toml` — deps + `[tool.pytest.ini_options]` (`pythonpath = ["src"]`, `testpaths = ["tests"]`)
  - `src/scraper/`
    - `apify_replica.py` — Stage 1 (ID harvest + hydration)
    - `apify_replica_old.py` — dead code, unreferenced; do not use as reference
    - `filter_jobs.py` — Stage 2 (stable filter logic)
    - `semantic_job_analyzer.py` — Stage 3 (scoring, categorization, skills, salary)
    - `__init__.py` — empty, makes `scraper` importable
  - `tests/` — pytest suite (`test_apify_replica.py`, `test_filter_jobs.py`, `test_semantic_analyzer.py`, `fixtures/`)
  - `jobs_output/` — all generated JSON/CSV/TXT artifacts; gitignored except `*_semantic.csv`
  - `jobs_analytics/`
    - `update_learning_plan.py` — rewrites the Master Learning Plan's tool-usage table
    - `market_insights.py` — historical cross-run aggregation
    - `scratch/` — ad-hoc scripts (excluded from pytest collection via `testpaths`)
  - `CV.pdf` — downloaded fresh from Google Docs every Stage 3 run; gitignored, never committed
  - `.agents/AGENTS.md` — full agent operating rules (source of truth for pipeline internals, referenced from `CLAUDE.md`)
  - `.github/workflows/python-tests.yml` — CI, runs `uv run pytest -v` on push/PR
