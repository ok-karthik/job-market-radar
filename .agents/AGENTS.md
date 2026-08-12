# Agent rules for this repo

Guidance for any AI agent working in this codebase. These are not style preferences — each one is here because doing the opposite produced a measurable defect.

## Orientation

A nightly ETL + analytics pipeline: scrape → filter → semantically score → report. Run it with `uv run python main.py`. Tests: `uv run pytest -v` (288, all green).

- `src/scraper/{apify_replica,filter_jobs,semantic_job_analyzer}.py` — the three ETL stages
- `jobs_analytics/` — standalone reporting, never wired into the scraper
- `jobs_output/` — all artefacts, timestamped

## Core rules

1. **The three ETL stages pass JSON by a fixed schema.** Changing a field name means changing all three in the same commit.
2. **Never delete raw scrape data.** A run cannot be reproduced — the source is a live, changing job board.
3. **Run `uv run pytest -v` after touching pipeline code.** Do not finalise on a red suite.
4. **No scratch files in the repo root.** Use `jobs_analytics/scratch/`. Don't name them `test_*.py` — pytest will collect them.
5. **Keep scraping and analytics strictly separate.** Analytics reads `*_semantic.json` from disk; it never triggers a fetch.
6. **Don't route job descriptions through an LLM.** Tested against the local regex + embedding + NER stack: worse extraction, rate-limited, and non-deterministic. A cheap empirical check here is a throwaway script over the JSON already on disk.

## 🔬 Verify your own output

Six defects were shipped into this repo's own analysis tooling over a single stretch of work. Every one looked correct in code review and would have been reported as fact:

| Defect | How it was caught |
|---|---|
| A two-proportion z-test called with **counts** where it expects proportions | Reading the function signature |
| A gap alert firing on prevalence instead of the required-% | Running it once and reading the verdict |
| `\b\.net` and `c\+\+\b` — regexes that **could never match** | A `.NET` role kept surfacing unclassified |
| A bare alternative `first` in a rejection pattern | Reading the top rejects by score |
| A vocabulary list silently missing AWS/Azure/GCP/Linux | Investigating why a sound rule looked broken |
| Chart labels overprinting into mush | *Looking at the rendered PNG* |

**Every one was found by inspecting the output, not by re-reading the code.** So:

1. **Run it and read what it prints**, including the boring parts.
2. **Render and look at any image.** Code that executes is not a chart that reads.
3. **Check a function's signature before quoting its statistic.**
4. **Sample both directions** — read what a rule *rejected* at high score, not just what it kept.
5. **When a rule looks unsafe, suspect the vocabulary list before weakening the rule.**
6. **A regex that matches nothing looks identical to a regex with nothing to match.** Assert a known positive whenever you add one.

## Analysis anti-patterns

1. **Don't quote headline prevalence as "what to learn".** Quote the required-%. Mention-% conflates a hard gate with a four-way alternative.
2. **Don't infer 0% from absence in a curated list.** Ranked tables have a cut-off; the exhaustive audit does not.
3. **Don't trust a regex written mid-analysis more than the taxonomy.** Both fail, in both directions. Sample-check both.
4. **Don't claim you read the job descriptions** when what you read was excerpt windows. State coverage honestly (`N of M, in fragments`). An overstated evidence base is worse than a thin one, because it stops the next reader re-checking.
5. **Don't compare two percentages** without confirming they share a denominator.
6. **Category membership is not a population.** Filter by title before quoting a percentage — 58% of postings in the four target categories have no infrastructure term in the title.
7. **"Senior-titled vs everything else" is not a seniority comparison.** The comparison bucket is dominated by ads that state no tier at all, so the statistic measures whether an ad names a tier. Compare explicitly-titled Senior against explicitly-titled Staff+, and exclude the untitled.

## Classifier design: a forced choice cannot reject

`categorize_job()` picks the nearest of 17 category centroids. It has no "none of the above", so a posting with no correct label still gets one.

Measured against titles that state their own category, it agrees **90.8%** of the time. Accuracy is not the problem — **coverage** is. Narrowing a definition to exclude junk does not reject it; it **relocates** it to the next-nearest centroid, and evicts correct matches on the way out. An A/B of exactly this made both sides worse.

**So rejection needs its own signal.** That is `classify_role_family()`, which returns `on-target` / `unclear` / `off-target` with an auditable reason string. Two rules were built and removed the same day:

- **"No infrastructure vocabulary in the description ⇒ off-target."** Rejected 928 postings and looked clean — including three genuine platform/MLOps roles that simply described the work in prose without naming a tool. **Absence of stack vocabulary is not evidence of absence of infrastructure work.**
- **Rescuing app-dev titles on stack density.** Backend ads name Kubernetes and AWS in passing; this readmitted eight of them, one at a stack score of 13. **A backend ad mentioning Kubernetes is still a backend job.**

Also: **industry verticals must never outrank the role.** Blocking `aviation` rejected a DevOps platform role at score 92. The test is which word is the *role*: `Robotics Software Engineer` is a robotics job; `SRE — Robotics Platform` is an SRE job.

## Regex traps found here

- **A token starting or ending with a non-word character cannot sit inside `\b(...)\b`.** `\b\.net` needs a word character before the dot, but titles read `Lead .NET Engineer`. Use explicit lookarounds.
- **Acronyms get spelled out.** `\bfpga\b` misses `Field-Programmable Gate Arrays Engineer`.
- **Adjacency assumptions fail.** `(unity)\s+(engineer)` misses `Lead Unity Software Engineer (Gameplay)`.
- **Bare alternatives leak.** `first` matched `(AI-First)`.
- **Case sensitivity can be load-bearing.** A case-insensitive `\bgo\b` matches *go wrong*, *go-live* and *Go-To-Market*.

## Changing a filter

Three layers reject postings. Know which you are editing:

| Layer | Drops at | Reversible? |
|---|---|---|
| Title pre-filter | **before hydration** | ❌ lost permanently |
| Language / contract / location | stage 2 | ✅ raw JSON retained |
| `RoleFamily` | stage 3 write | ✅ filtered JSON retained |

**Stage 1 is the dangerous one.** Validate any change against the historical corpus before shipping, check collateral damage among high-scoring postings, and add regression tests in both directions.

Apply classification changes with `backfill_role_family.py` — pure regex over existing files, seconds instead of a ten-minute model re-run.
