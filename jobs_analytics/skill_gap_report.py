#!/usr/bin/env python3
"""Weekly skill-gap check — "what should I actually learn next?"

Answers three questions against the CLEAN corpus, and prints an explicit
VERDICT rather than a wall of numbers, because the useful output is
"has the answer changed?" and not "here are 90 percentages".

  1. COVERAGE  — the most-demanded NOW skills, and whether the CV lists them.
  2. GAP       — demanded skills the CV does NOT list, split NOW vs NEXT.
  3. TREND     — is anything actually emerging, z-tested over the window.

Run it weekly:

    uv run python jobs_analytics/skill_gap_report.py

Read-only: parses existing ``*_filtered_semantic.json``. No models, no network,
no scraping — safe to run any time, costs seconds.

    --baseline   overwrite the stored baseline with today's numbers
    --top N      how many NOW skills to show in the coverage table (default 12)

The baseline lives in ``jobs_analytics/skill_gap_baseline.json`` and IS
committed on purpose: a git diff on it is the cheapest possible record of the
market moving, and it makes the week-over-week comparison below meaningful
instead of relying on memory.

WHY THE CV LIST IS RECOVERED FROM THE DATA, NOT PARSED FROM CV.pdf:
``_apply_fit`` already computes the intersection of the job's tools and the
CV's tools for every job, and writes the first five into ``WhyMatched``. Taking
the union across ~2,000 jobs reconstructs the CV's tool list without opening
the PDF or loading a model. It under-reports only for tools that never make any
job's top five — in practice a handful of rare ones — so treat a lone "GAP" on
a skill you know you have as a reporting artefact, not a finding.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from scraper.semantic_job_analyzer import TECH_SKILLS_PATTERNS  # noqa: E402
import update_learning_plan as u  # noqa: E402

DATA_GLOB = os.path.join(os.path.dirname(__file__), "..", "jobs_output",
                         "*_filtered_semantic.json")
BASELINE = os.path.join(os.path.dirname(__file__), "skill_gap_baseline.json")

NOW_CATS = {"Platform Engineering", "Site Reliability Engineering (SRE)",
            "DevOps Engineering", "Cloud Engineering"}
NEXT_CATS = {"AI Infrastructure", "MLOps", "AI Solutions Architecture"}

# Req% probes for the skills that sit on the NOW/NEXT boundary. These are the
# decision-relevant ones: everything else is either already covered or noise.
BOUNDARY_PROBES = {
    "AI agents / agentic": r"\bagentic\b|\bAI agents?\b|multi[- ]agent",
    "LLM APIs (consume)": r"\bLLM\b|\bOpenAI\b|\bAnthropic\b|\bBedrock\b|\bGenAI\b",
    "RAG / vector search": r"\bRAG\b|retrieval augmented|vector (database|search|db)",
    "GPU / CUDA": r"\bGPU\b|\bCUDA\b|\bNVIDIA\b",
    "MLOps tooling": r"\bMLflow\b|\bKubeflow\b|\bMLOps\b|model registry|feature store",
    "LangChain / LlamaIndex": r"\bLangChain\b|\bLangGraph\b|\bLlamaIndex\b",
}

# The threshold at which a gap stops being trivia and becomes a decision.
#
# Deliberately measured on **Req%**, not prevalence. A first version alerted on
# tagged prevalence and immediately flagged "AI Agents" (16.7% of NOW) as a
# high-demand gap — while its Req% is 5.6%, i.e. it is named as context far more
# often than it gates an application. That is precisely anti-pattern #1 in
# .agents/AGENTS.md ("do NOT quote a headline prevalence as what to learn"), and
# the alert existed to prevent exactly that mistake, so it must not make it.
GAP_ALERT_REQ_PCT = 10.0


def load():
    jobs, seen = [], set()
    for f in sorted(glob.glob(DATA_GLOB)):
        m = re.search(r"jobs_24h_(\d{8})_", os.path.basename(f))
        date = m.group(1) if m else "00000000"
        for j in json.load(open(f, encoding="utf-8")):
            if j.get("id") in seen:
                continue
            seen.add(j["id"])
            j["_date"] = date
            jobs.append(j)
    return jobs


def cv_tools(jobs):
    """Reconstruct the CV's tool list from WhyMatched (see module docstring)."""
    cv = set()
    for j in jobs:
        m = re.search(r"\((.*?)\)", j.get("WhyMatched", "") or "")
        if m and "no shared" not in m.group(1):
            cv |= {t.strip() for t in m.group(1).split(",") if t.strip()}
    cv.discard("...")
    return cv


def pct_tagged(pop, skill):
    if not pop:
        return 0.0
    return sum(1 for j in pop if skill in (j.get("TechSkills") or [])) / len(pop) * 100


def req_pct(pop, rx):
    if not pop:
        return 0.0
    n = 0
    for j in pop:
        t = j.get("descriptionText", "") or ""
        if rx.search(t) and u.skill_modality(t, rx) == "required":
            n += 1
    return n / len(pop) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true",
                    help="overwrite the stored baseline with today's numbers")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    jobs = load()
    now = [j for j in jobs if j.get("SemanticCategory") in NOW_CATS]
    nxt = [j for j in jobs if j.get("SemanticCategory") in NEXT_CATS]
    cv = cv_tools(jobs)
    dates = sorted({j["_date"] for j in jobs})

    print("=" * 72)
    print(f"SKILL GAP REPORT   corpus={len(jobs)}  NOW={len(now)}  NEXT={len(nxt)}")
    print(f"                   {len(dates)} scrape days, {dates[0]} → {dates[-1]}")
    print(f"                   CV tools recovered: {len(cv)}")
    print("=" * 72)

    # ── 1. coverage ──────────────────────────────────────────────────────────
    ranked = sorted(((pct_tagged(now, s), s) for s in TECH_SKILLS_PATTERNS), reverse=True)
    print(f"\n1. COVERAGE — top {args.top} most-demanded NOW skills\n")
    first_gap = None
    for rank, (p, s) in enumerate(ranked[:args.top], 1):
        have = s in cv
        if not have and first_gap is None:
            first_gap = rank
        print(f"   {rank:2}. {s:22} {p:5.1f}%   {'✅' if have else '❌ GAP'}")
    print(f"\n   → first gap at rank #{first_gap if first_gap else '>' + str(args.top)}")

    # ── 2. gaps ──────────────────────────────────────────────────────────────
    print("\n2. GAP — demanded, not on the CV (NOW ≥ 2%)\n")
    print(f"   {'skill':24} {'NOW%':>6} {'NEXT%':>7}")
    gaps = [(p, s) for p, s in ranked if s not in cv and p >= 2.0]
    for p, s in gaps:
        print(f"   {s:24} {p:5.1f}% {pct_tagged(nxt, s):6.1f}%")


    print("\n   Boundary skills by Req% (the decision-relevant number):\n")
    print(f"   {'skill':24} {'NOW req%':>9} {'NEXT req%':>10}")
    boundary = {}
    for name, pat in BOUNDARY_PROBES.items():
        rx = re.compile(pat, re.I)
        a, b = req_pct(now, rx), req_pct(nxt, rx)
        boundary[name] = [round(a, 1), round(b, 1)]
        print(f"   {name:24} {a:8.1f}% {b:9.1f}%")

    # A gap only counts if the skill is REQUIRED often in NOW and absent from
    # the CV. Prevalence alone would fire on context-mentions (see the note by
    # GAP_ALERT_REQ_PCT).
    gap_names = {s for _p, s in gaps}
    alerts = []
    for name, pat in BOUNDARY_PROBES.items():
        if boundary[name][0] < GAP_ALERT_REQ_PCT:
            continue
        if any(g.lower() in name.lower() or name.split()[0].lower() in g.lower()
               for g in gap_names):
            alerts.append(f"{name} ({boundary[name][0]:.1f}% required in NOW)")
    for p, s in gaps:
        rx = TECH_SKILLS_PATTERNS.get(s)
        if rx and req_pct(now, re.compile(rx, re.I)) >= GAP_ALERT_REQ_PCT:
            alerts.append(f"{s} ({req_pct(now, re.compile(rx, re.I)):.1f}% required in NOW)")

    # ── 3. trend ─────────────────────────────────────────────────────────────
    print("\n3. TREND — first half vs second half of the window\n")
    mid = dates[len(dates) // 2]
    early = [j for j in now if j["_date"] < mid]
    late = [j for j in now if j["_date"] >= mid]
    moves = []
    for s in TECH_SKILLS_PATTERNS:
        a, b = pct_tagged(early, s), pct_tagged(late, s)
        if max(a, b) < 3.0:
            continue
        z = u._two_proportion_z(b / 100, len(late), a / 100, len(early))
        if abs(z) >= 1.96 and abs(b - a) >= 3.0:
            moves.append((b - a, a, b, z, s))
    moves.sort(reverse=True)
    if moves:
        for d, a, b, z, s in moves:
            print(f"   {s:22} {a:5.1f}% → {b:5.1f}%  {d:+5.1f}pp  z={z:+.2f}")
        print("\n   ⚠️  At n≈%d per window a significant result can still be a sampling"
              % max(len(early), len(late)))
        print("      artefact. Confirm a move across TWO consecutive weeks before acting.")
    else:
        print("   No skill moved significantly. Nothing is emerging.")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if alerts:
        print(f"  ⚠️  NEW HIGH-DEMAND GAP: {', '.join(alerts)}")
        print(f"     (required in ≥{GAP_ALERT_REQ_PCT:.0f}% of NOW roles and not on the CV"
              " — re-read the plan)")
    else:
        print("  ✅ NOW-track coverage intact — nothing you lack is REQUIRED in "
              f"≥{GAP_ALERT_REQ_PCT:.0f}% of postings.")
        print("     The constraint is EVIDENCE and PIPELINE, not knowledge.")

    prev = {}
    if os.path.exists(BASELINE):
        prev = json.load(open(BASELINE, encoding="utf-8"))
    if prev.get("boundary"):
        print("\n  Change vs baseline (%s):" % prev.get("generated", "?"))
        for name, (a, b) in boundary.items():
            pa, pb = prev["boundary"].get(name, [None, None])
            if pa is None:
                continue
            da, db = a - pa, b - pb
            flag = "  ← moved" if abs(db) >= 3.0 else ""
            print(f"    {name:24} NOW {da:+5.1f}pp   NEXT {db:+5.1f}pp{flag}")
    else:
        print("\n  (no baseline yet — run with --baseline to record one)")

    if args.baseline:
        import datetime
        json.dump({"generated": datetime.date.today().isoformat(),
                   "corpus": len(jobs), "now": len(now), "next": len(nxt),
                   "first_gap_rank": first_gap, "boundary": boundary,
                   "top": [[s, round(p, 1)] for p, s in ranked[:args.top]]},
                  open(BASELINE, "w", encoding="utf-8"), indent=2)
        print(f"\n  baseline written → {os.path.relpath(BASELINE)}")


if __name__ == "__main__":
    main()
