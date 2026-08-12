"""Tests for the reporting layer's role-ladder classifier and lift statistics.

These guard the two things the seniority/leadership sections depend on being
right: that a job title is bucketed onto the correct ladder (IC vs Manager, and
which tier), and that a "signal" is only reported as differentiating when it
actually beats its baseline by more than sampling noise.
"""
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jobs_analytics"))

from update_learning_plan import (  # noqa: E402
    AGENT_INTENTS,
    AI_INTENT_SECTIONS,
    LLM_INTENTS,
    MODALITY_PROBES,
    _lift_rows,
    _pp,
    skill_modality,
    _is_self_mention,
    _looks_generic,
    _norm_term,
    _relevance,
    _singular,
    _SHORT_CATEGORY,
    TREND_WINDOW_DAYS,
    _short_cat,
    _split_windows,
    _trend_marker,
    _two_proportion_z,
    classify_role,
)


# ── role ladder ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("title", [
    "Staff Software Engineer",
    "Principal Cloud Engineer (m/f/d)",
    "Principal AI Engineer | Munich | Hybrid",
    "Distinguished Engineer, Infrastructure",
    "Lead Engineer Infrastructure (Platform Services) (m/w/d)",
    "Lead SRE Engineer (f/m/x)",
    "Senior Staff Platform Engineer",
    "Tech Lead - Cloud Platform",
])
def test_staff_ic_titles(title):
    assert classify_role(title) == ("Staff+", "IC"), title


@pytest.mark.parametrize("title", [
    "Engineering Manager - Platform",
    "Head of Engineering",
    "Director of Software Engineering",
    "Engineering Team Lead (f/m/d) - DevOps & Platform",
    "VP of Engineering",
    "Chief Technology Officer",
    "DevOps Engineering Lead (all genders)",
])
def test_manager_titles(title):
    """People-management roles must NOT land in the Staff IC bucket — mixing them
    made the Staff-bar signal read like a management job."""
    assert classify_role(title) == ("Staff+", "Manager"), title


@pytest.mark.parametrize("title", [
    "Senior DevOps Engineer",
    "Senior Site Reliability Engineer (m/w/d)",
    "(Senior) Cloud Site Reliability Engineer (Platform) (m/f/x)",
    "Expert Development Operations Engineer - SAP Cloud Infrastructure",
])
def test_senior_ic_titles(title):
    assert classify_role(title) == ("Senior", "IC"), title


@pytest.mark.parametrize("title", [
    "Junior Cloud Engineer",
    "Working Student Software Engineering (m/f/d)",
    "DevOps Engineer Intern",
    "Graduate Platform Engineer",
])
def test_junior_titles(title):
    tier, track = classify_role(title)
    assert (tier, track) == ("Junior/Mid", "IC"), title


@pytest.mark.parametrize("title", [
    "AI Sales Lead for Start-Ups",
    "Founding GTM Lead",
    "Data Partnerships Lead (Remote)",
    "BIM Lead",
    "CIVIL LEAD ENGINEER - BERLIN",
    "Construction Manager- Offshore Wind (all genders)",
    "Chief of Staff (m/w/d)",
    "Director of Product Management, Platform (d/f/m)",
    "Freelancer Content Lead - AWS Certified Data Engineer Exam Prep Course",
    "(Senior/Staff) Product Designer",
    "AI Lead - Venture Capital",
    "Engineering Director - Pump Solutions",
])
def test_off_ladder_titles_are_dropped(title):
    """A bare \\blead\\b/\\bstaff\\b match pulled these into the Staff bucket and
    diluted every tier comparison; they must classify as (None, None)."""
    assert classify_role(title) == (None, None), title


def test_untitled_engineer_is_not_assumed_senior():
    """The old code lumped every untitled role into 'Senior'. 70% of that bucket
    carried no seniority word at all, which made the Senior->Staff delta noise."""
    assert classify_role("DevOps Engineer (m/f/d)") == ("Junior/Mid", "IC")


def test_classify_role_handles_non_string():
    assert classify_role(None) == (None, None)
    assert classify_role("") == (None, None)


# ── lift statistics ──────────────────────────────────────────────────────────
def test_two_proportion_z_detects_real_gap():
    # 60% vs 20% on n=100 each is far beyond noise.
    assert abs(_two_proportion_z(0.6, 100, 0.2, 100)) > 1.96


def test_two_proportion_z_ignores_small_sample_noise():
    # The same 40pp gap on n=5 each is not significant.
    assert abs(_two_proportion_z(0.6, 5, 0.2, 5)) < 1.96


def test_two_proportion_z_is_zero_for_empty_or_degenerate_input():
    assert _two_proportion_z(0.5, 0, 0.5, 10) == 0.0
    assert _two_proportion_z(0.0, 10, 0.0, 10) == 0.0


def _jobs(texts):
    return [{"text": t, "title": "x", "cat": "c"} for t in texts]


def test_lift_rows_reports_delta_against_baseline_not_raw_prevalence():
    """The core fix: a signal present in most of the focus set is NOT a finding
    when it is equally present in the baseline."""
    themes = {"ubiquitous": r"security", "real": r"technical strategy"}
    focus = _jobs(["security and technical strategy"] * 10)
    baseline = _jobs(["security only"] * 10)
    rows = {r[0]: r for r in _lift_rows(themes, focus, baseline)}

    # Both are at 100% of the focus set — only the differentiating one lifts.
    assert rows["ubiquitous"][1] == 100.0 and rows["ubiquitous"][3] == 0.0
    assert rows["real"][1] == 100.0 and rows["real"][3] == 100.0
    assert rows["real"][4] is True        # significant
    assert rows["ubiquitous"][4] is False  # no gap at all
    # Sorted by delta descending, so the real signal ranks first.
    assert _lift_rows(themes, focus, baseline)[0][0] == "real"


def test_lift_rows_handles_empty_corpora():
    rows = _lift_rows({"t": r"x"}, [], [])
    assert rows == [("t", 0.0, 0.0, 0.0, False)]


@pytest.mark.parametrize("delta,expected", [
    (0.0, "0 pp"), (-0.2, "0 pp"), (0.4, "0 pp"),
    (12.3, "+12 pp"), (-9.6, "-10 pp"),
])
def test_pp_formatting_avoids_negative_zero(delta, expected):
    assert _pp(delta) == expected


# ── AI umbrella intent taxonomies ────────────────────────────────────────────
# These decide what the plan tells the user "LLMs / GenAI" and "AI Agents"
# actually demand, so a mis-scoped regex here becomes bad career advice.
def _match(intents, label, text):
    import re
    return bool(re.compile(intents[label][0]).search(text.lower()))


def test_intent_taxonomies_are_well_formed():
    import re
    for _, _, intents in AI_INTENT_SECTIONS:
        assert intents, "an intent taxonomy must not be empty"
        for label, (pattern, verdict) in intents.items():
            re.compile(pattern)  # raises if malformed
            assert verdict.strip(), f"{label} needs a verdict — it is the point of the table"


def test_research_intent_ignores_marketing_boilerplate():
    """'state-of-the-art' appears in 11% of NOW-track JDs as pure marketing. An
    earlier draft matched it and made model research look like a real requirement
    for platform roles; it must only fire on genuine research signals."""
    label = "Research / design new model architectures"
    for boilerplate in [
        "We use state-of-the-art technology to solve hard problems",
        "You will build our platform from scratch",
        "Read the latest paper on our engineering blog",
    ]:
        assert not _match(LLM_INTENTS, label, boilerplate), boilerplate
    for real in [
        "You will work as a Research Scientist on our modelling team",
        "Design novel architectures for multimodal reasoning",
        "A track record of publications at top venues",
    ]:
        assert _match(LLM_INTENTS, label, real), real


def test_llm_serving_intent_matches_infra_not_data_science():
    """The serving row is the one the user can act on from an SRE background, so
    it must catch inference-infra vocabulary and not generic ML talk."""
    label = "Serve / operate LLM workloads (vLLM, Triton, GPU scheduling)"
    for infra in [
        "Experience deploying LLMs with serving frameworks such as vLLM or SGLang",
        "Large Language Models at scale served via NVIDIA Triton",
        "You will own GPU cluster scheduling and inference latency",
    ]:
        assert _match(LLM_INTENTS, label, infra), infra
    assert not _match(LLM_INTENTS, label, "We apply machine learning to customer data")


def test_ops_agent_intent_matches_sre_flavoured_agent_work():
    """The NOW->NEXT bridge finding: agents applied to incident response."""
    label = "Agents applied to ops (incident triage, RCA, auto-remediation)"
    for s in [
        "Build and improve AI agents for incident triage, change validation, RCA generation, and auto-remediation",
        "Design and continuously improve AI agents for incident response and auto-remediation",
    ]:
        assert _match(AGENT_INTENTS, label, s), s


def test_consume_vs_train_intents_are_distinguishable():
    """The user's core question: is the ask 'use an API' or 'train a model'?
    A JD about shipping an LLM feature must not read as a training requirement."""
    consume = "Design and ship LLM-powered features end to end with prompt engineering and structured output"
    assert _match(LLM_INTENTS, "Consume LLM APIs — prompting, tool calling, structured output", consume)
    assert not _match(LLM_INTENTS, "Fine-tune / adapt existing models (LoRA, SFT)", consume)
    assert not _match(LLM_INTENTS, "Research / design new model architectures", consume)

    train = "You will run fine-tuning jobs with LoRA and DPO on our training pipeline"
    assert _match(LLM_INTENTS, "Fine-tune / adapt existing models (LoRA, SFT)", train)


# ── Emerging-skills radar hygiene ────────────────────────────────────────────
# The radar is fed by GLiNER, which happily emits employer names, benefit
# vendors, off-track domain vocabulary and generic nouns as if they were tools.
# These guard the structural filters that keep it readable — the failure mode
# they exist for is a plan section reading `zalando` / `jobrad` / `heat pumps`
# as "tools going mainstream".

def test_self_mention_detects_employer_naming_itself():
    """Biggest single noise source: 100% of `zalando` came from Zalando."""
    assert _is_self_mention("zalando", "Zalando")
    assert _is_self_mention("personio", "Personio SE")
    assert _is_self_mention("celonis", "Celonis GmbH")
    assert _is_self_mention("langdock", "Langdock GmbH")


def test_self_mention_is_per_occurrence_not_per_term():
    """A vendor that also hires here must keep its OTHER employers' mentions,
    or we'd lose genuinely emerging tools like n8n and Langfuse."""
    assert _is_self_mention("n8n", "n8n")
    assert not _is_self_mention("n8n", "Personio")
    assert not _is_self_mention("langfuse", "JetBrains")
    assert not _is_self_mention("kubernetes", "Zalando")


def test_self_mention_ignores_too_short_or_empty():
    assert not _is_self_mention("go", "Google")
    assert not _is_self_mention("", "Zalando")
    assert not _is_self_mention("terraform", "")


@pytest.mark.parametrize("term", [
    "networking fundamentals", "operational readiness", "service boundaries",
    "edge cases", "public cloud", "ai-first", "clean code", "cost-efficiency",
    "data sovereignty", "incident response", "pragmatism", "idempotency",
    "logs", "runbooks", "templates", "servers", "saas platforms",
    "ai platform", "ai coding assistants", "webhooks", "qa", "medical devices",
])
def test_generic_phrases_are_rejected(term):
    assert _looks_generic(term), f"{term!r} should read as a generic concept"


@pytest.mark.parametrize("term", [
    "langfuse", "sentry", "n8n", "tensorrt", "ceph", "slurm", "litellm",
    "bicep", "neo4j", "infiniband", "cloudflare", "parquet", "notion",
    "temporal", "victoriametrics", "prefect", "qdrant",
])
def test_real_tool_names_survive(term):
    assert not _looks_generic(term), f"{term!r} is a real product name"


def test_language_levels_are_rejected():
    """CEFR levels reached the radar as `c1+` — the analyzer only blocks bare a1-c2."""
    for t in ("b2", "c1", "c1+", "C2-", "a1"):
        assert _looks_generic(t), t


def test_separator_variants_normalize_to_the_same_form():
    """`infrastructure-as-code` must hit the same blocklist entry as the
    spaced form, instead of slipping past it on punctuation alone."""
    assert _norm_term("infrastructure-as-code") == "infrastructure as code"
    assert _norm_term("  Cost-Efficiency ") == "cost efficiency"
    assert _norm_term("CI/CD") == "ci cd"


def test_singularizer_does_not_mangle_plain_s_plurals():
    """`cases` is a plain +s plural; an over-eager -ses rule turned it into
    `ca` and let `edge cases` through the generic-word check."""
    assert _singular("cases") == "case"
    assert _singular("databases") == "database"
    assert _singular("releases") == "release"
    assert _singular("boxes") == "box"
    assert _singular("libraries") == "library"
    # The singularizer is deliberately crude and WILL mangle names that merely
    # end in -s ("kubernetes" -> "kubernete"). That is harmless by design: the
    # result is only ever used as a set lookup, never shown, so a mangled form
    # matters solely if it collides with a generic word. It must not.
    assert not _looks_generic("kubernetes")
    assert not _looks_generic("prometheus")


def test_relevance_prefers_fitscore_and_falls_back():
    """~75% of the historical corpus predates FitScore; those rows must still
    be gradeable or the radar would silently drop most of its input."""
    assert _relevance({"FitScore": 72.0, "SemanticMatchScore": 10.0}) == 72.0
    assert _relevance({"SemanticMatchScore": 61.5}) == 61.5
    assert _relevance({}) == 0.0


# ── Time windows (P1: trend axis) ────────────────────────────────────────────
# Every table used to be an all-time cumulative aggregate, which cannot tell a
# tool that peaked months ago from one climbing now. These guard the window
# split, which must stay balanced and must refuse to invent a trend from too
# little history.

def _dates(*days):
    return [f"2026-07-{d:02d}" for d in days]


def test_window_split_is_even_when_history_is_short():
    """The corpus is only ~19 days deep, so a true 30/30 split would leave the
    prior window empty. Below 2x the window we split the scrape DATES evenly."""
    recent, prior = _split_windows(_dates(13, 14, 15, 16, 20, 21, 22, 23))
    assert len(recent) == 4 and len(prior) == 4
    assert max(prior) < min(recent), "windows must not overlap"


def test_window_split_refuses_too_little_history():
    """Better no trend column than a trend built on two scrape days."""
    assert _split_windows(_dates(30, 31)) == ([], [])
    assert _split_windows(_dates(29, 30, 31)) == ([], [])
    assert _split_windows([]) == ([], [])


def test_window_split_uses_real_days_once_history_is_deep():
    deep = [f"2026-{m:02d}-{d:02d}" for m in (4, 5, 6, 7) for d in (1, 10, 20)]
    recent, prior = _split_windows(deep)
    assert recent and prior
    assert max(prior) < min(recent)
    # Recent side must sit inside the trailing TREND_WINDOW_DAYS
    newest = datetime.strptime(max(deep), "%Y-%m-%d")
    assert all((newest - datetime.strptime(d, "%Y-%m-%d")).days <= TREND_WINDOW_DAYS
               for d in recent)


def test_window_split_ignores_unparseable_dates():
    recent, prior = _split_windows(_dates(13, 14, 15, 16, 20, 21, 22, 23) + ["Unknown"])
    assert len(recent) + len(prior) == 8


def test_trend_marker_needs_significance_not_just_a_bigger_number():
    """A rise that is within sampling noise must render flat, or every table
    turns into a wall of arrows that mean nothing."""
    # Tiny sample, big-looking jump -> not significant
    marker, delta = _trend_marker(3, 10, 1, 10)
    assert marker == "▪️"
    # Large sample, clear move -> significant
    marker, delta = _trend_marker(300, 1000, 150, 1000)
    assert marker == "📈" and delta > 0
    marker, delta = _trend_marker(150, 1000, 300, 1000)
    assert marker == "📉" and delta < 0


def test_trend_marker_handles_empty_windows():
    assert _trend_marker(0, 0, 5, 10) == ("", None)
    assert _trend_marker(5, 10, 0, 0) == ("", None)


def test_short_category_names_stay_narrow_enough_for_a_column_header():
    """Each tracked category becomes one table column, so a long name blows the
    table width out. `_short_cat` must shorten every mapped name and pass
    unknown ones through unchanged rather than raising."""
    for full, short in _SHORT_CATEGORY.items():
        assert _short_cat(full) == short
        assert len(short) <= 18, f"{short!r} is too wide for a column header"
    assert _short_cat("Some New Category") == "Some New Category"


# ── skill modality: required vs alternative vs optional ──────────────────────
# These guard the distinction that a skill TAG cannot make. On 2026-08-07 the
# plan was rewritten because "Go" scored as a top-3 demand signal on raw
# prevalence, while the sentences showed most mentions were alternatives. If
# these tests regress, every prevalence-driven recommendation in the plan
# silently goes back to conflating "must know" with "one of four options".
GO = MODALITY_PROBES["Go"]


@pytest.mark.parametrize("text,expected", [
    # An explicit MUST beats an optional cue elsewhere in the same bullet: the
    # "is a plus" here belongs to Bash, not to Go. Proximity decides.
    ("Proficient in Go and Python is a MUST (additionally Bash is a plus)", "required"),
    ("• Solid software development skills in Go (strongly preferred, since our "
     "IaC runs on Pulumi in Go) or Python", "required"),
    ("• Strong Go (Golang) development skills with experience building "
     "production services, tools, or platform components.", "required"),
    ("• You write Go, you understand how Kubernetes components interact", "required"),
    # Slash-joined stack lists are alternatives, not requirements.
    ("• Programming: Bash/Python/Go\n• Linux OS: any major distribution", "alternative"),
    # An explicit quantifier downgrades an adjacent required cue.
    ("• Proficiency in at least one scripting language (Python, TypeScript, Go, Bash).",
     "alternative"),
    ("• Experience with one or more programming languages (e.g. Go, Python, Java, etc)",
     "alternative"),
    ("• Strong Python programming skills (required) plus one of: Go, TypeScript, or Rust.",
     "alternative"),
    # Section header wins outright, even though the bullet itself reads neutral.
    ("Nice to have:\n• Interest in building a Kubernetes operator (Go/Rust)", "optional"),
    ("It would be fantastic if you:\n• Have hands-on experience with backend "
     "programming languages (e.g., Go, Python, Ruby)", "optional"),
])
def test_skill_modality_classifies_real_jd_phrasing(text, expected):
    assert skill_modality(text, GO) == expected


def test_skill_modality_returns_none_when_absent():
    assert skill_modality("We use Python and Terraform exclusively.", GO) is None


def test_strongest_verdict_wins_across_mentions():
    """A JD that lists Go in its stack blurb AND requires it in the
    requirements is a Go job — the requirement must not be diluted by the
    casual mention."""
    text = ("Our stack: Python/Go/Bash.\n"
            "Requirements:\n• Strong Go development experience.")
    assert skill_modality(text, GO) == "required"


@pytest.mark.parametrize("phrase", [
    "Become the go-to expert for diagnosing issues",
    "we get alerted when things go wrong",
    "don't just go along for the ride",
    "convince a CISO that the agent won't go rogue",
    "ship, learn from what breaks, and go again",
    "collaborate with Go-To-Market and Engineering",
    "Go-lives & hypercare: support for pilots",
])
def test_go_probe_rejects_english_verb_and_compound_uses(phrase):
    """`TECH_SKILLS_PATTERNS["Go"]` matches all of these — every one is present
    in the real corpus and each inflates Go's apparent demand. Case sensitivity
    on bare "Go" is load-bearing here; re.I re-admits the lowercase verbs."""
    assert not GO.search(phrase), f"false positive on {phrase!r}"


@pytest.mark.parametrize("phrase", [
    "Strong experience with Go and Kubernetes",
    "proficiency in Golang",
    "we write our operators in golang",
    "Staff Software Engineer (Ruby or GOLANG)",
    "Programming: Bash/Python/Go",
])
def test_go_probe_still_matches_real_mentions(phrase):
    """The negative lookaheads must not cost real hits — the failure mode that
    matters more than the false positives."""
    assert GO.search(phrase), f"false negative on {phrase!r}"


def test_every_modality_probe_compiles_and_is_non_trivial():
    for name, rx in MODALITY_PROBES.items():
        assert rx.pattern, f"{name} has an empty pattern"
        assert skill_modality("", rx) is None
