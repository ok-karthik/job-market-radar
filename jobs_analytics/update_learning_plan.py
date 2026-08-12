import json
import glob
import math
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import os

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# Calculate absolute paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Re-use the analyzer's suppression sets so the emerging-skills radar renders
# clean even for historical *_semantic.json produced before those sets existed
# (the EmergingSkills lists in old files still contain git/react/observability
# etc). Best-effort: falls back to a frequency threshold alone if src isn't
# importable (e.g. running from an odd cwd without the package on the path).
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))
try:
    import re as _re
    from scraper.semantic_job_analyzer import (
        ESTABLISHED_NON_NOVEL, SOFT_SKILL_BLOCKLIST,
        TECH_SKILLS_PATTERNS, CASE_SENSITIVE_SKILLS,
    )
    _EMERGING_SUPPRESS = {s.lower() for s in (ESTABLISHED_NON_NOVEL | SOFT_SKILL_BLOCKLIST)}
    # Terms that now match the canonical taxonomy must not linger in the radar
    # either. Old *_semantic.json parked them in EmergingSkills before they were
    # promoted (e.g. `redshift`, plural `vector databases`); re-checking the
    # taxonomy here retroactively cleans them without regenerating those files.
    _CANON_PATTERNS = [
        _re.compile(pat, 0 if name in CASE_SENSITIVE_SKILLS else _re.IGNORECASE)
        for name, pat in TECH_SKILLS_PATTERNS.items()
    ]
    def _now_canonical(term: str) -> bool:
        return any(p.search(term) for p in _CANON_PATTERNS)
except Exception:
    _EMERGING_SUPPRESS = set()
    # The skill audit iterates this; an empty dict degrades to an empty audit
    # table rather than a NameError at render time.
    TECH_SKILLS_PATTERNS = {}
    CASE_SENSITIVE_SKILLS = set()
    def _now_canonical(term: str) -> bool:  # noqa: E301 - best-effort fallback
        return False

# Structural concept detector — the principled counterpart to the hand-curated
# blocklist. GLiNER emits an endless tail of generic *practices/concepts* (not
# products): "monitoring", "infrastructure as code", "design patterns",
# "operational efficiency". Rather than enumerate each, catch them by shape:
# real tool/product names are almost never morphological concept words. This
# keeps the emerging radar self-cleaning as new noise appears.
_GENERIC_WORDS = {
    "engineering", "architecture", "architectures", "practices", "practice",
    "standards", "systems", "system", "concepts", "concept", "skills", "tools",
    "tooling", "experience", "excellence", "management", "design", "development",
    "operations", "optimization", "optimisation", "strategy", "governance",
    "principles", "methodologies", "patterns", "fundamentals", "technologies",
    "technology", "solutions", "services", "processes", "process", "culture",
    "mindset", "frameworks", "best", "modern", "complex", "generative",
    # Generic infra/ops nouns GLiNER emits as if they were products. Listed in
    # the SINGULAR only — `_looks_generic` singularizes before the lookup, so
    # "log"/"logs", "runbook"/"runbooks" are both covered by one entry.
    "log", "metric", "trace", "alert", "dashboard", "runbook", "playbook",
    "template", "server", "cluster", "container", "pipeline", "workflow",
    "script", "report", "application", "platform", "database", "network",
    "team", "environment", "release", "deployment", "incident", "ticket",
    # Quality/outcome nouns that show up in "clean code", "cost efficiency",
    # "incident response", "data sovereignty" — never part of a product name.
    "code", "response", "efficiency", "sovereignty", "scale", "scalability",
    "quality", "reliability", "security", "availability", "performance",
    "cost", "delivery", "ownership", "collaboration", "communication",
    "assistant", "agent", "webhook", "endpoint", "api", "readiness",
    "boundary", "case", "cloud", "qa", "stack", "layer", "component",
    "device", "machine", "tool", "product", "project", "customer", "user",
    "desktop", "cli", "compiler", "driver", "package", "module", "library",
    "protocol", "usage", "memory", "solver", "routing", "storage",
    # Adjectives GLiNER lifts out of "fault-tolerant systems", "relational
    # databases" — descriptors, never product names.
    "tolerant", "scalable", "resilient", "robust", "secure", "reliable",
    "relational", "distributed", "automated", "seamless", "critical",
    # Modifier words that only ever appear in coined phrases ("AI-first",
    # "cloud-native", "data-driven") — never inside a product name.
    "first", "native", "driven", "centric", "led", "based", "ready", "aware",
    "end", "full", "cross", "multi", "hybrid", "public", "private", "open",
}
_GENERIC_SUFFIXES = ("ing", "tion", "sion", "ment", "ility", "ance", "ence",
                     "ization", "isation", "ism", "ency", "ness", " analytics")

# CEFR language levels ("B2", "C1+", "c1-") — GLiNER reads them as skills. The
# analyzer blocklists the bare a1..c2 forms; the decorated variants slip past.
_LANG_LEVEL_RE = re.compile(r'^[abc][12][+\-]?$')

# Separator noise: GLiNER emits both "infrastructure as code" and
# "infrastructure-as-code". The blocklists spell the spaced form, so normalize
# before every lookup rather than duplicating each entry.
_SEPARATORS_RE = re.compile(r'[-_/\\.]+')


def _norm_term(term: str) -> str:
    """Lowercase *term* and flatten separator punctuation to single spaces."""
    return re.sub(r'\s+', ' ', _SEPARATORS_RE.sub(' ', term.strip().lower())).strip()


def _singular(word: str) -> str:
    """Crude English singularizer, good enough for one-word noun checks."""
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    # Only -es plurals that genuinely add two letters ("boxes"->"box").
    # NOT bare "-ses": "cases"/"releases"/"databases" are plain +s.
    if len(word) > 4 and word.endswith(('xes', 'ches', 'shes', 'sses')):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def _looks_generic(term: str) -> bool:
    """True when *term* reads as a generic concept/practice rather than a named
    tool. Conservative: single words must be reasonably long to count, so short
    product names ('Notion') are never caught. Multi-word phrases are generic if
    any word is a concept word or the head word is a concept suffix."""
    t = _norm_term(term)
    if not t:
        return False
    if _LANG_LEVEL_RE.match(t):
        return True
    # Match a word against the set in BOTH its surface and singular form: the
    # set carries some entries as plurals ("fundamentals", "practices") and
    # others as singulars ("log", "runbook"), and singularizing blindly would
    # miss the former.
    def _is_generic_word(w):
        return w in _GENERIC_WORDS or _singular(w) in _GENERIC_WORDS

    words = t.split()
    if len(words) > 1:
        if any(_is_generic_word(w) for w in words):
            return True
        return _singular(words[-1]).endswith(_GENERIC_SUFFIXES)
    # single word: a known concept word, or long enough with a concept suffix
    w = _singular(words[0])
    return _is_generic_word(words[0]) or (len(w) >= 7 and w.endswith(_GENERIC_SUFFIXES))


# ── Emerging-radar relevance gate ────────────────────────────────────────────
# The radar answers "what tool is going mainstream in MY market", so it must be
# fed only jobs that ARE the user's market. Filtering by SemanticCategory does
# NOT work: `categorize_job()` is forced-choice with no "none of the above", so
# off-target postings land in the target tracks anyway (measured 2026-07-31 —
# 68% of `heat pumps` mentions sat inside NOW/NEXT/LATER). The per-job fit
# score is the pipeline's own relevance signal and does work.
EMERGING_RELEVANCE_GATE = 50.0
# A term needs several independent employers behind it to count as a trend.
EMERGING_MIN_COMPANIES = 3
EMERGING_MAX_COMPANY_SHARE = 0.5
# Minimum mentions inside the recent window before a term is ranked on momentum.
EMERGING_MIN_RECENT = 4

# Legal-form and filler tokens stripped before comparing a term to an employer
# name, so "Celonis SE" still matches the term "celonis".
_COMPANY_NOISE_RE = re.compile(
    r'\b(gmbh|mbh|ag|se|kg|ohg|ug|co|inc|corp|ltd|llc|plc|bv|nv|sa|srl|oy|ab|as'
    r'|group|holding|international|technologies|technology|solutions|systems'
    r'|software|labs|digital|deutschland|germany|europe|the)\b')


def _company_key(name: str) -> str:
    """Squash an employer name (or candidate term) to a comparable key."""
    n = re.sub(r'[^a-z0-9 ]', ' ', str(name).lower())
    n = _COMPANY_NOISE_RE.sub(' ', n)
    return re.sub(r'\s+', '', n)


def _is_self_mention(term: str, company: str) -> bool:
    """True when *term* is just the posting company naming itself.

    Employers name themselves throughout their own JDs, and GLiNER dutifully
    extracts it — this alone produced `zalando` (28 mentions, 100% from
    Zalando), `personio` (88%) and `celonis` (74%) in the radar. Dropping the
    occurrence rather than the term keeps genuine vendor-tools whose maker also
    hires here: n8n stays at 30 of its 35 mentions, Langfuse at 24 of 25.
    """
    co, t = _company_key(company), _company_key(term)
    if not co or not t or len(t) < 3:
        return False
    return t in co or co in t


# ── Time windows ─────────────────────────────────────────────────────────────
# Every table in this plan used to be an all-time cumulative aggregate, which
# cannot distinguish a tool that peaked in May and died from one climbing right
# now. Splitting the scrape history into a recent and a prior window turns each
# table from "how common" into "how common, and moving which way".
TREND_WINDOW_DAYS = 30
TREND_MIN_DATES_PER_SIDE = 3


def _split_windows(dates):
    """Split scrape dates into (recent, prior) windows.

    Prefers a true ``TREND_WINDOW_DAYS``-vs-``TREND_WINDOW_DAYS`` comparison,
    but the corpus is currently only ~19 days deep, which would leave the prior
    window empty. Below 2x the window we fall back to an even split of the
    available scrape DATES (not calendar days — scrapes skip days), so the
    comparison stays balanced while history builds up. Returns two sorted lists;
    either may be empty, and callers must handle that.
    """
    parsed = sorted({d for d in dates if _DATE_RE.match(str(d))})
    if len(parsed) < TREND_MIN_DATES_PER_SIDE * 2:
        return [], []
    span = ((datetime.strptime(parsed[-1], "%Y-%m-%d")
             - datetime.strptime(parsed[0], "%Y-%m-%d")).days)
    if span >= TREND_WINDOW_DAYS * 2:
        cutoff = datetime.strptime(parsed[-1], "%Y-%m-%d") - timedelta(days=TREND_WINDOW_DAYS)
        prior_start = cutoff - timedelta(days=TREND_WINDOW_DAYS)
        recent = [d for d in parsed if datetime.strptime(d, "%Y-%m-%d") > cutoff]
        prior = [d for d in parsed
                 if prior_start < datetime.strptime(d, "%Y-%m-%d") <= cutoff]
    else:
        mid = len(parsed) // 2
        prior, recent = parsed[:mid], parsed[mid:]
    if len(recent) < TREND_MIN_DATES_PER_SIDE or len(prior) < TREND_MIN_DATES_PER_SIDE:
        return [], []
    return recent, prior


def _window_label(dates):
    """'Jul 24–31 (8 scrape days)' for a window's date list."""
    if not dates:
        return "n/a"
    a = datetime.strptime(dates[0], "%Y-%m-%d")
    b = datetime.strptime(dates[-1], "%Y-%m-%d")
    span = a.strftime("%b %-d") + ("" if a == b else f"–{b.strftime('%-d' if a.month == b.month else '%b %-d')}")
    return f"{span} ({len(dates)} scrape day{'s' if len(dates) != 1 else ''})"


def _trend_marker(recent_with, recent_n, prior_with, prior_n):
    """(marker, delta_pp) for a recent-vs-prior prevalence comparison.

    Reuses the same two-proportion z-test as every other lift table here, so
    "rising" means rising beyond sampling noise rather than merely a bigger
    number this fortnight.
    """
    if not recent_n or not prior_n:
        return "", None
    p_r, p_p = recent_with / recent_n, prior_with / prior_n
    delta = (p_r - p_p) * 100
    z = _two_proportion_z(p_r, recent_n, p_p, prior_n)
    if abs(z) >= 1.96 and abs(delta) >= 2:
        return ("📈" if delta > 0 else "📉"), delta
    return "▪️", delta


# Column headers get one category name each, so the full SemanticCategory
# strings are too wide to read side by side.
_SHORT_CATEGORY = {
    "Platform Engineering": "Platform Eng",
    "Site Reliability Engineering (SRE)": "SRE",
    "DevOps Engineering": "DevOps",
    "Cloud Engineering": "Cloud Eng",
    "AI Infrastructure": "AI Infra",
    "MLOps": "MLOps",
    "AI Solutions Architecture": "AI Solutions Arch",
    "Staff / Principal Engineering": "Staff/Principal",
    "Solutions Architecture": "Solutions Arch",
}


def _short_cat(cat: str) -> str:
    return _SHORT_CATEGORY.get(cat, cat)


def _relevance(job: dict) -> float:
    """Per-job relevance to the CV, on a 0-100 scale.

    Prefers the composite `FitScore`, falling back to `SemanticMatchScore` for
    the ~75% of the historical corpus written before FitScore existed. Both are
    min-max normalized per run, so the gate reads as "an above-median match on
    the day it was scraped" rather than an absolute bar.
    """
    v = job.get('FitScore')
    if v is None:
        v = job.get('SemanticMatchScore')
    return float(v) if v is not None else 0.0


DATA_DIR = os.path.join(SCRIPT_DIR, '../jobs_output')
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))
from scraper import user_config as _cfg  # noqa: E402
MARKDOWN_PATH = os.path.join(SCRIPT_DIR, '..', _cfg.PLAN_MARKDOWN)


# ── Role ladder: tier × track ────────────────────────────────────────────────
# LinkedIn's `seniorityLevel` is unusable for this (Staff and Senior both land
# in "Mid-Senior level", most rows are "Not Applicable"), so the TITLE is the
# only reliable signal. We read TWO independent axes off it, because they answer
# different questions and a single "staff|principal|lead|head of" regex
# conflated them badly:
#   tier  — Junior/Mid → Senior → Staff+   (how far up the ladder)
#   track — IC vs Manager                  (which ladder)
# "Engineering Team Lead" and "Staff Software Engineer" are both Staff+, but
# only the latter is the IC bar this plan targets; mixing them made the Staff
# signal read like a management job. Titles that aren't a software/infra
# engineering role at all (Sales Lead, BIM Lead, Head of Billing, Construction
# Manager) classify as (None, None) and are dropped from every tier view — a
# bare `\blead\b` was pulling ~40% junk into the Staff bucket.
OFF_LADDER_RE = re.compile(
    r"\b(sales|gtm|go[- ]to[- ]market|account (executive|manager)|business development|"
    r"partnerships?|marketing|content|curriculum|recruit\w*|talent|people ops|"
    r"finance|controller|accounting|procurement|legal|customer success|community|"
    r"designer|\bux\b|brand|editor|copywriter|chief of staff|venture capital|billing|"
    r"product (manage\w+|owner|lead|director|designer)|program manager|project manager|"
    r"\bbim\b|civil|mechanical|electrical|construction|hvac|automotive|acoustic|"
    r"offshore|wind|pump|logistics|warehouse|supply chain|retail|facility|"
    r"manufacturing|maintenance|test manager|quality manager)\b")

# Must look like a software/infra engineering role at all.
ENG_CONTEXT_RE = re.compile(
    r"\b(engineer|engineering|developer|architect|sre|devops|devsecops|platform|"
    r"infrastructure|cloud|software|kubernetes|site reliability|"
    r"technology|technical|\bit\b|data|\bml\b|\bai\b)\b")

# People-management track (owns headcount / org design) vs the IC ladder.
MANAGER_RE = re.compile(
    r"\b(engineering manager|manager,? engineering|head of|director|vp of|vice president|"
    r"chief technology|\bcto\b|team lead|teamlead|leiter|people manager|line manager|"
    r"development manager|delivery manager|engineering lead)\b")
# 'Lead' on the IC ladder — a tech lead is a senior IC, not a people manager.
IC_LEAD_RE = re.compile(
    r"\b(tech(nical)? lead|lead engineer|lead architect|lead developer|lead sre)\b")

STAFF_TIER_RE = re.compile(r"\b(staff|principal|distinguished|fellow)\b")
SENIOR_TIER_RE = re.compile(r"\b(senior|snr|sr\.?|expert)\b")
JUNIOR_TIER_RE = re.compile(
    r"\b(junior|jr\.?|intern|graduate|working student|werkstudent|"
    r"trainee|entry[- ]level|apprentice|praktikant)\b")

# Categories that are never an engineering-ladder role even when the title looks
# managerial (used to keep the manager corpus free of PM/sales leadership).
NON_ENGINEERING_CATEGORIES = {"Product Management", "Technical Sales & Pre-sales"}


def classify_role(title):
    """Return ``(tier, track)`` for a job title.

    ``tier`` is ``'Junior/Mid'`` | ``'Senior'`` | ``'Staff+'`` and ``track`` is
    ``'IC'`` | ``'Manager'``. Returns ``(None, None)`` when the title is not a
    software/infrastructure engineering role, so off-ladder postings never
    pollute the seniority comparisons.
    """
    t = str(title).lower()
    if OFF_LADDER_RE.search(t) or not ENG_CONTEXT_RE.search(t):
        return None, None
    if JUNIOR_TIER_RE.search(t):
        return 'Junior/Mid', 'IC'
    # Management track wins unless the title also carries an explicit IC tier
    # word ("Staff Engineering Manager" is vanishingly rare; "Principal
    # Engineer, Head of Platform" should read as IC).
    if MANAGER_RE.search(t) and not STAFF_TIER_RE.search(t):
        return 'Staff+', 'Manager'
    if STAFF_TIER_RE.search(t) or IC_LEAD_RE.search(t):
        return 'Staff+', 'IC'
    if SENIOR_TIER_RE.search(t):
        return 'Senior', 'IC'
    return 'Junior/Mid', 'IC'


# ── Skill modality: is it REQUIRED, an ALTERNATIVE, or OPTIONAL? ─────────────
# A skill tag increments identically for "Proficient in Go and Python is a MUST"
# and "Programming: Bash/Python/Go". Every prevalence figure in this report
# therefore conflates "you must know this" with "we listed it among four
# interchangeable options".
#
# That conflation produced a real, costly error on 2026-08-07: Go read as a
# top-3 demand signal (~15-20% prevalence), but reading the sentences showed the
# large majority of those mentions were alternatives — only ~6% of NOW-track
# roles actually gate on it, and the ones that do overwhelmingly want Kubernetes
# controllers specifically. Better *extraction* cannot fix this; the information
# lives in the grammar around the hit, not in the hit.
#
# Prototyped and hand-validated in jobs_analytics/scratch/scratch_modality.py.

# Section headers dominate sentence wording: a JD listing "Kubernetes operator
# (Go/Rust)" as a plain bullet under "Nice to have:" reads as required without
# the header context.
_OPTIONAL_HEADER = re.compile(
    r"^\s*[^a-z0-9]{0,3}\s*("
    r"nice[- ]to[- ]have|nice to haves|bonus( points)?|"
    r"(it )?would be (a plus|fantastic|great|nice)|preferred qualifications|"
    r"desirable|desired( experience| skills)?|optional|"
    r"additional (desired|skills)|plus(es)?|stand out|"
    r"you might also|extra credit|advantageous|what (would )?set you apart"
    r")\b", re.I)

_REQUIRED_HEADER = re.compile(
    r"^\s*[^a-z0-9]{0,3}\s*("
    r"must[- ]haves?|hard requirements?|required qualifications|requirements|"
    r"basic qualifications|minimum qualifications|essential|core (technical )?skills|"
    r"what you('ll| will)? need|what we (expect|require)|you must"
    r")\b", re.I)

_OPTIONAL_CUE = re.compile(
    r"\b(nice to have|would be (a plus|an advantage|beneficial|nice)|is a plus|are a plus|"
    r"a plus\b|bonus|ideally|preferably|familiarity with|exposure to|"
    r"awareness of|not required|optional|advantageous|an advantage|"
    r"willingness to learn|open to learn|eager to learn|some knowledge|"
    r"basic (knowledge|understanding)|appreciate|desirable)\b", re.I)

_REQUIRED_CUE = re.compile(
    r"\b(must|required|strong|deep|expert(ise)?|advanced|proficien\w*|solid|"
    r"extensive|excellent|mastery|significant experience|proven|"
    r"in[- ]depth|hands[- ]on experience (with|in)|track record|"
    r"you write|we write|written in|you('ll| will)? (write|build|develop)|"
    r"coding in|develop(ing)? in|fluent in)\b", re.I)

# An explicit quantifier means the skill is one of several accepted options,
# and it OVERRIDES an adjacent required cue: "strong skills in at least one of
# X, Y" is not a requirement for X.
_ALT_QUANTIFIER = re.compile(
    r"\b(one or more|at least one|one of|any of|either|such as|for example|e\.g\.|"
    r"or (similar|comparable|equivalent|another)|or comparable)\b", re.I)


def _split_blocks(text):
    """[(header_or_None, block_text)] — bullets/lines tagged with their section.

    Descriptions carry '• ' bullets injected by the scraper (see
    apify_replica.populate_job_details), so bullets are reliable block edges.
    """
    blocks = []
    header = None
    for raw in re.split(r"\n|(?=•\s)", text):
        line = raw.strip()
        if not line:
            continue
        stripped = line.lstrip("•").strip()
        if not line.startswith("•") and len(stripped) < 80:
            if _OPTIONAL_HEADER.match(stripped):
                header = "optional"
                continue
            if _REQUIRED_HEADER.match(stripped):
                header = "required"
                continue
        if _OPTIONAL_HEADER.match(stripped):
            header = "optional"
        elif _REQUIRED_HEADER.match(stripped):
            header = "required"
        blocks.append((header, stripped))
    return blocks


def _alternative_shape(span, rx):
    """True when *rx* sits in an enumeration of interchangeable options."""
    if _ALT_QUANTIFIER.search(span):
        return True
    m = rx.search(span)
    if not m:
        return False
    lo, hi = max(0, m.start() - 60), min(len(span), m.end() + 60)
    around = span[lo:hi]
    if re.search(r"\bor\b", around, re.I) and re.search(r"[,/]", around):
        return True
    if re.search(r"\w\s*/\s*\w", around):          # "Bash/Python/Go"
        return True
    return False


def _nearest_cue(span, hit_pos):
    """'required' | 'optional' | None — whichever cue sits closest to the hit.

    Proximity is load-bearing: one bullet routinely carries both, as in
    "Proficient in Go and Python is a MUST (additionally Bash is a plus)",
    where the optional cue belongs to Bash rather than to Go.
    """
    def _closest(rx):
        best = None
        for m in rx.finditer(span):
            d = min(abs(m.start() - hit_pos), abs(m.end() - hit_pos))
            if best is None or d < best:
                best = d
        return best

    req, opt = _closest(_REQUIRED_CUE), _closest(_OPTIONAL_CUE)
    if req is None and opt is None:
        return None
    if opt is None:
        return "required"
    if req is None:
        return "optional"
    return "required" if req <= opt else "optional"


def skill_modality(text, rx):
    """'required' | 'alternative' | 'optional' | None for one skill in one JD.

    Precedence:
      1. an explicit optional SECTION header wins outright ("Nice to have:")
      2. otherwise the modality cue NEAREST the hit wins
      3. an explicit alternative quantifier downgrades a 'required' verdict
      4. with no cues, list shape decides; the default is 'alternative', which
         is the conservative reading

    The strongest verdict across all mentions wins: a JD that lists Go in its
    stack blurb and also says "strong Go" in the requirements is a Go job.
    """
    verdicts = []
    for header, block in _split_blocks(text):
        m = rx.search(block)
        if not m:
            continue
        if header == "optional":
            verdicts.append("optional")
            continue
        cue = _nearest_cue(block, m.start())
        if cue == "optional":
            verdicts.append("optional")
        elif cue == "required":
            verdicts.append("alternative" if _ALT_QUANTIFIER.search(block) else "required")
        elif _alternative_shape(block, rx):
            verdicts.append("alternative")
        elif header == "required":
            verdicts.append("required")
        else:
            verdicts.append("alternative")
    if not verdicts:
        return None
    for level in ("required", "alternative", "optional"):
        if level in verdicts:
            return level
    return None


# Regexes for the modality view. These are DELIBERATELY separate from
# TECH_SKILLS_PATTERNS: the canonical taxonomy optimises for tagging recall,
# this one optimises for precision on a single well-known term.
#
# Case sensitivity on bare "Go" is load-bearing — re.I re-admits "go wrong",
# "go along for the ride", "go again", "go rogue", all verified present in this
# corpus. The canonical pattern also matches "Go-To-Market" and "Go-live".
MODALITY_PROBES = {
    # The bare-"Go" alternative stays case-SENSITIVE (capital G) so the English
    # verb never matches; only the suffix list inside the lookahead is folded,
    # to catch title-cased compounds like "Go-To-Market".
    "Go": re.compile(
        r"\b(?:[Gg]o[Ll]ang|GOLANG)\b"
        r"|\bGo\b(?!\s*[-–]?\s*(?i:to\b|live|getter|deep|beyond|forth|again|wrong|"
        r"along|rogue|public|the\b|hand[- ]in))"),
    "Python": re.compile(r"\bPython\b", re.I),
    "AWS": re.compile(r"\bAWS\b|\bAmazon Web Services\b"),
    "Kubernetes": re.compile(r"\bKubernetes\b|\bK8s\b|\bEKS\b|\bAKS\b|\bGKE\b", re.I),
    "Terraform": re.compile(r"\bTerraform\b|\bOpenTofu\b", re.I),
    "Pulumi": re.compile(r"\bPulumi\b", re.I),
    "ArgoCD / GitOps": re.compile(r"\bArgo\s?CD\b|\bGitOps\b|\bFluxCD\b", re.I),
    "OpenTelemetry": re.compile(r"\bOpenTelemetry\b|\bOTel\b", re.I),
    "Prometheus": re.compile(r"\bPrometheus\b", re.I),
    "Service mesh (Istio/Linkerd/Cilium)": re.compile(r"\bIstio\b|\bLinkerd\b|\bCilium\b", re.I),
    "K8s operators / controllers": re.compile(
        r"\b(operator|controller|CRD|custom resource)s?\b", re.I),
    "Vault / secrets mgmt": re.compile(r"\bVault\b|\bSecrets Manager\b|\bSOPS\b", re.I),
    "AI coding assistants": re.compile(
        r"\bCursor\b|\bClaude Code\b|\bCopilot\b|\bCodex\b|AI[- ]assisted|"
        r"AI coding|coding (assistant|agent)", re.I),
    "RAG / vector search": re.compile(
        r"\bRAG\b|retrieval[- ]augmented|vector (database|search|db)", re.I),
}


# ── Theme prevalence with a baseline ─────────────────────────────────────────
# A raw "% of Staff JDs mentioning X" is not a signal: JD boilerplate makes
# almost everything look required (security appeared in 47% of Staff JDs — and
# 50% of non-Staff ones). What matters is the LIFT over a comparable baseline,
# so every qualitative table below reports focus% / baseline% / Δpp and sorts by
# Δ. A two-proportion z-test marks which gaps survive the sample size.
def _prevalence(jobs, rx):
    """Fraction of *jobs* whose description matches *rx* (0.0–1.0)."""
    if not jobs:
        return 0.0
    return sum(1 for j in jobs if rx.search(j['text'])) / len(jobs)


def _two_proportion_z(p1, n1, p2, n2):
    """Two-proportion z statistic. 0.0 when the pooled variance is degenerate."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    var = p * (1 - p) * (1 / n1 + 1 / n2)
    if var <= 0:
        return 0.0
    return (p1 - p2) / math.sqrt(var)


def _pp(delta):
    """Format a percentage-point delta, avoiding the ugly '-0 pp' for tiny gaps."""
    r = round(delta)
    return "0 pp" if r == 0 else f"{r:+.0f} pp"


def _lift_rows(themes, focus, baseline):
    """[(theme, focus%, baseline%, delta_pp, significant)] sorted by delta desc."""
    rows = []
    for theme, pattern in themes.items():
        rx = re.compile(pattern)
        p1 = _prevalence(focus, rx)
        p2 = _prevalence(baseline, rx)
        z = _two_proportion_z(p1, len(focus), p2, len(baseline))
        rows.append((theme, p1 * 100, p2 * 100, (p1 - p2) * 100, abs(z) >= 1.96))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


# ── Umbrella probes — one representative regex per market-table row ──────────
# Closes the "backfill modality into the umbrella tables" TODO (2026-08-10).
# The market table used to be MENTION-ONLY: its rows are keyword SETS matched
# against extracted TechSkills, which cannot express whether a JD gates on a
# skill or merely lists it as one of four alternatives. That is what made Go
# look like a top-3 blocker. `skill_modality()` needs a single regex per row, so
# each umbrella gets one here and the Req% column is computed from it.
#
# ⚠️ These trade recall for precision ON PURPOSE, exactly as MODALITY_PROBES
# does. UMBRELLA_DEFINITIONS optimises tagging recall (40+ AWS service names);
# these name only the terms a JD would use when stating a REQUIREMENT. So the
# Req% column is a floor, not an exact count — which is the safe direction for a
# number that drives study decisions. Case sensitivity on bare `Go` is
# load-bearing; `re.I` re-admits the English verb.
UMBRELLA_PROBES = {
    'umbrella_cloud': re.compile(
        r"\bAWS\b|\bAmazon Web Services\b|\bAzure\b|\bGCP\b|\bGoogle Cloud\b", re.I),
    'umbrella_python': MODALITY_PROBES["Python"],
    'umbrella_go': MODALITY_PROBES["Go"],
    'umbrella_k8s': re.compile(
        r"\bKubernetes\b|\bK8s\b|\bEKS\b|\bAKS\b|\bGKE\b|\bDocker\b|"
        r"\bcontaineri[sz]ation\b|\bHelm\b", re.I),
    'umbrella_iac': re.compile(
        r"\bTerraform\b|\bOpenTofu\b|\bTerragrunt\b|\bAnsible\b|\bPulumi\b|"
        r"\binfrastructure[- ]as[- ]code\b|\bIaC\b", re.I),
    'umbrella_cicd': re.compile(
        r"\bCI/?CD\b|\bGitHub Actions\b|\bGitLab CI\b|\bJenkins\b|\bArgo ?CD\b|"
        r"\bGitOps\b|\bcontinuous (integration|delivery|deployment)\b", re.I),
    'umbrella_observability': re.compile(
        r"\bPrometheus\b|\bGrafana\b|\bDatadog\b|\bOpenTelemetry\b|\bOTel\b|"
        r"\bDynatrace\b|\bSplunk\b|\bobservability\b|\bdistributed tracing\b", re.I),
    'umbrella_databases': re.compile(
        r"\bPostgreSQL\b|\bPostgres\b|\bMySQL\b|\bMongoDB\b|\bRedis\b|"
        r"\bDynamoDB\b|\bSnowflake\b|\bdatabases?\b", re.I),
    'umbrella_streaming': re.compile(
        r"\bKafka\b|\bRabbitMQ\b|\bPulsar\b|\bevent[- ]driven\b", re.I),
    'umbrella_ai_assistants': MODALITY_PROBES["AI coding assistants"],
    'umbrella_llms': re.compile(
        r"\bLLMs?\b|\blarge language models?\b|\bGenAI\b|\bgenerative AI\b|"
        r"\bGPT\b|\bOpenAI\b|\bAnthropic\b", re.I),
    'umbrella_rag': MODALITY_PROBES["RAG / vector search"],
    'umbrella_gpu': re.compile(r"\bGPUs?\b|\bCUDA\b|\bNVIDIA\b|\bTPUs?\b", re.I),
    'umbrella_agents': re.compile(
        r"\bagentic\b|\bAI agents?\b|\bmulti[- ]agent\b|\bagent framework\b", re.I),
}


def _render_lift_table(lines, rows, focus_label, baseline_label):
    """Emit a markdown lift table. ✅ = differentiating, ▪️ = same as baseline."""
    lines.append(f"| Signal | {focus_label} | {baseline_label} | Δ | |")
    lines.append("|--------|------------|--------------|---|---|")
    for theme, a, b, d, sig in rows:
        if sig and d >= 5:
            mark = "✅"
        elif sig and d <= -5:
            mark = "🔻"
        else:
            mark = "▪️"
        lines.append(f"| {theme} | {a:.0f}% | {b:.0f}% | **{_pp(d)}** | {mark} |")


# Org-scope / impact signals. Deliberately narrow phrasing: an earlier version
# matched bare words ("security", "influence", "cost") which fired on nearly
# every JD and produced a flat, meaningless table.
IMPACT_THEMES = {
    "Technical strategy / direction":
        r"\b(technical (direction|strategy|vision|roadmap|leadership)|architectural (direction|vision|decisions?)|"
        r"set(ting)? (the )?(technical )?standards|define (the )?architecture|long[- ]term (technical|architectural))\b",
    "Cross-org / multi-team influence":
        r"\b(cross[- ]?team|cross[- ]?functional|multiple teams|org[- ]wide|organi[sz]ation[- ]wide|company[- ]wide|"
        r"across (the )?(company|organi[sz]ation|teams|engineering)|influence without authority|stakeholders? across)\b",
    "Mentorship / growing engineers":
        r"\b(mentor\w*|coach\w*|grow(ing)? (the )?(team|engineers)|knowledge sharing|"
        r"onboard\w* (new )?engineers|technical guidance|upskill\w*)\b",
    "Driving adoption / standards":
        r"\b(drive adoption|drives? the adoption|evangeli[sz]\w*|best practices across|"
        r"establish\w* (standards|guidelines|governance)|advocate\w* for)\b",
    "Deep scale / distributed systems":
        r"\b(distributed systems|\bat scale\b|large[- ]scale|high[- ]scale|high[- ]throughput|"
        r"millions of|petabyte|low[- ]latency|fault[- ]toleran\w+)\b",
    "Platform / API / DevEx building":
        r"\b(internal (developer )?platform|developer (experience|productivity|platform)|self[- ]service|"
        r"golden path|paved (road|path)|backstage|platform as a product)\b",
    "AI-native engineering":
        r"\b(ai[- ]native|ai[- ]driven|ai[- ]assisted|ai transformation|agentic|ai agents?|copilot|cursor|llm[- ](powered|based))\b",
    "Cost / FinOps ownership":
        r"\b(cost optimi[sz]\w*|finops|cloud (cost|spend)|cost efficiency|reduce (the )?cost)\b",
    "Security / compliance ownership":
        r"\b(zero[- ]trust|threat model\w*|soc ?2|iso ?27001|security (posture|architecture|standards)|"
        r"compliance (framework|requirements)|hardening)\b",
    "Incident / on-call leadership":
        r"\b(on[- ]call|incident (response|management|commander)|postmortem|post[- ]mortem|"
        r"blameless|error budget|\bslo\b|\bsli\b)\b",
    "Coding / system-design bar":
        r"\b(algorithms?|data structures?|coding (challenge|interview|exercise|assessment)|"
        r"system design (interview|round)|leetcode|take[- ]?home)\b",
    "Hiring / interviewing bar":
        r"\b(hiring|recruit\w*|interview\w* (process|panel|loop)|headcount|build(ing)? (the|a) team)\b",
    "Exec / stakeholder communication":
        r"\b(executive|c[- ]level|leadership team|board|senior stakeholders|present(ing)? to|communicat\w+.{0,20}stakeholder)\b",
}

# ── What the AI umbrellas actually ASK FOR ───────────────────────────────────
# "LLMs / GenAI" and "AI Agents" are demand labels, not skills: a JD hitting
# either could want anything from "use Copilot daily" to "train a foundation
# model". Collapsing that into one row with a decision like "know the models"
# is unactionable — it doesn't say whether the ask is learnable in a weekend or
# needs a PhD. These taxonomies split each umbrella by INTENT so the plan can
# name the actual requirement and attach a concrete move.
#
# Each entry: label -> (regex, verdict). Verdicts are deliberately opinionated;
# they are the point of the table.
LLM_INTENTS = {
    "Use AI coding assistants in daily work":
        (r"\b(copilot|cursor|claude code|codex|ai[- ]assisted (coding|development)|"
         r"ai coding (tools?|assistants?)|ai[- ]native (developer|engineer|development)|"
         r"leverage ai to|use ai tools)\b",
         "✅ **Already true — make it explicit.** Name the tools and a workflow on your CV."),
    "Consume LLM APIs — prompting, tool calling, structured output":
        (r"\b(openai api|anthropic api|llm api|integrat\w+ (with )?(llms?|ai|genai)|"
         r"prompt engineer\w*|prompting|few[- ]shot|function calling|tool calling|"
         r"structured output|build\w* (features|products|applications) (with|using) (llms?|ai|genai))\b",
         "✅ **Days, not months.** Ship one LLM-backed feature end-to-end and you clear this bar."),
    "Evaluation, guardrails, tracing of LLM output":
        (r"\b(evals?\b|evaluation (framework|harness|pipeline)|hallucinat\w+|guardrails?|"
         r"llm[- ]as[- ]a[- ]judge|red[- ]team\w*|ai safety|responsible ai|ai governance|"
         r"observability for (llms?|ai)|tracing (llms?|agents?))\b",
         "✅ **Highest-leverage gap.** It is an observability problem — your existing SRE instinct transfers directly."),
    "RAG / retrieval / vector search":
        (r"\b(rag\b|retrieval[- ]augmented|vector (database|db|store|search)|embeddings?|"
         r"chunking|semantic search|qdrant|pinecone|weaviate|milvus|chromadb|faiss)\b",
         "✅ **One project is enough.** Conceptual depth + a working demo covers the ask."),
    "Serve / operate LLM workloads (vLLM, Triton, GPU scheduling)":
        (r"\b(vllm|tgi|text generation inference|triton|tensorrt|sglang|ray serve|kserve|"
         r"model serving|inference (server|endpoint|service|latency|optimi\w+|infrastructure)|"
         r"gpu (cluster|scheduling|provision\w*|orchestrat\w*|utilization)|"
         r"quantiz\w+|kv cache|batching|model deployment|serve (llms?|models))\b",
         "✅ **This is the real AI-Infra skill** and the closest to what you already do — "
         "it is Kubernetes + GPUs + latency, not data science."),
    "MLOps lifecycle around models (registry, monitoring, drift)":
        (r"\b(mlflow|kubeflow|model registry|feature store|model monitoring|drift|"
         r"experiment tracking|weights ?& ?biases|wandb|sagemaker|vertex ai|ml pipelines?)\b",
         "🟡 **Conceptual only.** Know the lifecycle vocabulary; you do not need to own it."),
    "Fine-tune / adapt existing models (LoRA, SFT)":
        (r"\b(fine[- ]?tun\w+|\bsft\b|\blora\b|\bpeft\b|\bdpo\b|\brlhf\b|\bgrpo\b|"
         r"instruction tuning|training (runs?|jobs?|pipeline)|distillation|"
         r"pre[- ]?train\w+|megatron|deepspeed|\btrl\b)\b",
         "🟡 **Awareness level.** Know what LoRA/SFT are and when they beat RAG. Rare outside the NEXT track."),
    "Research / design new model architectures":
        # Deliberately narrow: an earlier draft matched "state-of-the-art", which is
        # marketing boilerplate in 11% of NOW JDs and made this row look like a real
        # requirement. Genuine research signals sit near 0% on the tracked roles.
        (r"\b(research scientist|novel (architectures?|models?|methods?)|publications?|"
         r"peer[- ]review\w*|model architecture design|train\w* (a )?foundation model)\b",
         "🔴 **Not being asked of you.** Effectively absent from these tracks — ignore it and do not let it intimidate you."),
}

AGENT_INTENTS = {
    "Build agents on a framework (LangGraph, MCP, multi-agent)":
        (r"\b(langchain|langgraph|llamaindex|crewai|autogen|semantic kernel|pydantic[- ]ai|"
         r"agent (framework|sdk)|\bmcp\b|model context protocol|multi[- ]agent|agent orchestration)\b",
         "✅ **The main ask.** Learn one framework properly (LangGraph or plain tool-calling + MCP)."),
    "Agentic coding workflow as a productivity multiplier":
        (r"\b(agentic (coding|workflow|development)|claude code|cursor|copilot|"
         r"ai[- ]assisted (coding|development))\b",
         "✅ **Already true — make it explicit.** Employers ask about this in interviews now."),
    "Agents applied to ops (incident triage, RCA, auto-remediation)":
        (r"\b(incident (triage|response)|auto[- ]remediation|root cause|\brca\b|"
         r"self[- ]healing|agents? for (incident|ops|operations|infrastructure))\b",
         "✅ **Your unfair advantage.** This is SRE work with an agent on top — "
         "the single best portfolio project for bridging NOW → NEXT."),
    "Customer-facing agent products (chat, support, assistants)":
        (r"\b(customer[- ]facing agents?|conversational (ai|agents?)|chatbots?|"
         r"virtual assistants?|support agents?)\b",
         "🟡 **Product-side.** Nice context, not a platform skill."),
    "Tool / function calling & integration plumbing":
        (r"\b(tool calling|function calling|tool use|agent tools?|api integration for agents?)\b",
         "✅ **Comes free** with the framework work above."),
    "Operate agent infrastructure (sandboxing, scale, tracing)":
        (r"\b(sandbox\w*|agent (runtime|infrastructure|platform|deployment)|"
         r"scal\w+ agents?|agent observability|tracing agents?|agent evaluation)\b",
         "🟡 **Small but rising, and squarely on your track.** Worth watching in this radar."),
}

# (umbrella key, heading, intent taxonomy) — rendered into the umbrella breakdown.
AI_INTENT_SECTIONS = [
    ("umbrella_llms", "LLMs / GenAI", LLM_INTENTS),
    ("umbrella_agents", "AI Agents / Agentic", AGENT_INTENTS),
]


# People-management specifics — only meaningful on the Manager track.
MANAGEMENT_THEMES = {
    "People management / 1:1s / reviews":
        r"\b(1:1s?|one[- ]on[- ]ones?|performance (review|management|cycle)|career (development|growth|path)|"
        r"line management|direct reports|people (management|development))\b",
    "Hiring & team building":
        r"\b(hiring|recruit\w*|headcount|build(ing)? (the|a|out) team|interview\w* (process|panel|loop)|onboarding)\b",
    "Roadmap / delivery ownership":
        r"\b(roadmap|delivery|okrs?|quarterly planning|prioriti[sz]\w*|backlog|milestones?)\b",
    "Budget / cost ownership":
        r"\b(budget|finops|cost optimi[sz]\w*|cloud (cost|spend)|headcount planning|vendor management)\b",
    "Exec / stakeholder communication":
        r"\b(executive|c[- ]level|leadership team|board|senior stakeholders|present(ing)? to|"
        r"communicat\w+.{0,20}stakeholder)\b",
    "Agile / process ownership":
        r"\b(agile|scrum|kanban|sprint|retrospectives?|ways of working|engineering process)\b",
    "Technical direction (hands-on)":
        r"\b(technical (direction|strategy|vision|leadership)|architectural (decisions?|direction)|"
        r"hands[- ]on|still cod\w+|remain technical)\b",
}

def main():
    json_files = glob.glob(os.path.join(DATA_DIR, '*_filtered_semantic.json'))
    
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    # Tracking job counts per category
    category_counts = Counter()
    
    # Tracking tool counts per category: category -> { tool -> count }
    tool_counts = defaultdict(Counter)

    # Tracking daily stats
    daily_stats = defaultdict(Counter)
    daily_skills = defaultdict(lambda: defaultdict(Counter))
    
    # Tracking raw skill occurrences within umbrellas (for breakdown display)
    umbrella_breakdown = defaultdict(Counter)
    
    # Tracking per-job umbrella PREVALENCE: date -> cat -> umbrella -> set of job_ids
    # This fixes the >100% bug by counting "how many jobs mention at least one keyword"
    daily_umbrella_prevalence = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    
    # Salary tracking: per-job salary data for correlation
    salary_jobs = []

    # Emerging (non-canonical) skills surfaced by GLiNER but not yet in the taxonomy
    emerging_counter = Counter()
    # term -> Counter(companyName -> mentions); powers the concentration guard
    emerging_companies = defaultdict(Counter)
    # term -> Counter(date -> mentions); powers the recent-vs-prior trend
    emerging_dates = defaultdict(Counter)
    # date -> number of jobs that passed the relevance gate (radar denominator)
    gated_jobs_by_date = Counter()
    # date -> category -> deduped job count. `daily_stats` counts every row
    # including cross-day duplicates (that IS the day's yield), so it cannot be
    # used as a denominator for any windowed prevalence figure.
    dated_category_counts = defaultdict(Counter)

    # Qualitative corpora, keyed by (tier, track). A job lands in exactly one.
    # Built from the DEDUPED job set so a posting re-scraped on consecutive days
    # doesn't get a double vote in the theme percentages.
    ladder_corpus = defaultdict(list)  # (tier, track) -> [{'title', 'text', 'cat'}]

    UMBRELLA_DEFINITIONS = {
        'umbrella_cloud': {'aws', 'amazon', 'amazon web services', 's3', 'ec2', 'rds', 'lambda', 'ecs', 'eks', 'cloudformation', 'cloudwatch', 'sagemaker', 'bedrock', 'kinesis', 'dynamodb', 'sns', 'sqs', 'step functions', 'iam', 'fargate', 'ecr', 'cdk', 'gcp', 'google cloud', 'google cloud platform', 'bigquery', 'cloud run', 'gke', 'vertex ai', 'pub/sub', 'cloud functions', 'spanner', 'anthos', 'cloud composer', 'azure', 'aks', 'azure devops', 'azure functions', 'cosmos db', 'microsoft azure', 'azure ad', 'entra'},
        'umbrella_python': {'python', 'python3', 'fastapi', 'flask', 'django'},
        'umbrella_go': {'go', 'golang'},
        'umbrella_k8s': {'kubernetes', 'k8s', 'eks', 'aks', 'gke', 'kubectl', 'kustomize', 'docker', 'containerization', 'containers', 'container', 'podman', 'helm', 'helm charts'},
        'umbrella_iac': {'terraform', 'opentofu', 'hcl', 'terragrunt', 'ansible', 'ansible playbooks', 'pulumi', 'cloudformation', 'infrastructure as code'},
        'umbrella_cicd': {'ci/cd', 'cicd', 'continuous integration', 'continuous deployment', 'continuous delivery', 'github actions', 'gitlab ci', 'gitlab', 'jenkins', 'argocd', 'argo cd', 'argo workflows', 'gitops', 'circleci', 'flux', 'fluxcd', 'ci/cd pipelines', 'ci'},
        'umbrella_observability': {'prometheus', 'promql', 'thanos', 'grafana', 'grafana loki', 'loki', 'mimir', 'tempo', 'datadog', 'data dog', 'opentelemetry', 'otel', 'otlp', 'elk', 'elk stack', 'elasticsearch', 'logstash', 'kibana', 'opensearch', 'pagerduty', 'new relic', 'splunk', 'dynatrace', 'observability', 'cloudwatch'},
        'umbrella_streaming': {'kafka', 'confluent', 'kafka streams', 'rabbitmq', 'pulsar', 'apache pulsar', 'event-driven', 'event-driven architectures'},
        'umbrella_databases': {'sql', 'postgresql', 'postgres', 'mongodb', 'mongo', 'redis', 'dynamodb', 'snowflake', 'databricks', 'mysql', 'databases', 'relational databases'},
        'umbrella_ai_assistants': {'cursor', 'copilot', 'claude code', 'codex', 'github copilot', 'cody', 'tabnine', 'ai coding tools', 'ai tools', 'ai tooling'},
        'umbrella_llms': {'llms', 'llm', 'large language models', 'gpt', 'claude', 'anthropic', 'openai', 'chatgpt', 'gemini', 'mistral', 'llama', 'genai', 'generative ai', 'gen ai'},
        'umbrella_rag': {'rag', 'retrieval augmented generation', 'retrieval-augmented generation', 'haystack', 'vector dbs', 'qdrant', 'pinecone', 'weaviate', 'milvus', 'chromadb', 'chroma', 'faiss', 'vector database', 'vector search'},
        'umbrella_gpu': {'gpu', 'gpus', 'cuda', 'nvidia', 'tpu', 'a100', 'h100', 'h200', 'gpu/cuda'},
        'umbrella_agents': set(),  # Uses substring matching below
    }

    # Career-track category buckets (used both in the loop and for the report).
    NOW_CATEGORIES = [
        "Platform Engineering", "Site Reliability Engineering (SRE)",
        "DevOps Engineering", "Cloud Engineering",
    ]
    NEXT_CATEGORIES = ["AI Infrastructure", "MLOps", "AI Solutions Architecture"]
    LATER_CATEGORIES = ["Staff / Principal Engineering", "Solutions Architecture"]
    tracked_categories_set = set(NOW_CATEGORIES + NEXT_CATEGORIES + LATER_CATEGORIES)
    CAT_TO_TRACK_ID = {}
    for _tid, _cats in (("🎯 NOW", NOW_CATEGORIES), ("📈 NEXT", NEXT_CATEGORIES),
                        ("🧭 LATER", LATER_CATEGORIES)):
        for _c in _cats:
            CAT_TO_TRACK_ID[_c] = _tid

    # JD text of jobs hitting the AI umbrellas, per track — feeds the intent
    # breakdown that says WHAT those umbrellas actually demand.
    ai_intent_corpus = defaultdict(lambda: defaultdict(list))  # umbrella -> track -> [text]

    # Full raw JD text per track, deduped by job id — feeds the required-vs-
    # mentioned modality table and the skill audit. Case is preserved on purpose.
    track_corpus = defaultdict(list)  # track_id -> [descriptionText]

    # Senior-vs-Staff demand split on the IC ladder, restricted to the NOW track.
    core_tier_totals = Counter()                       # tier -> job count
    core_tier_prev = defaultdict(lambda: defaultdict(set))  # tier -> umbrella -> ids

    # A posting that straddles midnight is re-scraped by the next 24h window, so
    # the same job appears in two files. Daily stats intentionally keep every
    # row (that IS the day's yield), but every ALL-TIME aggregate below counts a
    # job once — otherwise ~3.5% of the corpus votes twice.
    seen_job_ids = set()

    for f_path in sorted(json_files):
        filename = os.path.basename(f_path)
        match = re.search(r'jobs_(?:24h|7d|1m)_(\d{8})_', filename)
        date_str = match.group(1) if match else "Unknown"
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        with open(f_path, 'r') as file:
            data = json.load(file)
            for job in data:
                # ── Defensive off-target guard (added 2026-08-09) ──────────────
                # semantic_job_analyzer.main() now drops off-target rows at write
                # time, so a freshly generated file has none. This guard matters
                # because THIS SCRIPT GLOBS THE WHOLE HISTORY: any file written
                # before that change still carries them, and a partially
                # regenerated corpus would silently mix clean and polluted days
                # into the same all-time aggregates.
                #
                # Why it matters: job selection here is by `SemanticCategory`,
                # a forced-choice pick with no "none of the above" (AGENTS.md
                # do-NOT-do #2). Measured 2026-08-09, 58% of the jobs in the four
                # NOW categories had no infrastructure term in the title — FPGA
                # engineers, QA leads, a CTO, a customer-success rep. Every
                # prevalence in the rendered tables was diluted ~2x as a result
                # (observability read 27% instead of 47.3%, IaC 27.9% vs 55.6%).
                #
                # 'unclear' is deliberately NOT excluded: it holds the genuinely
                # adjacent roles (Forward Deployed, AI/ML Engineer, Data
                # Engineering) that belong in the NEXT/Track-4 statistics.
                if job.get('RoleFamily') == 'off-target':
                    continue
                cat = job.get('SemanticCategory', 'Unknown')
                job_id = job.get('id', str(id(job)))
                title = str(job.get('title', 'N/A'))
                tier, track = classify_role(title)

                daily_stats[date_str][cat] += 1
                daily_stats[date_str]['Total'] += 1

                if job_id in seen_job_ids:
                    # Still counted in the daily yield above; skip all-time rollups.
                    for skill in job.get('TechSkills', []):
                        daily_skills[date_str][cat][skill.lower()] += 1
                    continue
                seen_job_ids.add(job_id)
                category_counts[cat] += 1
                dated_category_counts[date_str][cat] += 1

                # Collect per-job skills as a set for prevalence counting
                job_skills_lower = set()
                for skill in job.get('TechSkills', []):
                    skill_lower = skill.lower()
                    job_skills_lower.add(skill_lower)
                    tool_counts[cat][skill_lower] += 1
                    daily_skills[date_str][cat][skill_lower] += 1
                    
                    # Track breakdown (frequency - for the sub-category display)
                    if 'agent' in skill_lower:
                        umbrella_breakdown['umbrella_agents'][skill_lower] += 1
                    for umbrella_key, keywords in UMBRELLA_DEFINITIONS.items():
                        if umbrella_key == 'umbrella_agents':
                            continue
                        if skill_lower in keywords:
                            umbrella_breakdown[umbrella_key][skill_lower] += 1
                
                # Track PREVALENCE: does this job mention at least one skill from each umbrella?
                for umbrella_key, keywords in UMBRELLA_DEFINITIONS.items():
                    if umbrella_key == 'umbrella_agents':
                        # Substring match for agents
                        if any('agent' in s for s in job_skills_lower):
                            daily_umbrella_prevalence[date_str][cat][umbrella_key].add(job_id)
                    else:
                        if job_skills_lower & keywords:  # set intersection
                            daily_umbrella_prevalence[date_str][cat][umbrella_key].add(job_id)

                desc_lower = str(job.get('descriptionText', '')).lower()

                # Keep the JD text of AI-umbrella jobs so the breakdown can report
                # what "LLMs / GenAI" and "AI Agents" concretely demand per track.
                track_id = CAT_TO_TRACK_ID.get(cat)
                # Raw (case-preserving) JD text per track — the modality view
                # needs original case, since bare "Go" must stay capital-G.
                if track_id:
                    track_corpus[track_id].append(str(job.get('descriptionText', '')))
                if track_id:
                    for umbrella_key, _, _ in AI_INTENT_SECTIONS:
                        if umbrella_key == 'umbrella_agents':
                            hit = any('agent' in s for s in job_skills_lower)
                        else:
                            hit = bool(job_skills_lower & UMBRELLA_DEFINITIONS[umbrella_key])
                        if hit:
                            ai_intent_corpus[umbrella_key][track_id].append(desc_lower)

                # Senior-vs-Staff demand split across ALL target tracks, IC ladder.
                # Restricted to explicitly-tiered IC roles: lumping every
                # untitled "DevOps Engineer" into "Senior" (70% of that bucket
                # carried no seniority word at all) made the Δ meaningless.
                # Widened from NOW-only to all three tracks on 2026-07-31: the
                # NOW-only split gave Staff n=47, too few for the z-test to
                # separate a real gap from sampling noise, so every row rendered
                # ▪️ ("can't tell") — which reads identically to "no difference"
                # but means something else. All tracks gives n=125.
                if (cat in tracked_categories_set and track == 'IC'
                        and tier in ('Senior', 'Staff+')):
                    core_tier_totals[tier] += 1
                    for umbrella_key, keywords in UMBRELLA_DEFINITIONS.items():
                        if umbrella_key == 'umbrella_agents':
                            if any('agent' in s for s in job_skills_lower):
                                core_tier_prev[tier][umbrella_key].add(job_id)
                        elif job_skills_lower & keywords:
                            core_tier_prev[tier][umbrella_key].add(job_id)

                # Qualitative corpus, bucketed by (tier, track). Off-ladder roles
                # (tier is None) are dropped entirely.
                if tier is not None:
                    ladder_corpus[(tier, track)].append({
                        'title': title,
                        'text': desc_lower,
                        'cat': cat,
                    })

                # Track salary data
                sal_min = job.get('SalaryMin')
                sal_max = job.get('SalaryMax')
                if sal_min is not None:
                    salary_jobs.append({
                        'category': cat,
                        'skills': list(job_skills_lower),
                        'salary_min': sal_min,
                        'salary_max': sal_max,
                    })

                # Emerging skills (GLiNER hits parked outside the canonical
                # taxonomy). Only on-target jobs feed the radar, and a company
                # naming itself doesn't count — see EMERGING_RELEVANCE_GATE.
                if _relevance(job) >= EMERGING_RELEVANCE_GATE:
                    gated_jobs_by_date[date_str] += 1
                    company = job.get('companyName', '')
                    for e in job.get('EmergingSkills', []):
                        if not isinstance(e, str) or not e.strip():
                            continue
                        term = e.strip().lower()
                        if _is_self_mention(term, company):
                            continue
                        emerging_counter[term] += 1
                        emerging_companies[term][company] += 1
                        emerging_dates[term][date_str] += 1

    print(f"Found {sum(category_counts.values())} distinct jobs across target categories.")
    if salary_jobs:
        print(f"Found {len(salary_jobs)} jobs with salary data.")

    # ── Qualitative corpora, derived from the (tier, track) buckets ───────────
    tracked_cats_set = set(NOW_CATEGORIES + NEXT_CATEGORIES + LATER_CATEGORIES)
    staff_ic_jobs = [j for j in ladder_corpus[('Staff+', 'IC')]
                     if j['cat'] in tracked_cats_set]
    senior_ic_jobs = [j for j in ladder_corpus[('Senior', 'IC')]
                      if j['cat'] in tracked_cats_set]
    # Manager track: any tech-engineering leadership role (the tier axis is
    # degenerate here — every people-manager title classifies as Staff+).
    manager_jobs = [j for j in ladder_corpus[('Staff+', 'Manager')]
                    if j['cat'] not in NON_ENGINEERING_CATEGORIES]
    # Baseline for the manager comparison: every IC role, any tier.
    all_ic_jobs = [j for (t, tr), js in ladder_corpus.items() if tr == 'IC'
                   for j in js]
    print(f"Ladder: {len(staff_ic_jobs)} Staff+ IC · {len(senior_ic_jobs)} Senior IC "
          f"· {len(manager_jobs)} Manager · {len(all_ic_jobs)} IC total")

    # The list of core tools to track and their manually curated decisions
    core_tools = [
        ("Cloud Providers (AWS/GCP/Azure)", "umbrella_cloud", "🔴 CRITICAL — multi-account + VPC depth (NOT the cert: 4% mention it, all \"nice to have\")"),
        ("Python", "umbrella_python", "🔴 CRITICAL — keep building"),
        ("Go", "umbrella_go", "🟠 NARROW — only 6% require it; learn controller-runtime, not general Go"),
        ("Kubernetes & Containers", "umbrella_k8s", "🔴 CRITICAL — you have it"),
        ("IaC & Config", "umbrella_iac", "🔴 CRITICAL — you have it"),
        ("CI/CD & GitOps", "umbrella_cicd", "🔴 CRITICAL — you have it"),
        ("Observability", "umbrella_observability", "🟠 HIGH — build projects with it"),
        ("Databases", "umbrella_databases", "🟠 HIGH — conceptual"),
        ("Event Streaming", "umbrella_streaming", "🟡 LOW — conceptual"),
        ("AI Coding Assistants", "umbrella_ai_assistants", "🟢 USE DAILY — Cursor/Copilot/Claude"),
        # These two labels are demand buckets, not skills — the "what exactly?"
        # is answered by the intent breakdown further down, so the decision text
        # points there instead of guessing.
        ("AI Agents / Agentic", "umbrella_agents", "🟠 MEDIUM — build agents on a framework; see intent breakdown"),
        ("LLMs / GenAI", "umbrella_llms", "🟠 MEDIUM — consume + serve, not train; see intent breakdown"),
        ("RAG & Vector Search", "umbrella_rag", "🟠 MEDIUM for AI Infra — conceptual only"),
        ("GPU Computing", "umbrella_gpu", "🟠 HIGH for AI Infra — awareness needed"),
    ]

    # ── Career tracks: buckets of SemanticCategory values ─────────────────────
    # Each track gets its own demand / ROI / salary view so the plan reads as
    # "for the roles you actually target, here's the skill mix and your gaps."
    TRACKS = [
        ("🎯 NOW", "Platform / SRE / DevOps / Cloud (apply now)", NOW_CATEGORIES),
        ("📈 NEXT", "AI Infra / MLOps (high-pay pivot)", NEXT_CATEGORIES),
        ("🧭 LATER", "Staff/Principal IC + Architecture", LATER_CATEGORIES),
    ]

    # All categories feeding a tool-driven track (used for ROI universe).
    tracked_categories = [c for _, _, cats in TRACKS for c in cats]

    # Per-track total job count (all dates), and per-track prevalence for a tool.
    # Both sides use the DEDUPED counts: `category_counts` and
    # `daily_umbrella_prevalence` are only populated for first-seen job ids, so
    # taking the denominator from the raw per-date totals instead would understate
    # every demand percentage by the cross-day duplicate rate.
    def _track_total(categories):
        return sum(category_counts.get(c, 0) for c in categories)

    def _track_demand(internal_name, categories, dates=None):
        """(#jobs mentioning >=1 umbrella keyword, #jobs in scope).

        Categories are disjoint (a job has exactly one SemanticCategory) and each
        job id is recorded on exactly one date, so summing the per-date,
        per-category prevalence sets counts every job once. Passing *dates*
        restricts both sides to a time window; the denominator then comes from
        `dated_category_counts` (deduped) rather than `daily_stats`.
        """
        window = daily_umbrella_prevalence if dates is None else {
            d: daily_umbrella_prevalence[d] for d in dates if d in daily_umbrella_prevalence
        }
        jobs_with = 0
        for d in window:
            for c in categories:
                jobs_with += len(window[d][c].get(internal_name, set()))
        if dates is None:
            return jobs_with, _track_total(categories)
        total = sum(dated_category_counts[d].get(c, 0) for d in dates for c in categories)
        return jobs_with, total

    track_totals = {tid: _track_total(cats) for tid, _, cats in TRACKS}

    # Generate the market table: one sub-table per track, one column per JOB
    # CATEGORY inside it.
    #
    # This used to be a single track-level table, which was actively misleading:
    # a track is a bag of categories of very different sizes, so the track number
    # mostly reports whichever category is biggest. NOW is 47% Cloud Engineering
    # (526 of 1113), so "Go = 16% of NOW" was really "Go is rare in Cloud
    # Engineering" — and it hid that Go is a genuine Platform/SRE expectation.
    # Category-level cells are the smallest unit that is still a real hiring
    # market, so that is the unit reported.
    recent_dates, prior_dates = _split_windows(dated_category_counts.keys())
    trend_on = bool(recent_dates and prior_dates)

    new_table = []
    new_table.append(
        "> **Skill demand by job category** — each cell is *% of jobs in THAT CATEGORY* "
        "mentioning the tool, with the job count in brackets. Read down a column to see "
        "what one role expects; read across to see how portable a skill is."
    )
    new_table.append(">")
    new_table.append(
        "> Category-level rather than track-level because a track blends categories of very "
        f"different sizes — NOW is {int(round(category_counts.get('Cloud Engineering', 0) / max(track_totals['🎯 NOW'], 1) * 100))}% "
        "Cloud Engineering, so a single NOW number mostly reports Cloud Engineering's skill mix."
    )
    if trend_on:
        new_table.append(">")
        new_table.append(
            f"> **Trend** compares {_window_label(recent_dates)} against {_window_label(prior_dates)} "
            "at track level, two-proportion z-test: 📈/📉 = moved beyond sampling noise "
            "(≥2 pp and significant), ▪️ = flat."
        )
    new_table.append("")

    for tid, label, cats in TRACKS:
        present = [c for c in cats if category_counts.get(c, 0) > 0]
        if not present:
            continue
        new_table.append(f"#### {tid} — {label}  ·  _{track_totals[tid]} jobs_")
        new_table.append("")
        corpus = track_corpus.get(tid, [])
        probe_ok = len(corpus) >= 30
        header_cols = (["Tool"]
                       + [f"{_short_cat(c)} (n={category_counts[c]})" for c in present]
                       + [f"**{tid}** (n={track_totals[tid]})"]
                       + (["**Req%**"] if probe_ok else [])
                       + (["Trend"] if trend_on else [])
                       + ["Decision"])
        new_table.append("| " + " | ".join(header_cols) + " |")
        new_table.append("|" + "|".join(["---"] * len(header_cols)) + "|")

        for display_name, internal_name, decision in core_tools:
            row = [f"**{display_name}**"]
            for c in present:
                jobs_with, total = _track_demand(internal_name, [c])
                row.append(f"{int(round(jobs_with / total * 100))}% ({jobs_with})"
                           if total else "-")
            roll_with, roll_total = _track_demand(internal_name, present)
            roll_pct = int(round(roll_with / roll_total * 100)) if roll_total else None
            row.append(f"**{roll_pct}%**" if roll_pct is not None else "-")
            if probe_ok:
                # Req% = share of the track's JDs where the sentence around the
                # hit reads as a requirement, not an alternative or a bonus.
                # The ⚠️ marks rows whose headline prevalence most overstates
                # how much the skill actually gates an application.
                rx = UMBRELLA_PROBES.get(internal_name)
                if rx is None:
                    row.append("–")
                else:
                    req = sum(1 for t in corpus if skill_modality(t, rx) == "required")
                    req_pct = int(round(req / len(corpus) * 100))
                    flag = (" ⚠️" if roll_pct and req_pct < 0.4 * roll_pct
                            and roll_pct >= 10 else "")
                    row.append(f"**{req_pct}%** ({req}){flag}")
            if trend_on:
                r_w, r_n = _track_demand(internal_name, present, recent_dates)
                p_w, p_n = _track_demand(internal_name, present, prior_dates)
                marker, delta = _trend_marker(r_w, r_n, p_w, p_n)
                row.append(f"{marker} {_pp(delta)}" if delta is not None else "–")
            row.append(decision)
            new_table.append("| " + " | ".join(row) + " |")
        new_table.append("")

    new_table.append(
        "> **Req%** is the column to act on. The percentage columns count every mention "
        "equally, so a skill listed as one of four alternatives scores the same as one "
        "the JD gates on — that is what made Go look like a top-3 blocker when most JDs "
        "would have accepted Python. **Req%** counts only jobs where the sentence around "
        "the mention reads as a requirement (not *\"X or Y or Z\"*, not *Nice to have*). "
        "⚠️ marks rows whose headline prevalence most overstates the real gate."
    )
    new_table.append("")
    new_table.append(
        "_Small categories move a lot between scrapes — treat any column with n < 100 "
        "as directional, not precise. The **Decision** column is hand-maintained, not "
        "derived from these numbers. Req% uses one representative regex per row "
        "(`UMBRELLA_PROBES`), tuned for precision over recall, so read it as a floor._"
    )

    # ── Req% is now a COLUMN in the table above (merged 2026-08-10) ──────────
    # This used to be a separate "🎯 Required vs merely mentioned" section: three
    # more per-track tables over MODALITY_PROBES, ~166 lines. It rendered the
    # same NOW/NEXT/LATER x skill x % shape as the market table directly above
    # it, so the two read as duplicates — and the LESS actionable one (mention
    # counts) came first and larger, which is the wrong way round. Folding Req%
    # in as a column keeps the authoritative number and drops the duplication.
    # See UMBRELLA_PROBES for the precision/recall trade this required.
    new_table.append("")
    new_table.append(
        "_⚠️ = commonly mentioned but rarely required — the headline prevalence "
        "overstates how much this gates an application. Classifier: `skill_modality()`, "
        "hand-validated in `jobs_analytics/scratch/scratch_modality.py`; it reads the "
        "sentence and section header around each mention, so it inherits any "
        "false positives in the probe regex._"
    )

    # ── Skill audit: every canonical skill, with an explicit denominator ──────
    # Guards against the failure that put "Pulumi — 0%", "Vault — 0%" and
    # "Istio — 0%" in the plan's permanent-removal list on 2026-07-26. None of
    # those were zero; they sit at 1-4% and simply never surface in a top-N
    # ranking, so "absent from the table I was looking at" got recorded as
    # "measured at zero". Every skill now gets a row, and every row carries n/N,
    # so a real zero is visually distinct from a missing one.
    new_table.append("")
    new_table.append("<details><summary>🔍 <b>Full skill audit — every canonical skill, with denominators</b> "
                     "(click to expand)</summary>")
    new_table.append("")
    new_table.append(
        "> Answers \"is tool X dead in this market?\" without editing the curated list "
        "above. **A skill absent from a ranked table is not a skill measured at zero** — "
        "check here before writing anything off."
    )
    new_table.append("")
    now_cats = next(cats for tid, _, cats in TRACKS if tid == '🎯 NOW')
    now_n = _track_total(now_cats)
    audit = []
    for skill_name in sorted(TECH_SKILLS_PATTERNS):
        hits = sum(tool_counts[c].get(skill_name.lower(), 0) for c in now_cats)
        audit.append((skill_name, hits))
    audit.sort(key=lambda r: r[1], reverse=True)
    new_table.append(f"| Skill | NOW-track jobs | of n={now_n} |")
    new_table.append("|---|---|---|")
    for skill_name, hits in audit:
        new_table.append(
            f"| {skill_name} | {hits} | {100 * hits / now_n:.1f}% |" if now_n
            else f"| {skill_name} | {hits} | – |")
    new_table.append("")
    new_table.append("</details>")

    table_text = "\n".join(new_table)

    # Read and update Markdown file
    if os.path.exists(MARKDOWN_PATH):
        with open(MARKDOWN_PATH, 'r') as f:
            content = f.read()
            
        pattern = re.compile(r'(<!-- MARKET_DATA_TABLE_START -->\n).*?(\n<!-- MARKET_DATA_TABLE_END -->)', re.DOTALL)
        
        # Build Daily Stats Summary
        daily_summary = []
        for d in sorted(daily_stats.keys(), reverse=True):
            stats = daily_stats[d]
            total = stats.get('Total', 0)
            daily_summary.append(f"### {d} (Total: {total} jobs)")
            
            daily_cats = [c for c, count in stats.items() if c != 'Total' and count > 0]
            daily_cats.sort(key=lambda c: stats[c], reverse=True)
            
            for cat in daily_cats:
                cat_count = stats.get(cat, 0)
                if cat_count > 0:
                    cat_skills = daily_skills[d][cat]
                    top_skills = []
                    for k, v in cat_skills.most_common(10):
                        name = k.capitalize()
                        if name.lower() == 'ci/cd': name = 'CI/CD'
                        elif name.lower() == 'aws': name = 'AWS'
                        elif name.lower() == 'gcp': name = 'GCP'
                        elif name.lower() == 'llms': name = 'LLMs'
                        elif name.lower() == 'vllm': name = 'vLLM'
                        pct = int(round((v / cat_count) * 100))
                        top_skills.append(f"{name} ({pct}%)")
                    
                    skills_str = ", ".join(top_skills) if top_skills else "None extracted"
                    daily_summary.append(f"- **{cat}** ({cat_count} jobs): {skills_str}")
            daily_summary.append("")
            
        daily_table_text = "\n".join(daily_summary).strip()

        # Build Umbrella Breakdown Summary
        # NOTE: Do NOT include START/END markers in the generated text — the regex
        # replacement preserves the file's existing markers via capture groups.
        breakdown_summary = ["### 🔍 Umbrella Sub-Categories Breakdown (All-Time)\n"]
        for display_name, internal_name, _ in core_tools:
            if internal_name.startswith('umbrella_'):
                breakdown_summary.append(f"<details><summary><b>{display_name}</b></summary>\n")
                breakdown_summary.append("<ul>")
                
                sorted_skills = umbrella_breakdown[internal_name].most_common(15)
                if not sorted_skills:
                    breakdown_summary.append("<li>None extracted</li>")
                else:
                    for skill, count in sorted_skills:
                        breakdown_summary.append(f"<li><code>{skill}</code>: {count} mentions</li>")
                breakdown_summary.append("</ul>\n</details>\n")

        # 🔬 What the AI umbrellas actually demand. "LLMs / GenAI" as a single row
        # is unactionable — it hides everything from "use Copilot" to "train a
        # foundation model". This splits each by intent, per track, with a verdict.
        breakdown_summary.append("\n### 🔬 What “LLMs / GenAI” and “AI Agents” Actually Mean in These JDs\n")
        breakdown_summary.append(
            "> The umbrella rows above answer *how often* these come up, not *what is "
            "being asked*. Percentages below are **of the jobs in that track that "
            "mention the umbrella at all** — so read them as: \"when a NOW-track job "
            "says GenAI, this is what it wants.\" The **Your move** column is the "
            "point of the table.\n"
        )
        for umbrella_key, heading, intents in AI_INTENT_SECTIONS:
            per_track = ai_intent_corpus.get(umbrella_key, {})
            counts = {tid: len(per_track.get(tid, [])) for tid, _, _ in TRACKS}
            total_n = sum(counts.values())
            breakdown_summary.append(f"\n#### {heading}\n")
            if total_n < 10:
                breakdown_summary.append(
                    f"*Only {total_n} job(s) mention this so far — too few to break down.*")
                continue
            header = ["What the JD actually wants"] + [
                f"{tid} (n={counts[tid]})" for tid, _, _ in TRACKS] + ["Your move"]
            breakdown_summary.append("| " + " | ".join(header) + " |")
            breakdown_summary.append("|" + "|".join(["---"] * len(header)) + "|")
            rows = []
            for label, (pat, verdict) in intents.items():
                rx = re.compile(pat)
                pcts = []
                for tid, _, _ in TRACKS:
                    texts = per_track.get(tid, [])
                    pcts.append(sum(1 for t in texts if rx.search(t)) / len(texts) * 100
                                if texts else 0.0)
                overall = sum(
                    sum(1 for t in per_track.get(tid, []) if rx.search(t))
                    for tid, _, _ in TRACKS) / total_n * 100
                rows.append((label, pcts, verdict, overall))
            rows.sort(key=lambda r: r[3], reverse=True)
            for label, pcts, verdict, _ in rows:
                cells = [label] + [f"{p:.0f}%" for p in pcts] + [verdict]
                breakdown_summary.append("| " + " | ".join(cells) + " |")

        # 📈 Emerging skills radar — GLiNER hits not yet in the canonical taxonomy.
        # Early-warning for tools trending before they're common enough to be "core".
        breakdown_summary.append("\n### 📈 Emerging / Not-Yet-Canonical Skills\n")
        # GLiNER is the only source of EmergingSkills and it defaults to OFF
        # since 2026-08-09. Say so plainly rather than rendering an empty table,
        # which would read as "nothing is emerging in your market".
        if not any(emerging_counter.values()):
            breakdown_summary.append(
                "> **Not collected.** This radar is populated by GLiNER, which is now "
                "disabled by default: measured over 1,500 jobs it contributed **0 of 6,581** "
                "canonical skill tags that the regex taxonomy did not already find, while "
                "being the dominant per-job cost of the pipeline. Its only unique output was "
                "this table — whose top entry sat at ~2% prevalence and was never "
                "significance-testable, against Kubernetes at 30.4% *required*.\n"
            )
            breakdown_summary.append(
                "> Re-enable with `SEMANTIC_ENABLE_GLINER=1` and re-run the semantic stage.\n"
            )
        breakdown_summary.append(
            "> Surfaced by GLiNER, outside the curated taxonomy. Counted only over jobs "
            f"scoring ≥{EMERGING_RELEVANCE_GATE:.0f} fit, and only when ≥{EMERGING_MIN_COMPANIES} "
            "different employers ask for it — so this is a tool trend in *your* market, not "
            "one company's vocabulary.\n"
        ) if any(emerging_counter.values()) else None

        def _passes_radar(term, count):
            """Keep only plausible, corroborated, not-yet-canonical tool names."""
            if count < 2:
                return False
            # Established / soft-skill / generic-concept suppression. Checked on
            # the separator-normalized form so "infrastructure-as-code" hits the
            # same blocklist entry as "infrastructure as code".
            normed = _norm_term(term)
            if term in _EMERGING_SUPPRESS or normed in _EMERGING_SUPPRESS:
                return False
            # A phrase built entirely from suppressed tokens is suppressed too
            # ("egym wellpass" = two blocklisted benefit vendors glued together).
            parts = normed.split()
            if len(parts) > 1 and all(p in _EMERGING_SUPPRESS for p in parts):
                return False
            if _now_canonical(term) or _looks_generic(term):
                return False
            # Concentration guard: a term carried by one employer is that
            # employer's jargon (or its industry's), not an emerging tool.
            companies = emerging_companies.get(term)
            if not companies or len(companies) < EMERGING_MIN_COMPANIES:
                return False
            top_share = companies.most_common(1)[0][1] / count
            return top_share <= EMERGING_MAX_COMPANY_SHARE

        candidates = [
            (s, c) for s, c in emerging_counter.most_common() if _passes_radar(s, c)
        ]
        gated_total = sum(gated_jobs_by_date.values())
        r_n = sum(gated_jobs_by_date[d] for d in recent_dates)
        p_n = sum(gated_jobs_by_date[d] for d in prior_dates)

        if not candidates:
            breakdown_summary.append("*None surfaced in the current dataset.*")
        elif trend_on and r_n and p_n:
            # Rank by MOMENTUM, not by cumulative count. A cumulative ranking
            # answers "what is rare", which is the wrong question — everything
            # here is rare by construction (see the note above). Sorting on the
            # recent-vs-prior delta answers "what is arriving", which is what a
            # radar is for: a tool at 12 mentions all in the last fortnight
            # matters more than one at 25 spread evenly over two months.
            rows = []
            for skill, count in candidates:
                by_date = emerging_dates[skill]
                r_w = sum(by_date[d] for d in recent_dates)
                p_w = sum(by_date[d] for d in prior_dates)
                # Momentum floor: a term needs a real recent footprint before its
                # growth is worth ranking, or 1-vs-0 noise tops the table.
                if r_w < EMERGING_MIN_RECENT:
                    continue
                # Growth is a RATIO, not a pp delta. These prevalences are ~1%,
                # so pp deltas are all under 1pp — they round to the same value
                # and rank on nothing. Add-one smoothing keeps a zero prior
                # finite and stops tiny numbers from producing huge multiples.
                ratio = ((r_w + 1) / r_n) / ((p_w + 1) / p_n)
                marker = ("🆕" if p_w == 0 else
                          "📈" if ratio >= 1.5 else
                          "📉" if ratio <= 0.67 else "▪️")
                rows.append((r_w, ratio, skill, count, p_w, marker))
            # Order by RECENT footprint, not by the growth ratio. Ranking on the
            # ratio looks right but has no resolution at these counts: a 4-vs-1
            # split and an 11-vs-4 split both read as ~2.6x, so a 4-mention term
            # with 4 employers outranked an 11-mention term with 13. Recent
            # count is the evidence; Growth is the direction, shown alongside.
            rows.sort(key=lambda x: (-x[0], -x[1]))
            breakdown_summary.append(
                f"> Scoped to the **recent window only** — {_window_label(recent_dates)} — so a "
                f"tool that mattered two months ago drops off. **Growth** compares it against "
                f"{_window_label(prior_dates)}. Needs ≥{EMERGING_MIN_RECENT} recent mentions "
                "to appear.\n"
            )
            breakdown_summary.append(
                "> ⚠️ **Directional, not significance-tested.** At ~1% prevalence over "
                f"{r_n} recent jobs, no single term can clear a z-test — read this as "
                "*where to look*, not proof a tool is taking off. 🆕 = absent from the "
                "prior window entirely.\n"
            )
            breakdown_summary.append(
                "| Skill | Recent | Prior | Growth | | Total | Employers |")
            breakdown_summary.append(
                "|-------|--------|-------|--------|---|-------|-----------|")
            for r_w, ratio, skill, count, p_w, marker in rows[:20]:
                breakdown_summary.append(
                    f"| `{skill}` | {r_w/r_n*100:.1f}% ({r_w}) | {p_w/p_n*100:.1f}% ({p_w}) "
                    f"| **{ratio:.1f}×** | {marker} | {count} | {len(emerging_companies[skill])} |"
                )
        else:
            breakdown_summary.append(
                f"> Not enough scrape history yet for a trend split — showing cumulative "
                f"counts over {gated_total} on-target jobs.\n"
            )
            breakdown_summary.append("| Skill | Mentions | % of on-target jobs | Employers |")
            breakdown_summary.append("|-------|----------|---------------------|-----------|")
            for skill, count in candidates[:20]:
                share = f"{count / gated_total * 100:.1f}%" if gated_total else "-"
                breakdown_summary.append(
                    f"| `{skill}` | {count} | {share} | {len(emerging_companies[skill])} |"
                )
        breakdown_text = "\n".join(breakdown_summary)

        # Build "Next Best Action" ROI Table
        YOUR_SKILLS = {
            "umbrella_k8s": 0.9,
            "umbrella_cicd": 0.85,
            "umbrella_iac": 0.8,
            "umbrella_cloud": 0.6,
            "umbrella_python": 0.7,
            "umbrella_observability": 0.5,
            "umbrella_databases": 0.4,
            "umbrella_go": 0.15,
            "umbrella_ai_assistants": 0.6,
            "umbrella_agents": 0.1,
            "umbrella_llms": 0.2,
            "umbrella_rag": 0.1,
            "umbrella_gpu": 0.05,
            "umbrella_streaming": 0.3,
        }
        
        level_labels = {0.0: "None", 0.05: "Awareness", 0.1: "Beginner", 0.15: "Beginner",
                       0.2: "Basic", 0.3: "Basic", 0.4: "Intermediate", 0.5: "Intermediate",
                       0.6: "Competent", 0.7: "Competent", 0.8: "Advanced", 0.85: "Advanced",
                       0.9: "Expert", 1.0: "Expert"}

        # Per-track ROI: demand is measured WITHIN each track, so the ranking tells
        # you what to learn for THAT track specifically (Core vs AI-pivot vs Staff).
        roi_lines = ["### 🎯 Next Best Actions (Data-Driven ROI, per Track)\n"]
        roi_lines.append(
            "> ROI Score = Track Demand % × (1 − Your Proficiency). Ranked separately "
            "per track so you can prep the Core track for interviews while pursuing the "
            "growth track deliberately.\n"
        )
        for tid, label, cats in TRACKS:
            track_total = track_totals[tid]
            roi_lines.append(f"\n#### {tid} — {label}  ·  _{track_total} jobs_\n")
            if track_total == 0:
                roi_lines.append("*No jobs in this track yet.*")
                continue
            roi_data = []
            for display_name, internal_name, _ in core_tools:
                jobs_with, _ = _track_demand(internal_name, cats)
                demand_pct = jobs_with / track_total * 100
                proficiency = YOUR_SKILLS.get(internal_name, 0.0)
                roi = demand_pct * (1.0 - proficiency)
                level = level_labels.get(proficiency, f"{int(proficiency*100)}%")
                roi_data.append((display_name, demand_pct, level, roi))
            roi_data.sort(key=lambda x: x[3], reverse=True)
            roi_lines.append("| Rank | Skill | Track Demand | Your Level | ROI Score |")
            roi_lines.append("|------|-------|--------------|------------|-----------|")
            for rank, (name, demand, level, roi) in enumerate(roi_data[:8], 1):
                roi_lines.append(f"| {rank} | **{name}** | {demand:.0f}% | {level} | {roi:.1f} |")

        # ── Senior → Staff gap (NOW/Core track, IC ladder) ───────────────────────
        # The delta column is the crux: what Staff JDs demand MORE than the Senior
        # roles currently landing interviews. Positive = under-prepared for Staff.
        roi_lines.append("\n### 🪜 Senior → Staff Gap · tools (all target tracks, IC ladder)\n")
        sen_n = core_tier_totals.get('Senior', 0)
        stf_n = core_tier_totals.get('Staff+', 0)
        if sen_n < 15 or stf_n < 15:
            roi_lines.append(
                f"> *Not enough explicitly-tiered IC data yet (Senior={sen_n}, "
                f"Staff+={stf_n}; need ≥15 each). Tier comes from the title — "
                "staff/principal/distinguished/fellow/tech-lead vs senior/expert — "
                "and untitled roles are excluded rather than assumed Senior.*"
            )
        else:
            roi_lines.append(
                f"> IC roles across NOW/NEXT/LATER whose title states a tier: "
                f"Senior (n={sen_n}) vs Staff+ (n={stf_n}). **Δ = Staff% − Senior%**. "
                "Scoped to all three tracks rather than Core alone so the sample is "
                "large enough for ✅/🔻 to mean *\"a real gap\"* rather than ▪️ "
                "*\"too few Staff roles to tell\"*.\n>\n"
                "> **Nothing is significantly higher at Staff — several tools are "
                "significantly LOWER.** Staff JDs spend their word count on scope "
                "and impact instead of listing tools, so do not read a falling bar "
                "as \"Staff needs less Terraform\". It means the tool is assumed. "
                "The actual step up is in the next table.\n"
            )
            gap_rows = []
            for display_name, internal_name, _ in core_tools:
                s_hits = len(core_tier_prev['Senior'].get(internal_name, set()))
                t_hits = len(core_tier_prev['Staff+'].get(internal_name, set()))
                sp, tp = s_hits / sen_n * 100, t_hits / stf_n * 100
                z = _two_proportion_z(tp / 100, stf_n, sp / 100, sen_n)
                gap_rows.append((display_name, sp, tp, tp - sp, abs(z) >= 1.96))
            gap_rows.sort(key=lambda x: x[3], reverse=True)
            roi_lines.append("| Skill | Senior demand | Staff+ demand | Δ (Staff − Senior) | |")
            roi_lines.append("|-------|---------------|---------------|--------------------|---|")
            for name, sp, tp, d, sig in gap_rows:
                mark = ("✅" if d > 0 else "🔻") if sig else "▪️"
                roi_lines.append(f"| **{name}** | {sp:.0f}% | {tp:.0f}% | **{_pp(d)}** | {mark} |")
            roi_lines.append(
                "\n_✅/🔻 = the gap survives a two-proportion z-test at 95%; "
                "▪️ = within noise for this sample size._"
            )

        # ── Staff-bar readiness: org-scope signals, measured as LIFT ──────────────
        # The question is not "what do Staff JDs mention" (boilerplate makes almost
        # everything look required) but "what do they demand that Senior JDs do NOT".
        roi_lines.append("\n### 🏗️ Staff-Bar Signal · org-scope (Staff+ IC vs Senior IC)\n")
        if len(staff_ic_jobs) < 15 or len(senior_ic_jobs) < 15:
            roi_lines.append(
                f"*Not enough tiered IC roles on the tool tracks yet "
                f"(Staff+={len(staff_ic_jobs)}, Senior={len(senior_ic_jobs)}).*"
            )
        else:
            roi_lines.append(
                f"> {len(staff_ic_jobs)} Staff+ IC JD(s) vs {len(senior_ic_jobs)} "
                "Senior IC JD(s) on the same tracks. **The Δ column is the real bar** "
                "— a signal at 50% in Staff JDs means nothing if it is also at 50% in "
                "Senior JDs. Build projects and interview stories against the ✅ rows; "
                "the ▪️ rows are table stakes you already clear.\n"
            )
            staff_rows = _lift_rows(IMPACT_THEMES, staff_ic_jobs, senior_ic_jobs)
            _render_lift_table(roi_lines, staff_rows, "Staff+ IC", "Senior IC")
            roi_lines.append(
                "\n_✅ = genuinely differentiates Staff from Senior (≥5 pp and "
                "significant at 95%); ▪️ = demanded equally at both tiers; "
                "🔻 = more of a Senior-role expectation._"
            )
            # Turn the table into an instruction. The two tables above only pay
            # off if they change what goes ON THE CV, and the natural instinct
            # (list more tools) is precisely what the data says not to do.
            winners = [r for r in staff_rows if r[4] and r[3] >= 5][:3]
            if winners:
                roi_lines.append("\n#### ✍️ What this means for your CV\n")
                roi_lines.append(
                    "> **Write bullets about scope, not tools.** The tools table "
                    "above shows nothing is demanded *more* at Staff — several "
                    "things are demanded *less*, because they are assumed. The "
                    "step up is entirely in these signals:\n"
                )
                for theme, a, b, d, _ in winners:
                    roi_lines.append(
                        f"> - **{theme}** — {a:.0f}% of Staff JDs vs {b:.0f}% of "
                        f"Senior ({_pp(d)})"
                    )
                roi_lines.append(
                    "\n> Rewrite each CV bullet so the *blast radius* is the "
                    "subject. Not \"used Terraform and ArgoCD to build an IDP\" — "
                    "every Senior says that. Instead: \"set the deployment "
                    "standard adopted by N teams, cutting provisioning toil X%\". "
                    "Same project, Staff framing: **who else changed because of "
                    "you.** Your Aldi Süd platform-SME work and the Rakuten "
                    "400-engineer platform are already Staff-scope stories — they "
                    "are just currently written as tool lists."
                )

        # ── Manager track (org-wide impact / people signals) ──────────────────────
        roi_lines.append("\n### 🧭 Manager Track Signal (optional — EM/Head-of, vs IC baseline)\n")
        if len(manager_jobs) < 15:
            roi_lines.append(
                f"*Not enough engineering-management roles surfaced yet (n={len(manager_jobs)}).*")
        else:
            roi_lines.append(
                f"> {len(manager_jobs)} engineering manager/lead role(s) vs "
                f"{len(all_ic_jobs)} IC role(s) as the baseline. These JDs list "
                "responsibilities, not tools, so this is a themes table. Non-software "
                "leadership (construction, product, sales) is excluded from both sides.\n"
            )
            _render_lift_table(
                roi_lines, _lift_rows(MANAGEMENT_THEMES, manager_jobs, all_ic_jobs),
                "Manager", "IC baseline")
            roi_lines.append(
                "\n**Where the two ladders overlap** — org-scope signals shared with "
                "the Staff IC bar, so a story you build for one counts for both:\n"
            )
            _render_lift_table(
                roi_lines, _lift_rows(IMPACT_THEMES, manager_jobs, all_ic_jobs),
                "Manager", "IC baseline")
        roi_text = "\n".join(roi_lines)

        # Build salary view per track (does the growth track actually pay more?).
        cat_to_track = {c: tid for tid, _, cats in TRACKS for c in cats}
        salary_lines = ["### 💰 Salary by Track & High-Pay Correlates\n"]

        if not salary_jobs:
            salary_lines.append("> *No salary data found in the recent scrape.*")
        else:
            # Per-track median-ish band (min of SalaryMin, max of SalaryMax, avg).
            salary_lines.append("| Track | Jobs w/ salary | Avg min | Avg max | Range |")
            salary_lines.append("|-------|---------------|---------|---------|-------|")
            for tid, label, _ in TRACKS:
                tjobs = [j for j in salary_jobs if cat_to_track.get(j['category']) == tid]
                if not tjobs:
                    salary_lines.append(f"| **{tid}** | 0 | – | – | – |")
                    continue
                mins = [j['salary_min'] for j in tjobs if j.get('salary_min')]
                maxs = [j['salary_max'] for j in tjobs if j.get('salary_max')] or mins
                avg_min = sum(mins) / len(mins) if mins else 0
                avg_max = sum(maxs) / len(maxs) if maxs else 0
                salary_lines.append(
                    f"| **{tid}** | {len(tjobs)} | €{avg_min/1000:.0f}k | €{avg_max/1000:.0f}k "
                    f"| €{min(mins)/1000:.0f}k–€{max(maxs)/1000:.0f}k |"
                )

            # High-pay skill correlates. The cut is the top TERCILE of the observed
            # salaried jobs, not a hardcoded €95k: a fixed threshold silently drifts
            # as the market moves and gave an unstable n. Reported as LIFT against
            # the bottom two terciles — "% of high-paying jobs mentioning cloud" is
            # meaningless when cloud is in most jobs at every pay level.
            def _umbrellas_of(job):
                """Set of umbrella display names this job's skills hit (once each)."""
                job_skills = set(job['skills'])
                hits = set()
                if any('agent' in s for s in job_skills):
                    hits.add("AI Agents / Agentic")
                for u_key, u_keywords in UMBRELLA_DEFINITIONS.items():
                    if u_key == 'umbrella_agents':
                        continue
                    if job_skills & u_keywords:
                        disp = next((d for d, i, _ in core_tools if i == u_key), None)
                        if disp:
                            hits.add(disp)
                return hits

            ranked = sorted(salary_jobs, key=lambda j: j.get('salary_min') or 0)
            cut_idx = int(len(ranked) * 2 / 3)
            high_salary_jobs, rest = ranked[cut_idx:], ranked[:cut_idx]
            cut_value = (high_salary_jobs[0].get('salary_min') or 0) if high_salary_jobs else 0
            salary_lines.append(
                f"\n**High-pay correlates** — top third of salaried postings "
                f"(≥ €{cut_value/1000:.0f}k, n={len(high_salary_jobs)}) vs the rest "
                f"(n={len(rest)}):\n"
            )
            if len(high_salary_jobs) < 10 or len(rest) < 10:
                salary_lines.append(
                    "*Too few salaried postings to split into pay bands yet — German "
                    "listings rarely state salary, so this fills in slowly.*")
            else:
                hi_c, lo_c = Counter(), Counter()
                for j in high_salary_jobs:
                    hi_c.update(_umbrellas_of(j))
                for j in rest:
                    lo_c.update(_umbrellas_of(j))
                rows = []
                for disp, _, _ in core_tools:
                    if disp not in hi_c and disp not in lo_c:
                        continue
                    p1 = hi_c[disp] / len(high_salary_jobs)
                    p2 = lo_c[disp] / len(rest)
                    z = _two_proportion_z(p1, len(high_salary_jobs), p2, len(rest))
                    rows.append((disp, p1 * 100, p2 * 100, (p1 - p2) * 100, abs(z) >= 1.96))
                rows.sort(key=lambda r: r[3], reverse=True)
                salary_lines.append("| Skill Umbrella | Top third | Rest | Δ | |")
                salary_lines.append("|----------------|-----------|------|---|---|")
                for disp, a, b, d, sig in rows[:12]:
                    mark = ("✅" if d > 0 else "🔻") if sig else "▪️"
                    salary_lines.append(
                        f"| **{disp}** | {a:.0f}% | {b:.0f}% | **{_pp(d)}** | {mark} |")
                salary_lines.append(
                    "\n_✅ = the tool really does skew toward better-paid postings; "
                    "▪️ = equally common at every pay level (so it is a baseline "
                    "expectation, not a raise)._"
                )

        salary_text = "\n".join(salary_lines)

        if pattern.search(content):
            new_content = pattern.sub(rf'\g<1>{table_text}\g<2>', content)
            
            # Inject Umbrella Breakdown
            umbrella_pattern = re.compile(r'(<!-- UMBRELLA_BREAKDOWN_START -->\n).*?(\n<!-- UMBRELLA_BREAKDOWN_END -->\n)', re.DOTALL)
            if umbrella_pattern.search(new_content):
                new_content = umbrella_pattern.sub(rf'\g<1>{breakdown_text}\g<2>', new_content)
            else:
                inject_pattern = re.compile(r'(<!-- MARKET_DATA_TABLE_END -->\n)')
                new_content = inject_pattern.sub(rf'\g<1>{breakdown_text}', new_content)

            # Inject ROI Table
            roi_pattern = re.compile(r'(<!-- ROI_TABLE_START -->\n).*?(\n<!-- ROI_TABLE_END -->\n)', re.DOTALL)
            if roi_pattern.search(new_content):
                new_content = roi_pattern.sub(rf'\g<1>{roi_text}\g<2>', new_content)
            else:
                inject_roi = re.compile(r'(<!-- UMBRELLA_BREAKDOWN_END -->\n)')
                new_content = inject_roi.sub(rf'\g<1>\n{roi_text}', new_content)

            # Inject Salary Correlation
            salary_pattern = re.compile(r'(<!-- SALARY_CORRELATION_START -->\n).*?(\n<!-- SALARY_CORRELATION_END -->\n)', re.DOTALL)
            if salary_pattern.search(new_content):
                new_content = salary_pattern.sub(rf'\g<1>{salary_text}\g<2>', new_content)
            else:
                inject_salary = re.compile(r'(<!-- ROI_TABLE_END -->\n)')
                new_content = inject_salary.sub(rf'\g<1>\n{salary_text}', new_content)

            daily_pattern = re.compile(r'(<!-- DAILY_STATS_TABLE_START -->\n).*?(\n<!-- DAILY_STATS_TABLE_END -->)', re.DOTALL)
            if daily_pattern.search(new_content):
                new_content = daily_pattern.sub(rf'\g<1>{daily_table_text}\g<2>', new_content)
            else:
                print("⚠️ Could not find <!-- DAILY_STATS_TABLE_START --> markers in the markdown file.")

            with open(MARKDOWN_PATH, 'w') as f:
                f.write(new_content)
            print(f"✅ Successfully updated the Markdown tables in {MARKDOWN_PATH}")
        else:
            print("⚠️ Could not find <!-- MARKET_DATA_TABLE_START --> markers in the markdown file.")
            print("Here is the generated table instead:\n")
            print(table_text)
    else:
        print(f"⚠️ Could not find Markdown file at {MARKDOWN_PATH}")
        print("Here is the generated table instead:\n")
        print(table_text)

if __name__ == "__main__":
    main()
