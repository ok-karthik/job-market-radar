"""User-specific configuration, loaded from a gitignored ``config.json``.

WHY THIS EXISTS
---------------
Everything personal used to be hard-coded in source: the CV's Google Docs URL,
a Notion page id, the learning-plan filename, and the search keywords and geos.
That makes the repository unshareable — the CV link alone is a private document,
and the search parameters plus the plan reveal personal circumstances that have
no business in version control.

Resolution order for every value:

    config.json  →  environment variable  →  the default below

``config.json`` is gitignored. ``config.example.json`` is committed and carries
placeholders, so a fresh clone runs immediately with generic defaults and simply
isn't pointed at anyone.

DESIGN NOTES
------------
* **Defaults are generic, not the user.** A missing config must not silently
  fall back to someone else's CV URL or Notion page — that is the failure mode
  this module exists to prevent. The CV default is empty and the code errors
  clearly if it is needed and unset.
* **JSON, not YAML**, to avoid adding a dependency to a pipeline whose whole
  point is that it runs locally with no services.
* **Read once at import.** These are process-lifetime settings; reloading
  mid-run would make a scrape non-reproducible.
"""
import json
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.environ.get("JOBS_CONFIG", os.path.join(_REPO_ROOT, "config.json"))
EXAMPLE_PATH = os.path.join(_REPO_ROOT, "config.example.json")


def _load():
    for path in (CONFIG_PATH, EXAMPLE_PATH):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f), path
            except (OSError, json.JSONDecodeError) as e:
                print(f"[!] Could not read {path}: {e} — falling back to defaults.")
    return {}, None


_CFG, CONFIG_SOURCE = _load()
USING_EXAMPLE = CONFIG_SOURCE == EXAMPLE_PATH


def get(path, default=None):
    """Fetch a dotted key, e.g. get('search.time_range'). Env override wins.

    The env var name is the dotted path upper-cased with dots as underscores and
    a ``JOBS_`` prefix — ``search.time_range`` → ``JOBS_SEARCH_TIME_RANGE``.
    """
    env = "JOBS_" + path.upper().replace(".", "_")
    if env in os.environ:
        raw = os.environ[env]
        if isinstance(default, bool):
            return raw not in ("0", "", "false", "False")
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                return int(raw)
            except ValueError:
                pass
        return raw
    node = _CFG
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ── profile ──────────────────────────────────────────────────────────────────
# Empty by default ON PURPOSE: a shared clone must not silently score against
# somebody else's CV. semantic_job_analyzer raises a clear error if it needs
# this and it is unset.
CV_URL = get("profile.cv_url", "")
CV_LOCAL_PATH = get("profile.cv_local_path", "CV.pdf")
PLAN_MARKDOWN = get("profile.plan_markdown", "Learning_Plan.md")
NOTION_PAGE_ID = get("profile.notion_page_id", "")

# ── search ───────────────────────────────────────────────────────────────────
TIME_RANGE = get("search.time_range", "r86400")
MAX_RESULTS_PER_PROFILE = get("search.max_results_per_profile", 150)
HYDRATION_CONCURRENCY = get("search.hydration_concurrency", 2)
KEYWORD_GROUPS = get("search.keyword_groups", {})
GEOS = get("search.geos", [])
# {pool: [geo name, ...]} — geos this pool should NOT search. Optional.
SKIP_GEOS = get("search.skip_geos", {})

# ── targeting ────────────────────────────────────────────────────────────────
NOW_CATEGORIES = get("targeting.now_categories", [
    "Platform Engineering", "Site Reliability Engineering (SRE)",
    "DevOps Engineering", "Cloud Engineering"])
NEXT_CATEGORIES = get("targeting.next_categories",
                      ["AI Infrastructure", "MLOps", "AI Solutions Architecture"])
LATER_CATEGORIES = get("targeting.later_categories",
                       ["Staff / Principal Engineering", "Solutions Architecture"])
YOUR_SKILLS = get("targeting.your_skills", {})
SALARY_FLOOR_EUR = get("targeting.salary_floor_eur", 0)

# ── shortlist ────────────────────────────────────────────────────────────────
SHORTLIST_THRESHOLD = get("shortlist.score_threshold", 70)
SHORTLIST_CAP = get("shortlist.cap", 60)
SHORTLIST_DAYS = get("shortlist.days", 7)


def build_search_urls():
    """Render (url, pool, geo) tuples from the configured keywords and geos.

    Keywords are joined with ``+OR+`` inside a URL-encoded group and left
    UNQUOTED — quoting drops compound titles, and the guest API breaks on NOT.
    Each geo is a separate URL because comma-encoded multi-geoId params are
    silently ignored and fall back to a global result set.
    """
    # The SEARCH-PAGE base, not the guest API endpoint. extract_job_ids() parses
    # the geoId and keywords back out of these and rebuilds the guest-API request
    # itself, injecting TIME_RANGE and pagination. Emitting the API URL here
    # produced query-params that matched while the base silently did not.
    base = "https://www.linkedin.com/jobs/search/"
    out = []
    for pool, words in KEYWORD_GROUPS.items():
        kw = "%28" + "+OR+".join(w.replace(" ", "+") for w in words) + "%29"
        # Per-pool geo suppression. A pool whose stream is fully claimed by an
        # earlier pool contributes ~nothing, and every extra pool widens the
        # soft-block window for the WHOLE run. Measured over 10 runs on
        # 2026-08-12 (see AGENTS.md, "Group C review"): the three Munich pools
        # produced 2 unique jobs total, and C-cloud's five CITY pools produced 4
        # (max FitScore 39.3) while its Germany pool carried every high scorer,
        # including the corpus's single best match at 100.0.
        #
        # NOTE: suppressing a geo POOL does not suppress that geo's JOBS —
        # Munich-located roles are ~20% of every run and arrive via other pools.
        # `sourceGeo` is the pool's geo, never the job's location.
        skip = {g.lower() for g in SKIP_GEOS.get(pool, [])}
        for geo in GEOS:
            if geo["name"].lower() in skip:
                continue
            # Parameter ORDER matches the hand-written list so a diff of the
            # rendered URLs is empty, not just equivalent.
            parts = ["f_WT=2"] if geo.get("remote_only") else []
            parts += [f"f_JT=F", f"geoId={geo['geo_id']}", f"keywords={kw}"]
            out.append((f"{base}?{'&'.join(parts)}", pool, geo["name"]))
    return out


def target_profiles():
    """Just the URLs, in the shape apify_replica.TARGET_PROFILES expects."""
    return [u for u, _pool, _geo in build_search_urls()]


def warn_if_unconfigured():
    """Print a clear notice when running on the committed example config."""
    if USING_EXAMPLE:
        print("[!] No config.json found — using config.example.json. "
              "Copy it to config.json and set your CV URL, Notion page and "
              "search parameters.")
    elif CONFIG_SOURCE is None:
        print("[!] No config file found at all — running on built-in defaults.")
