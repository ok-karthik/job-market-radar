import pytest
from scraper.semantic_job_analyzer import (
    SemanticJobAnalyzer,
    ESTABLISHED_NON_NOVEL,
    SOFT_SKILL_BLOCKLIST,
    _download_cv,
)


def test_emerging_radar_suppression_sets():
    """The emerging-skills radar must not surface established/ubiquitous tech or
    generic architecture concepts. These sets are the source of truth (re-used at
    render time by jobs_analytics/update_learning_plan.py), so lock their contents."""
    # Ubiquitous / out-of-scope tools => never 'emerging'
    for tool in ["git", "github", "react", "node.js", "sql", "sap", "tableau", "google"]:
        assert tool in ESTABLISHED_NON_NOVEL, f"{tool!r} should be suppressed from the radar"
    # Generic architecture/practice concepts => not tools at all
    for concept in ["observability", "distributed systems", "microservices",
                    "ai tools", "data quality", "system architecture", "ci"]:
        assert concept in SOFT_SKILL_BLOCKLIST, f"{concept!r} should be blocklisted"
    # Generic concepts/practices/domains + NER junk added 2026-07-27 after they
    # leaked into the emerging radar. The tools that implement them are canonical;
    # the practice names themselves are noise.
    for concept in ["monitoring", "cloud infrastructure", "infrastructure as code",
                    "secrets management", "performance optimization", "api design",
                    "cybersecurity", "computer vision", "continuous improvement",
                    "event-driven architectures", "data platform", "efficiency",
                    "mathematics", "erp", "sdks", "routing", "c1", "b2",
                    # second-tier tail noise
                    "agile", "algorithms", "deep learning", "engineering manager",
                    "slos", "testing", "design patterns", "iot", "guardrails"]:
        assert concept in SOFT_SKILL_BLOCKLIST, f"{concept!r} should be blocklisted"
    # Off-track products belong in the established set, not the radar
    for tool in ["nestjs", "android", "facebook"]:
        assert tool in ESTABLISHED_NON_NOVEL, f"{tool!r} should be established/off-track"
    # The two sets must stay disjoint in *purpose* but overlap is harmless; ensure
    # a genuinely novel tool (n8n) is in NEITHER so it can still surface.
    assert "n8n" not in ESTABLISHED_NON_NOVEL
    assert "n8n" not in SOFT_SKILL_BLOCKLIST


def test_promoted_terms_are_canonical_not_emerging(analyzer):
    """`redshift` and the plural `vector databases` were leaking to the emerging
    radar; they must now fold onto canonical taxonomy tools instead."""
    canonical, emerging = analyzer.extract_tech_skills(
        "Experience with Redshift and vector databases is required.",
        with_emerging=True,
    )
    assert "Redshift" in canonical
    assert "Vector DBs" in canonical
    assert "redshift" not in [e.lower() for e in emerging]
    assert "vector databases" not in [e.lower() for e in emerging]

    # Other promoted tools (Athena/DuckDB/ClickHouse/Okta/Kubeflow/Opsgenie)
    text = ("We run Athena, DuckDB, ClickHouse, Okta for SSO, Kubeflow pipelines "
            "and Opsgenie for on-call.")
    canon2 = analyzer.extract_tech_skills(text)
    assert "AWS" in canon2                 # Athena folds into the AWS umbrella
    assert "DuckDB" in canon2
    assert "ClickHouse" in canon2
    assert "Okta" in canon2
    assert "Kubeflow" in canon2
    assert "Opsgenie" in canon2

# Use a dummy CV path for initialization (the file might not exist during some CI runs, 
# but we can mock or just pass a string since the init checks os.path.exists and prints a warning)
# Session-scoped: loading BGE + GLiNER is the expensive part, and every method
# exercised here (categorize_job / extract_salary / extract_tech_skills) is
# read-only on the analyzer, so one shared instance is safe and keeps the suite
# from reloading both models for every single test.
@pytest.fixture(scope="session")
def analyzer(tmp_path_factory):
    dummy_cv = tmp_path_factory.mktemp("cv") / "dummy_cv.txt"
    dummy_cv.write_text("dummy cv text")
    return SemanticJobAnalyzer(cv_path=str(dummy_cv))

def test_extract_tech_skills(analyzer):
    text = "We are looking for a Python developer with experience in AWS, Kubernetes and Terraform. Golang is a plus."
    skills = analyzer.extract_tech_skills(text)
    
    assert "Python" in skills
    assert "AWS" in skills
    assert "Kubernetes" in skills
    assert "Terraform" in skills
    assert "Go" in skills # "Golang" should map to "Go"
    
    # Test "Go" specifically
    text_go = "Experience with Go and GCP."
    skills_go = analyzer.extract_tech_skills(text_go)
    assert "Go" in skills_go
    assert "GCP" in skills_go
    
    # Test boundary issues
    text2 = "Good interpersonal skills. We use AWS."
    skills2 = analyzer.extract_tech_skills(text2)
    assert "Go" not in skills2 # "Good" should not match "Go"
    assert "AWS" in skills2

    # Case-sensitivity: English prose 'go'/'rest' must NOT count as the tools
    prose = "We move fast and go live weekly; on the go. The rest of the team uses AWS."
    prose_skills = analyzer.extract_tech_skills(prose)
    assert "Go" not in prose_skills
    assert "REST" not in prose_skills
    assert "AWS" in prose_skills

    # ...but real (even lowercase) mentions are still captured
    assert "Go" in analyzer.extract_tech_skills("Backend written in golang and Python.")
    assert "REST" in analyzer.extract_tech_skills("Design RESTful APIs and gRPC services.")

    # with_emerging returns a (canonical, emerging) tuple; canonical stays clean
    canonical, emerging = analyzer.extract_tech_skills(
        "Python and Kubernetes experience required.", with_emerging=True
    )
    assert "Python" in canonical and "Kubernetes" in canonical
    assert isinstance(emerging, list)

def test_categorize_job_platform(analyzer):
    title = "Platform Engineer"
    desc = "You will build the internal developer platform using Kubernetes, ArgoCD, and AWS."
    
    category = analyzer.categorize_job(title, desc)
    assert category == "Platform Engineering"

def test_categorize_job_ai_infra(analyzer):
    title = "AI Infrastructure Engineer"
    desc = "Deploy LLMs and build RAG pipelines on GPU clusters."

    category = analyzer.categorize_job(title, desc)
    assert category == "AI Infrastructure"


# --- _download_cv resilience (regression for the 2026-07-27 nightly crash) ---
# A transient httpx.ReadTimeout on the Google Docs CV export took down the whole
# pipeline because the download had no retry and no fallback to the cached CV.

def test_download_cv_retries_then_succeeds(mocker, tmp_path):
    """A transient network error is retried, and a later success writes the CV."""
    import httpx
    local = tmp_path / "CV.pdf"
    ok = mocker.Mock()
    ok.content = b"%PDF-1.4 fake"
    ok.raise_for_status = mocker.Mock()
    client = mocker.MagicMock()
    client.get.side_effect = [httpx.ReadTimeout("boom"), ok]
    mocker.patch(
        "scraper.semantic_job_analyzer.httpx.Client",
    ).return_value.__enter__.return_value = client
    sleep = mocker.patch("scraper.semantic_job_analyzer.time.sleep")

    result = _download_cv(url="http://example/cv", local_path=str(local),
                          max_attempts=4)

    assert result == str(local)
    assert local.read_bytes() == b"%PDF-1.4 fake"
    assert client.get.call_count == 2
    sleep.assert_called_once()  # one backoff between the two attempts


def test_download_cv_falls_back_to_cached_copy(mocker, tmp_path):
    """When every attempt fails but a cached CV exists, use it instead of raising."""
    import httpx
    local = tmp_path / "CV.pdf"
    local.write_bytes(b"stale cached cv")  # simulate a prior successful run
    client = mocker.MagicMock()
    client.get.side_effect = httpx.ReadTimeout("boom")
    mocker.patch(
        "scraper.semantic_job_analyzer.httpx.Client",
    ).return_value.__enter__.return_value = client
    mocker.patch("scraper.semantic_job_analyzer.time.sleep")

    result = _download_cv(url="http://example/cv", local_path=str(local),
                          max_attempts=3)

    assert result == str(local)
    assert local.read_bytes() == b"stale cached cv"  # cached copy untouched
    assert client.get.call_count == 3


def test_download_cv_raises_when_no_cache(mocker, tmp_path):
    """With no cached CV to fall back to, an exhausted download must raise."""
    import httpx
    local = tmp_path / "CV.pdf"  # never created
    client = mocker.MagicMock()
    client.get.side_effect = httpx.ReadTimeout("boom")
    mocker.patch(
        "scraper.semantic_job_analyzer.httpx.Client",
    ).return_value.__enter__.return_value = client
    mocker.patch("scraper.semantic_job_analyzer.time.sleep")

    with pytest.raises(RuntimeError, match="no cached"):
        _download_cv(url="http://example/cv", local_path=str(local),
                     max_attempts=2)


# ---------------------------------------------------------------------------
# Score normalization (pure static helper — no model needed)
# ---------------------------------------------------------------------------

def test_normalize_scores_spreads_to_0_100():
    out = SemanticJobAnalyzer._normalize_scores([0.5, 0.6, 0.8])
    assert out[0] == 0.0 and out[-1] == 100.0     # min→0, max→100
    assert 0 < out[1] < 100                        # middle preserved in order
    assert out == sorted(out)                      # order preserved


def test_normalize_scores_all_equal_collapses_to_50():
    # No spread → everything 50 (avoids divide-by-zero, signals "no ranking")
    assert SemanticJobAnalyzer._normalize_scores([0.7, 0.7, 0.7]) == [50.0, 50.0, 50.0]
    assert SemanticJobAnalyzer._normalize_scores([0.42]) == [50.0]   # single job


def test_normalize_scores_empty():
    assert SemanticJobAnalyzer._normalize_scores([]) == []


# ---------------------------------------------------------------------------
# Salary extraction (regex, EUR-denominated, 30k–500k sanity clamp)
# ---------------------------------------------------------------------------

def test_extract_salary_full_range(analyzer):
    s = analyzer.extract_salary("Compensation is €80,000 - €100,000 per year.")
    assert s["min"] == 80000 and s["max"] == 100000


def test_extract_salary_k_format(analyzer):
    s = analyzer.extract_salary("We offer €90k - €120k depending on experience.")
    assert s["min"] == 90000 and s["max"] == 120000


def test_extract_salary_rejects_out_of_range(analyzer):
    # Below the 30k sanity floor → treated as not-a-salary
    assert analyzer.extract_salary("Stipend of €10,000 - €20,000.") == {}


def test_extract_salary_none_when_absent(analyzer):
    assert analyzer.extract_salary("Great team, fully remote, no numbers here.") == {}


# ---------------------------------------------------------------------------
# Categorization — deterministic title overrides (bypass the embedding model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Senior Backend Engineer", "Backend Engineering"),
    ("DevOps Engineer (m/f/d)", "DevOps Engineering"),
    ("SRE Engineer", "Site Reliability Engineering (SRE)"),
    ("Staff Engineer, Infrastructure", "Staff / Principal Engineering"),
    ("Engineering Manager", "Engineering Leadership"),
    ("Cloud Engineer", "Cloud Engineering"),
])
def test_categorize_title_overrides(analyzer, title, expected):
    # These paths early-return before the embedding call, so they're deterministic.
    assert analyzer.categorize_job(title, "irrelevant description") == expected


def test_categorize_product_title_not_overridden_to_engineering(analyzer):
    # A real product role must NOT hit the engineering fast-paths.
    assert analyzer.categorize_job("Product Manager", "roadmap and stakeholders") \
        not in {"Backend Engineering", "DevOps Engineering", "Platform Engineering"}


# ---------------------------------------------------------------------------
# Per-job score cache — a recurring job must not be re-scored on the next run
# ---------------------------------------------------------------------------

def test_process_jobs_score_cache_reuses(analyzer, tmp_path, mocker):
    cache = str(tmp_path / "score_cache.json")
    job = {"id": "job-1", "title": "Platform Engineer",
           "descriptionText": "Build the platform with Kubernetes, ArgoCD and AWS."}

    first = analyzer.process_jobs([dict(job)], cache_path=cache)
    assert (tmp_path / "score_cache.json").exists()
    assert first[0]["SemanticCategory"] == "Platform Engineering"

    # Second run: the expensive per-job work must be skipped entirely.
    spy_skills = mocker.spy(analyzer, "extract_tech_skills")
    spy_cat = mocker.spy(analyzer, "categorize_job")
    spy_score = mocker.spy(analyzer, "compute_match_score")
    second = analyzer.process_jobs([dict(job)], cache_path=cache)

    assert spy_skills.call_count == 0
    assert spy_cat.call_count == 0
    assert spy_score.call_count == 0
    assert second[0]["SemanticCategory"] == "Platform Engineering"


def test_process_jobs_cache_invalidated_by_text_change(analyzer, tmp_path, mocker):
    cache = str(tmp_path / "score_cache.json")
    analyzer.process_jobs(
        [{"id": "job-1", "title": "Platform Engineer", "descriptionText": "Kubernetes."}],
        cache_path=cache,
    )
    # Same id, different text → fingerprint mismatch → must recompute.
    spy_cat = mocker.spy(analyzer, "categorize_job")
    analyzer.process_jobs(
        [{"id": "job-1", "title": "Platform Engineer", "descriptionText": "Totally different now: Terraform."}],
        cache_path=cache,
    )
    assert spy_cat.call_count == 1


# ---------------------------------------------------------------------------
# Transparent fit signal (FitScore / CVSkillOverlap / WhyMatched)
# ---------------------------------------------------------------------------

def test_apply_fit_rewards_cv_skill_overlap(analyzer):
    # Pin the CV's tools so the overlap is deterministic regardless of dummy CV.
    analyzer.cv_skills = {"Kubernetes", "AWS", "Terraform"}
    job = {"SemanticMatchScore": 80.0, "SemanticCategory": "Platform Engineering",
           "TechSkills": ["Kubernetes", "AWS", "Go"]}
    analyzer._apply_fit(job)
    assert job["CVSkillOverlap"] == 2                       # Kubernetes + AWS
    # FitScore = 0.7*80 + 0.3*(2/10*100) = 56 + 6 = 62.0
    assert job["FitScore"] == 62.0
    assert "Kubernetes" in job["WhyMatched"] and "2 CV tools" in job["WhyMatched"]


def test_apply_fit_no_overlap(analyzer):
    analyzer.cv_skills = {"Rust"}
    job = {"SemanticMatchScore": 50.0, "SemanticCategory": "SRE", "TechSkills": ["Go"]}
    analyzer._apply_fit(job)
    assert job["CVSkillOverlap"] == 0
    assert job["FitScore"] == 35.0                          # 0.7*50 + 0
    assert "no shared tools" in job["WhyMatched"]


# ---------------------------------------------------------------------------
# Role-family rejection (classify_role_family)
# ---------------------------------------------------------------------------
# Every case below is a REAL title from jobs_output/. `categorize_job()` is
# forced-choice and cannot reject, so this is the only place off-target postings
# get filtered. Both directions are locked: junk must be rejected AND genuine
# infra roles must survive. See the design notes in semantic_job_analyzer.py.

# These must exceed 300 chars: classify_role_family() downgrades an infra title
# to 'unclear' below that, because a description too short to state requirements
# cannot support an on-target verdict. Real JDs in the corpus average ~4,400
# chars, so the previous ~130-char fixtures were unrealistically short.
_INFRA_JD = ("We run Kubernetes on AWS with Terraform, Argo CD and Prometheus. "
             "You will own the CI/CD pipelines and improve observability across "
             "the platform. Experience with Linux, VPC and IAM is expected, and "
             "you will participate in the on-call rotation and incident reviews. "
             "We value automation, infrastructure as code and a strong ownership "
             "mindset across the whole delivery lifecycle.")
_WEB_JD = ("You will build React components with Redux and Tailwind CSS, "
           "own the responsive design system and improve the browser bundle. "
           "Strong HTML and SCSS skills are required, along with experience of "
           "webpack, vite and Storybook. You will work closely with designers in "
           "Figma to deliver a polished user interface across all breakpoints.")
_PROSE_JD = ("You will join our Developer Experience Platform team and shape how "
             "product teams build and deliver software across the company. " * 12)


@pytest.mark.parametrize("title, expected", [
    # --- hard off-discipline: rejected whatever the description says
    ("Field-Programmable Gate Arrays Engineer", "off-target"),
    ("Senior QA Engineer", "off-target"),
    ("Test Engineer for In-Car-Entertainment (ICE) System", "off-target"),
    ("Senior Value Engineer - Scale Team", "off-target"),
    ("Enterprise Account Executive- Germany", "off-target"),
    ("Customer Success Engineer - YouTrack", "off-target"),
    ("(Lead/Senior) Technical Product Manager - Platform Engineering", "off-target"),
    ("Senior Engineering Program Manager", "off-target"),
    ("Mechanical Engineer", "off-target"),
    ("DSP Chemical Engineer", "off-target"),
    ("Android Engineer", "off-target"),
    ("Senior Avionics System Engineer - Electrical Power System", "off-target"),
    ("Autonomous Systems Engineer (UAV / Drone)", "off-target"),
    ("Field Service Engineer - Germany (f/m/x)", "off-target"),
    ("IT Support Specialist (all genders)", "off-target"),
    ("In-House Legal Counsel", "off-target"),
    # --- families added 2026-08-09 after reading a full run end to end
    ("Postdoctoral Fellow in Lunar Positioning and Timing", "off-target"),
    ("Lecturer/Senior Lecturer - Databases", "off-target"),
    ("Bar & Lounge Service Staff (m/f/d)", "off-target"),      # was Platform Engineering
    ("Senior Video Producer", "off-target"),                    # was Platform Engineering
    ("Lead Unity Software Engineer (Gameplay)", "off-target"),  # non-adjacent game words
    ("Lead C++ Software Engineer (Gameplay)", "off-target"),
    ("Robotics Software Engineer, Amazon Robotics R&D", "off-target"),
    ("Go to Market Engineer / RevOps Engineer", "off-target"),
    ("Manufacturing Engineer - plating (m/w/d)", "off-target"),
    ("Aircraft System Engineer (m/f/d) AIRBUS", "off-target"),
    ("3DX Platform Administrator | Remote | Portugal", "off-target"),
    ("Senior Software Development Engineer in Test", "off-target"),
    ("Solutions Architect, Graduate Program", "off-target"),
    ("Senior Director - Performance Marketing", "off-target"),
    ("Senior Business Intelligence Analyst (f/d/m)", "off-target"),
    # --- genuine infra: must survive even with an infra-poor description
    ("Senior Platform Engineer", "on-target"),
    ("Site Reliability Engineer", "on-target"),
    ("(Sr.) Linux / DevOps Engineer (f/m/d)", "on-target"),
    ("Cloud Infrastructure Engineer", "on-target"),
    ("Senior ML Ops Engineer, AI Platform Team", "on-target"),
    ("Kubernetes Administrator", "on-target"),
    ("FinOps Engineer (w/m/d)", "on-target"),
])
def test_role_family_title_verdicts(title, expected):
    from scraper.semantic_job_analyzer import classify_role_family
    family, _ = classify_role_family(title, _INFRA_JD)
    assert family == expected, f"{title!r} -> {family}"


def test_infra_keep_overrides_app_dev_title():
    """A separate keep-pattern (not an inline lookahead) is load-bearing: the
    stage-1 filter learned that a lookahead only looks FORWARD, so the infra term
    must be found anywhere in the title. Real posting from the corpus."""
    from scraper.semantic_job_analyzer import classify_role_family
    for title in ["Full-Stack Engineer & Infrastructure Co-Builder",
                  "Senior Fullstack Engineer - Developer Platform (f/m/d)",
                  "Software Engineer, Full-Stack & DevOps (all genders, full time)"]:
        family, reason = classify_role_family(title, _INFRA_JD)
        assert family == "on-target", f"{title!r} -> {family} ({reason})"


def test_industry_verticals_never_reject_an_infra_role():
    """The `Automotive` lesson from the stage-1 filter, re-learned here: an
    industry word must never outrank the role. Blocking `aviation` rejected
    `Senior DevOps & Engineering Platform Engineer - Aviation` at FitScore 92.1 --
    the worst false positive of the second validation pass. The distinction is
    which word is the ROLE: `Robotics Software Engineer` is a robotics job,
    `SRE - Robotics Platform` is an SRE job."""
    from scraper.semantic_job_analyzer import classify_role_family
    for title in ["Senior DevOps & Engineering Platform Engineer - Aviation",
                  "Sr Solutions Architect GenAI, Automotive",
                  "Platform Engineer, Manufacturing Systems",
                  "SRE - Robotics Platform"]:
        family, reason = classify_role_family(title, _INFRA_JD)
        assert family == "on-target", f"{title!r} -> {family} ({reason})"


def test_dotnet_pattern_matches_after_a_space():
    r"""`\b\.net` can never match "Lead .NET Engineer" -- \b needs a word char
    before the dot, and there is a space there."""
    from scraper.semantic_job_analyzer import _ROLE_SOFT_OFF
    assert _ROLE_SOFT_OFF.search("Lead .NET Engineer (m/f/d)")
    assert _ROLE_SOFT_OFF.search(".NET Software Engineer (m/f/d) - Berlin")


def test_pure_frontend_and_fullstack_are_rejected():
    from scraper.semantic_job_analyzer import classify_role_family
    for title in ["Frontend Developer", "Senior Full Stack Engineer",
                  "Web Developer (m/f/d)", "React Developer"]:
        family, _ = classify_role_family(title, _WEB_JD)
        assert family == "off-target", title


def test_app_dev_title_is_not_rescued_by_stack_vocabulary():
    """An `infra >= 5` stack rescue for app-dev titles was tried and REMOVED on
    2026-08-09. Fintech backend ads name AWS/Kubernetes/CI-CD in passing, so it
    readmitted eight of them, including `Backend Software Engineer - Golang or
    Java` at infra 13. A backend JD mentioning Kubernetes is still a backend job.
    The only escape hatch is an explicit infra term in the TITLE."""
    from scraper.semantic_job_analyzer import classify_role_family
    heavy = ("Kubernetes, AWS, Terraform, Argo CD, Prometheus, Grafana, Linux, "
             "CI/CD, observability, on-call, VPC, IAM, Docker, Helm, incident "
             "response and capacity planning are all part of this role. You will "
             "own the delivery pipeline end to end, run the on-call rotation, and "
             "drive infrastructure as code across every environment we operate.")
    assert classify_role_family("Senior Backend Engineer", heavy)[0] == "off-target"
    thin = ("We use Java and Spring Boot for our services. You will work on the "
            "checkout domain, own features end to end, and collaborate with "
            "product managers on the roadmap. We value clean code and testing.")
    assert classify_role_family("Senior Backend Engineer", thin)[0] == "off-target"
    # ...but an infra term in the title still wins.
    assert classify_role_family("Backend / SRE Engineer", heavy)[0] == "on-target"


def test_it_support_pattern_does_not_match_ai_first():
    """Regression: an earlier version listed `first` as a bare alternative and
    rejected `Senior Infrastructure Engineer (AI-First)` at FitScore 74.4 -- the
    worst false positive found during validation."""
    from scraper.semantic_job_analyzer import classify_role_family
    family, reason = classify_role_family("Senior Infrastructure Engineer (AI-First)", _INFRA_JD)
    assert family == "on-target", reason


def test_prose_only_jd_is_never_rejected_for_naming_no_tools():
    """Regression: a 'zero infra vocabulary => off-target' rule was implemented
    and removed the same day. It rejected MOIA's `(Senior) Platform Engineer`,
    GetYourGuide's `Senior ML Ops Engineer` and Cohere's `Forward Deployed
    Engineer, Agentic Platform` -- all genuine targets that describe the role in
    prose. Absence of stack vocabulary is not evidence of absence of infra work."""
    from scraper.semantic_job_analyzer import classify_role_family
    assert classify_role_family("(Senior) Platform Engineer (all genders)", _PROSE_JD)[0] == "on-target"
    # An uninformative title with a prose-only JD must be 'unclear', never rejected.
    family, _ = classify_role_family("Founding Engineer", _PROSE_JD)
    assert family == "unclear"


def test_process_jobs_labels_every_row_without_dropping(analyzer, mocker):
    """`process_jobs` itself only LABELS -- the drop happens later, in main(),
    so the analyzer stays usable for analysis over the full set."""
    mocker.patch.object(analyzer, "extract_tech_skills", return_value=([], []))
    mocker.patch.object(analyzer, "categorize_job", return_value="Platform Engineering")
    mocker.patch.object(analyzer, "compute_match_score", return_value=0.5)
    mocker.patch.object(analyzer, "extract_salary", return_value=None)
    jobs = [{"id": "1", "title": "Senior Platform Engineer", "descriptionText": _INFRA_JD},
            {"id": "2", "title": "Frontend Developer", "descriptionText": _WEB_JD}]
    out = analyzer.process_jobs(jobs, cache_path=None)
    assert len(out) == 2, "process_jobs must never drop a row"
    assert out[0]["RoleFamily"] == "on-target"
    assert out[1]["RoleFamily"] == "off-target"
    assert all("RoleFamilyReason" in j for j in out)


def test_main_drops_off_target_but_keeps_unclear(tmp_path, analyzer, mocker):
    """main() is where the RoleFamily label is acted on. Off-target is dropped
    from the written output; `unclear` is kept, because that bucket holds the
    genuinely adjacent roles (Forward Deployed, AI/ML, Data Engineering)."""
    import json as _json
    from scraper import semantic_job_analyzer as sja

    jobs = [
        {"id": "1", "title": "Senior Platform Engineer", "descriptionText": _INFRA_JD},
        {"id": "2", "title": "Frontend Developer", "descriptionText": _WEB_JD},
        {"id": "3", "title": "Senior QA Engineer", "descriptionText": _INFRA_JD},
        {"id": "4", "title": "Founding Engineer", "descriptionText": _PROSE_JD},
    ]
    src = tmp_path / "jobs_24h_x_filtered.json"
    src.write_text(_json.dumps(jobs))

    mocker.patch.object(sja, "SemanticJobAnalyzer", return_value=analyzer)
    mocker.patch.object(analyzer, "extract_tech_skills", return_value=([], []))
    mocker.patch.object(analyzer, "categorize_job", return_value="Platform Engineering")
    mocker.patch.object(analyzer, "compute_match_score", return_value=0.5)
    mocker.patch.object(analyzer, "extract_salary", return_value=None)
    mocker.patch.object(sja, "generate_insight_reports", return_value=None)
    mocker.patch.object(sja, "_download_cv", return_value="dummy.pdf")
    mocker.patch("sys.argv", ["prog", str(src)])
    sja.main()

    out = _json.loads((tmp_path / "jobs_24h_x_filtered_semantic.json").read_text())
    titles = {j["title"] for j in out}
    assert "Senior Platform Engineer" in titles          # on-target
    assert "Founding Engineer" in titles                  # unclear -> KEPT
    assert "Frontend Developer" not in titles             # off-target -> dropped
    assert "Senior QA Engineer" not in titles             # off-target -> dropped
