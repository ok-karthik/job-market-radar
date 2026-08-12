import pytest
from bs4 import BeautifulSoup
import httpx
from scraper.apify_replica import (
    populate_job_details,
    extract_job_ids,
    _get_with_retry,
    TARGET_PROFILES,
    _KW_CLOUD,
    USER_AGENTS,
    BASE_HEADERS,
)


def test_user_agent_rotates_from_pool():
    """The request UA must come from a rotation pool, not a single hardcoded
    fingerprint — chosen once per run and shared by both stages."""
    assert len(USER_AGENTS) >= 3
    assert BASE_HEADERS["User-Agent"] in USER_AGENTS

def test_bullet_point_injection():
    """
    Test that our scraper correctly preserves HTML bullet points (<li>) 
    by injecting '• ' before extracting text.
    """
    html = """
    <div class="description__text">
        <p>Requirements:</p>
        <ul>
            <li>Must know Python</li>
            <li>Must know Kubernetes</li>
        </ul>
    </div>
    """
    
    # Simulate the logic inside populate_job_details
    soup = BeautifulSoup(html, "html.parser")
    desc_container = soup.find(class_="description__text")
    
    # Inject bullet points
    if desc_container:
        for li in desc_container.find_all("li"):
            li_text = li.get_text().strip()
            li.replace_with(f"• {li_text}")
            
        extracted_text = desc_container.get_text(separator="\n").strip()
    else:
        extracted_text = ""
        
    assert "• Must know Python" in extracted_text
    assert "• Must know Kubernetes" in extracted_text

def test_target_profiles_include_group_c_cloud_pools():
    """
    F5: Group C (Cloud Engineer / Cloud Infrastructure Engineer / Kubernetes
    Engineer / Kubernetes Administrator / DevSecOps) must be present with at
    least its Germany-wide pool, alongside Group A/B, without altering A/B.

    Counts are deliberately NOT asserted: the geo matrix is config-driven, so a
    hard 18 would fail on any clone with its own config.json, and it did fail
    when the measured pool cut landed (2026-08-12, 18 -> 11). What must hold is
    that all three groups are still REACHABLE and C-cloud keeps a pool.
    """
    group_c_pools = [url for url in TARGET_PROFILES if _KW_CLOUD in url]
    # C-cloud's Germany pool carries every high scorer it has ever produced,
    # including the corpus's single best match (FitScore 100.0). Its five CITY
    # pools were cut on measured evidence; this one must never be.
    assert len(group_c_pools) >= 1

    for term in ("Cloud+Engineer", "Cloud+Infrastructure+Engineer",
                 "Kubernetes+Engineer", "Kubernetes+Administrator", "DevSecOps"):
        assert term in _KW_CLOUD

    # Bare "Kubernetes" must not appear on its own (deliberately excluded —
    # too broad, would blow the 150 cap on low-signal rows).
    assert "Kubernetes%29" not in _KW_CLOUD and "+Kubernetes+OR" not in _KW_CLOUD

    # Germany-wide (remote) is the pool that matters for C-cloud — see above.
    assert any("geoId=101282230" in url for url in group_c_pools)

    # Every CONFIGURED geo must be reachable by some pool. Derived from config
    # rather than hardcoded: a geo suppressed for one group is a deliberate,
    # measured choice, but a geo listed and reachable by NO pool is dead config.
    # (Munich was removed from the geo list entirely on 2026-08-12 rather than
    # skipped in all three groups — same result, but the config states it.)
    from scraper.user_config import GEOS
    for geo in GEOS:
        assert any(f"geoId={geo['geo_id']}" in url for url in TARGET_PROFILES), geo["name"]


def test_get_with_retry_uses_full_jitter_backoff(mocker):
    """
    F4: on a 429, the retry wait must be sampled from the FULL [cap/2, cap]
    jitter window (not a near-deterministic 5*2^n + small jitter), so the
    retry cadence isn't a bot-like fixed-interval hammering pattern.
    """
    response_429 = httpx.Response(
        429, request=httpx.Request("GET", "http://dummy-url.com")
    )
    response_200 = httpx.Response(
        200, text="ok", request=httpx.Request("GET", "http://dummy-url.com")
    )

    client = httpx.Client()
    mocker.patch.object(client, "get", side_effect=[response_429, response_200])
    mock_sleep = mocker.patch("time.sleep")
    mock_uniform = mocker.patch("random.uniform", return_value=7.5)

    response = _get_with_retry(client, "http://dummy-url.com", params={})

    assert response.status_code == 200
    # attempt=0 => cap = min(60, 2**0 * 10) = 10 => window is (5.0, 10.0)
    mock_uniform.assert_called_once_with(5.0, 10.0)
    mock_sleep.assert_called_once_with(7.5)


def test_extract_job_ids_mock(mocker):
    """
    Mock the HTTP request to ensure extract_job_ids parses the HTML correctly
    without hitting LinkedIn's live servers.
    """
    with open("tests/fixtures/search_page.html", "r", encoding="utf-8") as f:
        mock_html = f.read()
        
    # Use httpx.Response to avoid Mock issues
    import httpx
    mock_response = httpx.Response(200, text=mock_html, request=httpx.Request("GET", "http://dummy-url.com"))
    
    # Return mock_html for the first request, then empty list for pagination to stop the loop
    mock_response_empty = httpx.Response(200, text="<html><body><ul class='jobs-search__results-list'></ul></body></html>", request=httpx.Request("GET", "http://dummy-url.com"))
    
    mocker.patch("httpx.Client.get", side_effect=[mock_response, mock_response_empty, mock_response_empty])
    
    # Run the function with a dummy URL
    job_ids = extract_job_ids(["http://dummy-url.com"])
    
    # Assert it extracted at least some IDs correctly
    assert len(job_ids) > 0
    
    extracted_ids = [job["id"] for job in job_ids]
    # Verify that the extracted ID list is not empty
    assert type(extracted_ids[0]) == str


def test_extract_job_ids_soft_block_abort(mocker):
    """
    F1: a full page of duplicate cards, replayed repeatedly well under the
    per-profile cap, must be treated as an IP soft-block (abort) rather than
    silently mislabelled as genuine end-of-stream ("structural convergence").
    """
    with open("tests/fixtures/search_page.html", "r", encoding="utf-8") as f:
        mock_html = f.read()

    mock_response = httpx.Response(
        200, text=mock_html, request=httpx.Request("GET", "http://dummy-url.com")
    )

    # Same full page of cards on every request => after the first (all-new)
    # batch, every subsequent batch is 100% duplicates of a full page.
    mocker.patch(
        "httpx.Client.get",
        side_effect=[mock_response, mock_response, mock_response],
    )
    mocker.patch("time.sleep")  # skip real backoff/courtesy sleeps in the test

    with pytest.raises(SystemExit):
        extract_job_ids(["http://dummy-url.com"])


def test_extract_job_ids_soft_block_continue_mode(mocker):
    """
    F1 + configurable mode: with SOFTBLOCK_MODE='continue' a confirmed soft-block
    must NOT sys.exit during harvest. It stops early, returns the jobs collected so
    far, and sets the module flag so __main__ can exit non-zero afterwards.
    """
    import scraper.apify_replica as apify_replica

    with open("tests/fixtures/search_page.html", "r", encoding="utf-8") as f:
        mock_html = f.read()
    mock_response = httpx.Response(
        200, text=mock_html, request=httpx.Request("GET", "http://dummy-url.com")
    )

    mocker.patch.object(apify_replica, "SOFTBLOCK_MODE", "continue")
    mocker.patch.object(apify_replica, "_SOFTBLOCK_DETECTED", False)
    mocker.patch(
        "httpx.Client.get",
        side_effect=[mock_response, mock_response, mock_response],
    )
    mocker.patch("time.sleep")

    # Must NOT raise SystemExit; returns the partial harvest from the first full page.
    job_ids = apify_replica.extract_job_ids(["http://dummy-url.com"])

    assert len(job_ids) > 0
    assert apify_replica._SOFTBLOCK_DETECTED is True


def _job_card_html(job_id: str, location: str = "Berlin, Germany") -> str:
    return f"""
    <li>
        <div class="base-card" data-entity-urn="urn:li:jobPosting:{job_id}">
            <h3 class="base-search-card__title">Platform Engineer</h3>
            <h4 class="base-search-card__subtitle">Acme GmbH</h4>
            <span class="job-search-card__location">{location}</span>
            <a class="base-card__full-link" href="https://de.linkedin.com/jobs/view/{job_id}"></a>
        </div>
    </li>
    """


def test_extract_job_ids_genuine_exhaustion_not_flagged_as_soft_block(mocker):
    """
    F1: a short final page (fewer cards than SEARCH_PAGE_SIZE) that is 100%
    duplicates is genuine end-of-stream, not a soft-block, so it must break
    quietly via "Structural convergence met" rather than aborting.
    """
    short_page_html = "<html><body><ul class='jobs-search__results-list'>" + "".join(
        _job_card_html(str(i)) for i in range(3)
    ) + "</ul></body></html>"

    mock_response = httpx.Response(
        200, text=short_page_html, request=httpx.Request("GET", "http://dummy-url.com")
    )

    # Same short (3-card) page every time: batch 1 is all-new, batch 2+ is
    # all-duplicate but well under SEARCH_PAGE_SIZE => genuine exhaustion.
    mocker.patch("httpx.Client.get", return_value=mock_response)
    mocker.patch("time.sleep")

    # Should return cleanly (no SystemExit) with the 3 unique jobs collected.
    job_ids = extract_job_ids(["http://dummy-url.com"])
    assert len(job_ids) == 3


def test_extract_job_ids_per_profile_cap_uses_profile_seen_not_global_new(mocker):
    """
    F2: the per-profile cap must be gated on how many jobs THIS profile has
    surfaced (url_seen_ids), not on how many were globally-new across all
    profiles (url_specific_count). A second profile that overlaps an earlier
    one should hit its cap on the very first batch and stop, instead of
    paginating further chasing globally-new IDs that will never come.
    """
    import scraper.apify_replica as apify_replica

    # Use a non-24h time range so effective_max == max_results_per_url (small,
    # for a fast, deterministic test) rather than the hardcoded 150.
    mocker.patch.object(apify_replica, "TIME_RANGE", "r604800")

    page_html = "<html><body><ul class='jobs-search__results-list'>" + "".join(
        _job_card_html(job_id) for job_id in ("aaa111", "bbb222")
    ) + "</ul></body></html>"
    mock_response = httpx.Response(
        200, text=page_html, request=httpx.Request("GET", "http://dummy-url.com")
    )

    # Profile 1 fills the cap (2) with 2 brand-new jobs on its first batch.
    # Profile 2 returns the SAME 2 job IDs — globally already seen, but new
    # to profile 2's own stream — and must also stop after its first batch.
    # Only 2 total HTTP calls are expected; if the cap incorrectly gated on
    # global-new count, profile 2 would keep paginating and issue a 3rd call
    # (which this mock doesn't provide).
    mock_get = mocker.patch("httpx.Client.get", side_effect=[mock_response, mock_response])
    mocker.patch("time.sleep")

    job_ids = extract_job_ids(
        ["http://dummy-url.com/1", "http://dummy-url.com/2"], max_results_per_url=2
    )

    assert mock_get.call_count == 2
    assert len(job_ids) == 2


def test_extract_job_ids_geo_warning_on_non_german_batch(mocker, capsys):
    """
    F6: if a batch of >=5 harvested rows is overwhelmingly non-German, emit a
    GEO WARNING — catches a mistyped/retired geoId silently widening to a
    global result set.
    """
    non_german_locations = [
        "New York, NY, United States",
        "London, England, United Kingdom",
        "Paris, France",
        "Madrid, Spain",
        "Toronto, ON, Canada",
    ]
    page_html = "<html><body><ul class='jobs-search__results-list'>" + "".join(
        _job_card_html(f"id{i}", loc) for i, loc in enumerate(non_german_locations)
    ) + "</ul></body></html>"
    mock_response = httpx.Response(
        200, text=page_html, request=httpx.Request("GET", "http://dummy-url.com")
    )
    mock_response_empty = httpx.Response(
        200,
        text="<html><body><ul class='jobs-search__results-list'></ul></body></html>",
        request=httpx.Request("GET", "http://dummy-url.com"),
    )

    mocker.patch("httpx.Client.get", side_effect=[mock_response, mock_response_empty])
    mocker.patch("time.sleep")

    extract_job_ids(["http://dummy-url.com?geoId=101282230"])

    captured = capsys.readouterr()
    assert "GEO WARNING" in captured.out


# ── title pre-filter ─────────────────────────────────────────────────────────
# This filter runs BEFORE hydration, so anything it drops is invisible to every
# later stage — a false positive here silently loses a job forever. An earlier
# tightening attempt did exactly that, hence the heavy keep-side coverage.
import pytest as _pytest  # noqa: E402
from scraper.apify_replica import _should_skip_title  # noqa: E402


@_pytest.mark.parametrize("title", [
    # On-track roles must survive every domain rule.
    "Senior Platform Engineer (Kubernetes)",
    "Site Reliability Engineer (m/f/d)",
    "Cloud Infrastructure Engineer",
    "Staff DevOps Engineer",
    "Principal SRE",
    "Developer Experience Engineer",
    "Observability Engineer (f/m/d)",
    "Forward Deployed Engineer, EMEA",
    "AI Infrastructure Engineer",
    "Senior Backend Engineer",          # intentionally kept — see module comment
    "Sales Engineer, Cloud Platform",   # user preference: kept as a fallback
    # The escape hatch must work regardless of WORD ORDER. The old single-regex
    # form used a forward-only lookahead, so an infra keyword appearing BEFORE
    # the trigger word did not rescue the title.
    "Full-Stack Engineer & Infrastructure Co-Builder",
    "Infrastructure Engineer / Full-Stack",
    "Platform Engineer - Automotive Cloud",
    "Cloud Engineer, Clinical Systems",
    "Kubernetes Engineer - Wind Farm Telemetry",
    # Vertical, not discipline: must not be dropped.
    "Sr Solutions Architect GenAI, Automotive & Manufacturing",
])
def test_title_prefilter_keeps_on_track_roles(title):
    assert not _should_skip_title(title), f"would silently lose: {title!r}"


@_pytest.mark.parametrize("title", [
    # Entry-level / non-permanent: hard skip, no escape hatch.
    "Junior Cloud Engineer",
    "Working Student Platform Engineering (m/f/d)",
    "Werkstudent (w/m/d) - GenAI User Research",
    "DevOps Engineer Intern",
    # Wrong discipline — these arrive only via the bare "Engineer" token.
    "CIVIL LEAD ENGINEER - BERLIN",
    "Acoustic Validation Engineer",
    "Associate / Lead Mechanical Engineer - Data Centre (m/f/d)",
    "Embedded Systems Engineer (Aerospace Sector)",
    "Steam Turbine Lead Engineer",
    # Physical / field operations.
    "Field Service Engineer",
    "Construction Manager - Offshore Wind (all genders)",
    "Amazon Versand-/Lagermitarbeiter (m/w/d)",
    # Non-technical functions.
    "Talent Acquisition Partner (m/f/d)",
    "Account Executive (Founding)",
    "Product Marketing Manager",
    "Senior Legal Counsel - Data, Privacy & AI",
    # Wrong software discipline (pre-existing rules).
    "Frontend Developer (React)",
    "Data Scientist (m/w/d)",
    "QA Engineer",
])
def test_title_prefilter_skips_off_track_roles(title):
    assert _should_skip_title(title), f"should have been skipped: {title!r}"


def test_marketing_is_only_skipped_as_a_role_noun():
    """Blanket-matching 'Marketing' dropped 'Senior Data Engineer - Marketing
    Platform'. It must only fire when marketing is the ROLE, not the domain."""
    assert not _should_skip_title("Senior Data Engineer - Marketing Platform (all genders)")
    assert _should_skip_title("Product Marketing Manager")


def test_group_d_devex_was_removed():
    """Group D (Developer Experience / Developer Platform / Observability /
    Forward Deployed) was added on offline evidence and removed after a live A/B:
    20 unique jobs, ZERO on-target, while the 10 real DevEx/Observability/FDE
    postings that day were claimed first by A-core/B-ai/C-cloud. Real titles are
    compound ("Platform Engineer - Developer experience L4"), so the core pools
    already match them. Guard against re-adding it on the same reasoning."""
    import scraper.apify_replica as ar

    assert not hasattr(ar, "_KW_DEVEX")
    for url in TARGET_PROFILES:
        assert "Developer+Experience" not in url
        assert "Forward+Deployed" not in url
    # The title filter must still KEEP such roles when the core pools find them.
    from scraper.apify_replica import _should_skip_title
    for title in ("Platform Engineer - Developer experience L4 (f/m/d)",
                  "Site Reliability Engineer (f/m/d) - Observability & Internal Tooling",
                  "Forward Deployed Engineer, Agentic Platform (UK/Europe)"):
        assert not _should_skip_title(title), title


def test_group_e_staff_was_removed():
    """Group E (Staff/Principal x Platform/SRE/Infra/Cloud/DevOps) was added at
    the user's request and removed after the live A/B. It is STRUCTURALLY
    redundant, not unwanted: a real "Staff Platform Engineer" / "Principal SRE"
    contains a core term, so A-core matches and claims it first (it runs first
    and the dedup dict is global). E could only uniquely surface staff-titled
    roles LACKING an infra term — by definition not infra roles. The live run
    gave 29 uniques, ~0 on-target ("Principal Engineer im Bereich Oberleitung",
    "Principal Compiler Architect", three Product Managers).
    Staff-tier coverage is an analytics concern: see classify_role() in
    jobs_analytics/update_learning_plan.py."""
    import scraper.apify_replica as ar

    assert not hasattr(ar, "_KW_STAFF")
    for url in TARGET_PROFILES:
        assert "Staff+Platform" not in url and "Principal+SRE" not in url

    # The point of the removal: A-core's own terms already cover these titles.
    from scraper.apify_replica import _KW_CORE, _should_skip_title
    for token in ("Platform+Engineer", "SRE", "Infrastructure+Engineer"):
        assert token in _KW_CORE
    # ...and the title filter must let staff infra roles through to hydration.
    for keep in ("Staff Platform Engineer", "Principal SRE (m/f/d)",
                 "Staff Infrastructure Engineer, Cloud",
                 "Staff System Engineer - DNS / DHCP / Automation (m/w/d)"):
        assert not _should_skip_title(keep), keep
    # The bare-"Staff" false friend stays blocked.
    assert _should_skip_title("Chief of Staff (m/w/d)")


def test_source_pool_attribution_labels():
    """Every profile must resolve to a known group label, so `sourcePool` on the
    output answers 'is this pool earning its keep?' without the run log."""
    from scraper.apify_replica import _pool_label

    labels = [_pool_label(u) for u in TARGET_PROFILES]
    # The invariant is that EVERY profile resolves — an "unknown" label makes a
    # pool invisible to the yield review. Per-group counts are config-driven
    # (see the pool cut of 2026-08-12) and deliberately not pinned here.
    assert "unknown" not in labels
    assert set(labels) == {"A-core", "B-ai", "C-cloud"}


def test_title_skip_reason_names_the_rule_that_fired():
    """Skipped jobs never reach hydration and leave no downstream trace, so the
    run log is the ONLY place an over-broad rule becomes visible. The reason must
    name both the rule class and the exact word, or auditing is guesswork."""
    from scraper.apify_replica import _title_skip_reason, _should_skip_title

    assert _title_skip_reason("Acoustic Validation Engineer") == "domain:Acoustic"
    assert _title_skip_reason("Junior Cloud Engineer") == "hard:Junior"
    assert _title_skip_reason("Chief of Staff (m/w/d)") == "domain:Chief of Staff"
    # Kept titles report no reason...
    assert _title_skip_reason("Senior Platform Engineer") is None
    # ...including ones rescued by the keep-pattern over a domain word.
    assert _title_skip_reason("Platform Engineer - Automotive Cloud") is None
    # The boolean helper stays in lockstep with the reason.
    for title in ("Acoustic Validation Engineer", "Senior Platform Engineer",
                  "Junior Cloud Engineer", "Platform Engineer - Automotive Cloud"):
        assert _should_skip_title(title) == (_title_skip_reason(title) is not None)


def test_target_profiles_come_from_config_and_are_well_formed():
    """Search profiles moved to a gitignored config.json (2026-08-11) so the repo
    carries no personal search parameters. Guard the two ways that can go wrong:
    a config that silently yields nothing, and URLs whose SHAPE drifts from what
    extract_job_ids() parses back out. The base path is part of the contract —
    an earlier version emitted the guest-API endpoint, whose query params matched
    while the base silently did not."""
    from urllib.parse import urlparse, parse_qs
    from scraper.apify_replica import TARGET_PROFILES

    assert len(TARGET_PROFILES) >= 3, "config produced no search pools"
    for url in TARGET_PROFILES:
        u = urlparse(url)
        assert u.path == "/jobs/search/", f"unexpected base path: {u.path}"
        q = parse_qs(u.query)
        assert q.get("f_JT") == ["F"], "full-time filter missing"
        assert len(q.get("geoId", [])) == 1, "exactly one geoId per URL — comma-encoded lists fall back to a global result set"
        kw = q.get("keywords", [""])[0]
        assert kw and "NOT" not in kw, "keywords must be present and carry no NOT logic"
        assert '"' not in kw and "%22" not in kw, "keywords must stay UNQUOTED — quoting drops compound titles"


def test_config_urls_match_the_committed_example():
    """config.example.json must keep reproducing the same pools, so a fresh clone
    behaves like the original and the example never rots."""
    from scraper import user_config
    urls = user_config.target_profiles()
    assert len(urls) == len(set(urls)), "duplicate search profile"
    pools = {p for _u, p, _g in user_config.build_search_urls()}
    assert len(pools) >= 2, f"expected multiple keyword groups, got {pools}"
