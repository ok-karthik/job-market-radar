import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- CONFIGURATION ---
TIME_RANGE = (
    "r86400"  # Past Week (r2592000 = Month, r604800 = 1 Week, r86400 = 24 Hours)
)

# Max parallel hydration requests. 2 is very conservative — reduces Stage 2 from
# ~60 min (serial) to ~15-20 min while minimizing rate limit triggers.
HYDRATION_CONCURRENCY = 2

# Observed guest-API search page size. Used to tell a soft-block replay (a FULL
# page of duplicates, returned early) from genuine stream exhaustion (an empty
# or short final page).
SEARCH_PAGE_SIZE = 10
SOFTBLOCK_ABORT_THRESHOLD = 2   # consecutive full-page all-duplicate batches

# Behavior when a soft-block is CONFIRMED (>= SOFTBLOCK_ABORT_THRESHOLD consecutive
# full-page all-duplicate batches):
#   "abort"    -> sys.exit(1) immediately (default). Best for INTERACTIVE runs where
#                 you can rotate IP (flight-mode) and re-run right away.
#   "continue" -> stop harvesting early but still hydrate/save whatever was collected,
#                 then exit non-zero so an UNATTENDED (launchd) run's log flags it.
# A soft-block is IP-scoped, so "continue" stops ALL further pools rather than grinding
# through the remaining (also-blocked) pools. Set via env, e.g. in the launchd plist:
#   <key>SOFTBLOCK_MODE</key><string>continue</string>
SOFTBLOCK_MODE = os.environ.get("SOFTBLOCK_MODE", "abort").strip().lower()
_SOFTBLOCK_DETECTED = False  # module-level flag; drives the process exit code

# A fixed User-Agent is a stable fingerprint. Rotate it across runs from a small
# pool of realistic current desktop browsers. Chosen ONCE per process (_SESSION_UA)
# rather than per-request — a session whose UA changes mid-flight looks *more*
# bot-like, not less. This complements the full-jitter backoff / soft-block work.
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
]
_SESSION_UA = random.choice(USER_AGENTS)

BASE_HEADERS = {
    "User-Agent": _SESSION_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Restli-Protocol-Version": "2.0.0",
}

# --- TITLE PRE-FILTER (cheap O(1) skip before the expensive hydration GET) ---
#
# WHY THERE IS SO MUCH NOISE TO FILTER: the search queries below are deliberately
# UNQUOTED (see the Precision-vs-Recall note near TARGET_PROFILES), so LinkedIn
# ORs the individual TOKENS. "Infrastructure Engineer" therefore contributes the
# bare token "Engineer", which matches acoustic, civil, turbine and naval
# engineers. That is the price of recall, and it is paid here rather than by
# narrowing the query — narrowing is what drops good jobs.
#
# The filter is split into three patterns evaluated in order, because a single
# regex with inline `(?!...)` escape hatches only looks FORWARD from the match:
# "Full-Stack Engineer & Infrastructure Co-Builder" was kept (Infra follows) but
# "Infrastructure Engineer / Full-Stack" was dropped (nothing follows). Splitting
# it makes the escape hatch apply to the WHOLE title, which is the behaviour the
# original comment intended.
#
#   1. _TITLE_HARD_SKIP  — never wanted, no exceptions (entry-level / non-permanent)
#   2. _TITLE_KEEP       — a strong infra/platform signal anywhere in the title
#                          overrides every domain rule below
#   3. _TITLE_DOMAIN_SKIP — wrong discipline / wrong function
_TITLE_HARD_SKIP = re.compile(
    r"(?i)\b(Junior|Trainee|Intern|Apprentice|Working[\s-]Student|Werkstudent|Praktikant)\b"
)

# Strong on-track signals. Deliberately generous: an automotive company hiring a
# "Platform Engineer" or a lab hiring a "Cloud Infrastructure Engineer" is still
# a target, and rule 3's docstring says to allow through when in doubt.
_TITLE_KEEP = re.compile(
    r"(?i)\b(Platform|Infrastructure|DevOps|DevSecOps|SRE|Site\s+Reliability|"
    r"Kubernetes|Cloud|MLOps|GitOps|Terraform|Observability|Developer\s+Experience|"
    r"Developer\s+Productivity|Internal\s+Developer)\b"
)

_TITLE_DOMAIN_SKIP = re.compile(
    r"(?i)"
    r"\bFull[\s-]?Stack\b"
    r"|\bFrontend\b"
    r"|\b(QA|Quality Assurance|Test\s+Engineer|Embedded|Hardware)\b"
    r"|\bData\s+(Scientist|Analyst)\b"
    r"|\bMachine\s+Learning\s+Engineer\b"
    # NOTE: 'Backend Engineer' rule intentionally omitted — too many false
    # positives (e.g. 'Senior Backend Engineer' categorised as Platform
    # Engineering by the semantic model). filter_jobs.py handles pure
    # backend roles via full description analysis.
    #
    # Non-software engineering disciplines. These arrive purely via the bare
    # "Engineer" token and are the largest junk family in the corpus.
    # NOTE: 'Automotive' is deliberately NOT here. It is an industry VERTICAL,
    # not a discipline — dropping it cost a "Sr Solutions Architect GenAI,
    # Automotive" (a real LATER-track role) while the genuinely embedded
    # automotive jobs are already caught by Embedded/Hardware/Electrical below.
    r"|\b(Civil|Structural|Mechanical|Electrical|Electronic|Acoustic|Thermal|"
    r"Chemical|Aerospace|Naval|Marine|Mining|Geotechnical|Hydraulic|Turbine|"
    r"Welding|HVAC|BIM|Surveying|Metallurg\w*)\b"
    # Physical / field operations rather than software delivery.
    r"|\b(Field\s+(Service|Applications?)|Construction|Installation|Commissioning|"
    r"Maintenance\s+Technician|Service\s+Technician|Production\s+Line|Offshore|"
    r"Wind\s+(Farm|Turbine)|Warehouse|Logistik)\b"
    # Non-technical functions. 'Marketing' etc. are matched only as the ROLE noun,
    # never as a domain qualifier — a "Data Engineer - Marketing Platform" is a
    # legitimate (if low-scoring) hit, and blanket-matching the word dropped it.
    r"|\b(Recruiter|Talent\s+Acquisition|Account\s+Executive|"
    r"Business\s+Development\s+(Manager|Representative)|"
    r"Sales\s+(Manager|Representative|Executive)|"
    r"(Product\s+)?Marketing\s+(Manager|Specialist|Lead|Analyst)|"
    r"Copywriter|Customer\s+Success|Accountant|Payroll|Procurement|Legal\s+Counsel|"
    # Kept after Group E was removed: it is a non-technical role regardless, and
    # the bare "Staff" token still reaches it via other pools.
    r"Chief\s+of\s+Staff)\b"
    # Clinical / laboratory / life sciences.
    r"|\b(Clinical|Nurse|Pharmac\w+|Laborator\w+|Biolog\w+|Chemist)\b"
    # Obviously unrelated non-tech German roles that waste HTTP requests
    r"|\b(Lagermitarbeiter|Koch|Mitarbeiter\s+Logistik|Verkäufer|Pflegekraft)\b"
    # NOTE: 'Sales Engineer', 'Consultant' and similar are intentionally NOT
    # excluded — per user preference they are kept as lower-ranked fallbacks.
)
# --- END CONFIGURATION ---


def _title_skip_reason(title: str) -> str | None:
    """Why this title is skipped, or None to keep it.

    Returns the matched rule AND the exact word that fired it (e.g.
    ``"domain:Acoustic"``), because that is what makes the pre-filter auditable:
    anything dropped here never reaches hydration, so a single over-broad word
    silently loses jobs forever with no downstream trace. Seeing the firing word
    in the run log turns "did the filter eat something?" into a 10-second check.
    """
    m = _TITLE_HARD_SKIP.search(title)
    if m:
        return f"hard:{m.group(0)}"
    if _TITLE_KEEP.search(title):
        return None
    m = _TITLE_DOMAIN_SKIP.search(title)
    if m:
        return f"domain:{m.group(0)}"
    return None


def _should_skip_title(title: str) -> bool:
    """Return True if the job title is obviously outside the target domains.

    This is a cheap O(1) pre-filter applied before the expensive HTTP hydration
    step.  It must be permissive: when in doubt, allow the job through so that
    ``filter_jobs.py`` can make the definitive call on the full description.
    """
    return _title_skip_reason(title) is not None



def _get_with_retry(client: httpx.Client, url: str, params: dict, retries: int = 3):
    """Synchronous GET with exponential back-off on 429/503 and transient errors.
    Used only by Stage 1 (extract_job_ids) which remains serial.
    """
    for attempt in range(retries):
        try:
            response = client.get(url, params=params)
            if response.status_code in (429, 503):
                # Full-jitter capped exponential backoff (AWS-style): sample the
                # whole [0, cap] window so retry spacing isn't a fixed cadence.
                cap = min(60.0, (2 ** attempt) * 10)
                wait = random.uniform(cap / 2, cap)   # e.g. 5-10s, 10-20s, 20-40s
                print(f" [!] Rate-limited ({response.status_code}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            return response
        except httpx.RequestError as exc:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f" [!] Network error: {exc}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None


async def _get_with_retry_async(
    client: httpx.AsyncClient, url: str, retries: int = 3
) -> httpx.Response | None:
    """Async GET with exponential back-off. Used by Stage 2 concurrent hydration."""
    for attempt in range(retries):
        try:
            response = await client.get(url)
            if response.status_code in (429, 503):
                cap = min(60.0, (2 ** attempt) * 10)
                wait = random.uniform(cap / 2, cap)
                print(f" [!] Rate-limited ({response.status_code}). Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
                continue
            return response
        except httpx.RequestError as exc:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f" [!] Network error: {exc}. Retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)
    return None


async def _hydrate_one(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    job: dict,
    detail_base_url: str,
    pbar: tqdm,
) -> dict | None:
    """Fetch and parse one job's detail page. Runs concurrently under a semaphore."""
    async with semaphore:
        try:
            response = await _get_with_retry_async(
                client, detail_base_url.format(job["id"])
            )
            if not response or response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            desc_container = soup.find("div", class_="description__text")
            if not desc_container:
                return None

            for li in desc_container.find_all("li"):
                li_text = li.get_text().strip()
                li.replace_with(f"• {li_text}")

            desc_text = desc_container.get_text(separator="\n")
            desc_text = re.sub(r"(?i)\n*\s*show more\s*\n*", "\n", desc_text)
            desc_text = re.sub(r"(?i)\n*\s*show less\s*\n*", "\n", desc_text)
            desc_text = re.sub(r"\n{3,}", "\n\n", desc_text).strip()
            job["descriptionText"] = desc_text
            job["seniorityLevel"] = "Not Applicable"
            job["employmentType"] = "Full-time"

            time_tag = soup.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                job["postedAt"] = time_tag["datetime"]
            elif time_tag and time_tag.text:
                job["postedAt"] = time_tag.text.strip()

            criteria_list = soup.find_all(
                "li", class_="description__job-criteria-item"
            )
            for item in criteria_list:
                header = item.find(
                    "h3", class_="description__job-criteria-subheader"
                )
                value = item.find("span", class_="description__job-criteria-text")
                if header and value:
                    header_text = header.text.strip().lower()
                    if "seniority" in header_text:
                        job["seniorityLevel"] = value.text.strip()
                    elif "employment" in header_text:
                        job["employmentType"] = value.text.strip()

            # Per-worker courtesy sleep — staggers requests even within a batch
            await asyncio.sleep(1.5 + random.uniform(0.5, 1.5))
            return job

        except Exception as exc:
            print(f" [!] Failed to hydrate job {job.get('id', '?')} ({job.get('title', '?')}): {exc}")
            return None
        finally:
            pbar.update(1)


def extract_job_ids(target_urls: list, max_results_per_url: int = 2000):
    global _SOFTBLOCK_DETECTED
    search_base_url = (
        "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    )
    all_scraped_jobs = {}

    # Per-profile harvest cap. Raised from a hard-coded 150 to config-driven on
    # 2026-08-11 after the 2026-08-11 run showed Profile 1 (Germany-remote,
    # A-core) hitting exactly 150/150 with EVERY page still returning 10 new
    # ids — i.e. truncated by our cap, not by end-of-stream. Only that one pool
    # was cap-bound; Frankfurt decayed to 1/page and Hamburg hit genuine
    # convergence at 5, so the extra request volume lands on one or two profiles
    # rather than all 18.
    #
    # ⚠️ Raising this also raises HYDRATION work (more ids -> more detail
    # fetches), and that is where the 429s appeared in that same run. If a run
    # starts aborting on soft-block, this is the first thing to put back.
    effective_max = (user_config.MAX_RESULTS_PER_PROFILE
                     if TIME_RANGE == "r86400" else max_results_per_url)

    print(
        f"[*] Starting Stage 1: Harvesting Job IDs across {len(target_urls)} distinct profiles..."
    )
    print(
        f"[*] Configured Time Range: {TIME_RANGE} (Hard UI Cap: {effective_max} jobs per profile)"
    )

    with httpx.Client(
        headers=BASE_HEADERS, timeout=15.0, follow_redirects=True
    ) as client:
        for url_idx, target_url in enumerate(target_urls, 1):
            # Use native URL parsing instead of fragile regex strings
            parsed_url = urlparse(target_url)
            query_params = parse_qs(parsed_url.query)

            # Safely extract core requirements
            keywords = query_params.get("keywords", [""])[0]
            geo_id = query_params.get("geoId", ["101282230"])[0]
            pool_label = _pool_label(target_url)

            # Log targeted telemetry with human-readable mappings
            geo_map = {
                "101282230": "Germany",
                "106967730": "Berlin",
                "101356337": "Munich",
                "100477049": "Frankfurt",
                "106430557": "Rhine-Ruhr",
                "105347383": "Hamburg"
            }
            
            wt_map = {
                "1": "On-site",
                "2": "Remote",
                "3": "Hybrid"
            }
            
            loc_name = geo_map.get(geo_id, f"GeoID:{geo_id}")
            
            wt_raw = query_params.get("f_WT", [])
            if wt_raw:
                wts = []
                for val in wt_raw:
                    wts.extend(val.split(","))
                wt_names = [wt_map.get(w, f"WT:{w}") for w in wts]
                wt_str = " & ".join(wt_names)
            else:
                wt_str = "All Work Types"
                
            display_location = f"{loc_name} ({wt_str})"
            print(f"\n--------------------------------------------------")
            print(f"[Profile {url_idx}/{len(target_urls)}] Target: {display_location}")
            print(f"--------------------------------------------------")

            start_index = 0
            url_specific_count = 0
            consecutive_duplicates = 0
            url_seen_ids = set()
            soft_block_stop = False  # set in SOFTBLOCK_MODE=continue to stop all pools

            while len(url_seen_ids) < effective_max:
                # Rebuild standard parameters accepted natively by the guest endpoint
                params = {
                    "keywords": keywords,
                    "geoId": geo_id,
                    "f_TPR": TIME_RANGE,
                    "start": start_index,
                }

                # Forward structural workplace type array if present
                if "f_WT" in query_params:
                    wt_values = []
                    for val in query_params["f_WT"]:
                        wt_values.extend(val.split(","))
                    params["f_WT"] = wt_values

                # Forward employment contract type array if present
                if "f_JT" in query_params:
                    jt_values = []
                    for val in query_params["f_JT"]:
                        jt_values.extend(val.split(","))
                    params["f_JT"] = jt_values

                try:
                    response = _get_with_retry(client, search_base_url, params=params)
                    if response is None:
                        print(f" [!] All retries exhausted at index {start_index}. Skipping profile.")
                        break

                    # Only reachable when TIME_RANGE is widened (effective_max=2000);
                    # for the default r86400 the 150 cap breaks the loop far sooner.
                    if response.status_code == 400 and start_index >= 950:
                        print(
                            f" [*] Reached maximum native ceiling of LinkedIn public pagination engine."
                        )
                        break

                    if response.status_code != 200:
                        print(
                            f" [!] HTTP Error {response.status_code} at index {start_index}. Skipping profile allocation..."
                        )
                        break

                    if not response.text.strip():
                        break

                    soup = BeautifulSoup(response.text, "html.parser")
                    job_cards = soup.find_all("li")
                    if not job_cards:
                        break

                    new_in_url = 0
                    new_unique_in_batch = 0
                    cards_seen_in_batch = 0   # F3: real job cards, for stride
                    for card in job_cards:
                        id_card = card.find(
                            "div", class_=lambda x: x and "base-card" in x
                        )
                        if not id_card or not id_card.has_attr("data-entity-urn"):
                            continue

                        cards_seen_in_batch += 1
                        job_id = id_card["data-entity-urn"].split(":")[-1]

                        if job_id not in url_seen_ids:
                            url_seen_ids.add(job_id)
                            new_in_url += 1

                        if job_id not in all_scraped_jobs:
                            title_tag = card.find(
                                "h3", class_="base-search-card__title"
                            )
                            company_tag = card.find(
                                "h4", class_="base-search-card__subtitle"
                            )
                            location_tag = card.find(
                                "span", class_="job-search-card__location"
                            )
                            link_tag = card.find("a", class_="base-card__full-link")

                            all_scraped_jobs[job_id] = {
                                "id": job_id,
                                "link": link_tag["href"].split("?")[0]
                                if link_tag
                                else f"https://de.linkedin.com/jobs/view/{job_id}",
                                "title": title_tag.text.strip() if title_tag else "N/A",
                                "companyName": company_tag.text.strip()
                                if company_tag
                                else "N/A",
                                "location": location_tag.text.strip()
                                if location_tag
                                else "N/A",
                                "postedAt": datetime.now().strftime("%Y-%m-%d"),
                                "sourcePool": pool_label,
                                "sourceGeo": geo_map.get(geo_id, geo_id),
                            }
                            new_unique_in_batch += 1
                            url_specific_count += 1

                            if len(url_seen_ids) >= effective_max:
                                break

                    print(
                        f" -> Index {start_index:04d}: Extracted {new_unique_in_batch:02d} unique jobs ({new_in_url} new in profile stream). (Profile Collected: {url_specific_count} | Total Master Dataset Size: {len(all_scraped_jobs)})"
                    )

                    # F6: geo sanity — warn if a batch returns almost nothing that
                    # looks German. Catches a mistyped/retired geoId that silently
                    # widens to a global result set (guest API returns HTTP 200).
                    if new_unique_in_batch >= 5:
                        batch_locs = [
                            all_scraped_jobs[i]["location"].lower()
                            for i in url_seen_ids
                        ]
                        de_hits = sum(
                            1 for l in batch_locs
                            if any(m in l for m in ("germany", "deutschland", "berlin",
                                                    "munich", "münchen", "hamburg",
                                                    "frankfurt", "köln", "cologne",
                                                    "düsseldorf", ", de"))
                        )
                        if batch_locs and de_hits / len(batch_locs) < 0.25:
                            print(
                                f" [!] GEO WARNING: only {de_hits}/{len(batch_locs)} harvested "
                                f"rows look German for {display_location}. "
                                f"Possible geoId fallback to a global set — verify geoId={geo_id}."
                            )

                    if new_in_url == 0:
                        consecutive_duplicates += 1

                        # Distinguish soft-block replay from genuine exhaustion.
                        full_page = cards_seen_in_batch >= SEARCH_PAGE_SIZE
                        early = url_specific_count < effective_max  # not near the cap

                        if full_page and early:
                            # A FULL page that is 100% duplicates, arriving before
                            # we've hit the cap == LinkedIn is replaying page 1 at
                            # us. That's an IP soft-block, NOT the end of results.
                            print(
                                f" [!] SOFT-BLOCK SUSPECTED: full page of duplicates at "
                                f"index {start_index} (well under the {effective_max} cap). "
                                f"LinkedIn is likely replaying cached results for this IP."
                            )
                            if consecutive_duplicates >= SOFTBLOCK_ABORT_THRESHOLD:
                                print(
                                    "\n [!] IP appears rate-limited / soft-blocked.\n"
                                    " [!] Turn on Flight Mode for ~10s, turn it off to obtain a\n"
                                    " [!] fresh mobile IP, then re-run.\n"
                                    " [!] (A soft-block is IP-scoped, so every later pool on this\n"
                                    " [!]  IP would be degraded too.)"
                                )
                                if SOFTBLOCK_MODE == "continue":
                                    # Unattended (launchd) path: keep what we have,
                                    # stop harvesting, flag a non-zero exit at the end.
                                    _SOFTBLOCK_DETECTED = True
                                    soft_block_stop = True
                                    print(
                                        " [*] SOFTBLOCK_MODE=continue: stopping harvest early and "
                                        "processing what was collected so far (will exit non-zero)."
                                    )
                                    break
                                # Default interactive path: abort loudly.
                                print(" [!] Aborting so we don't harvest degraded (stale) data on a burned IP.")
                                sys.exit(1)
                            # First strike: one long, randomized human-plausible
                            # pause, then re-probe the SAME index once more.
                            cooldown = random.uniform(20.0, 40.0)
                            print(f" [*] Backing off {cooldown:.0f}s before re-probing...")
                            time.sleep(cooldown)
                            continue  # retry same start_index; do NOT advance
                        else:
                            # Empty / short final page => genuine end of stream.
                            print(
                                f" [*] Structural convergence met (profile stream end). "
                                f"Moving to subsequent profile stream..."
                            )
                            break
                    else:
                        consecutive_duplicates = 0

                    # F3: advance by real cards parsed, not raw <li> count, so a
                    # stray non-card <li> can't inflate the stride and skip jobs.
                    # Floor of 1 guards against a non-empty page that yielded no
                    # cards wedging the offset (convergence/empty checks still fire).
                    start_index += max(cards_seen_in_batch, 1)
                    time.sleep(1.0 + random.uniform(0.3, 1.5))

                except Exception as exc:
                    print(f" [!] Unexpected error at index {start_index}: {exc}")
                    break

            # SOFTBLOCK_MODE=continue: a confirmed soft-block stops ALL further pools
            # (they share the same burned IP), keeping whatever was already harvested.
            if soft_block_stop:
                break

    return list(all_scraped_jobs.values())


def populate_job_details(job_list: list):
    """Concurrently hydrate full job descriptions using async HTTP.

    Stage 2 pipeline:
    1. Title pre-filter  — skips obvious noise titles before any HTTP call
    2. Async hydration   — HYDRATION_CONCURRENCY workers in parallel
    """
    detail_base_url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}"

    # ── Step 1: Title pre-filter ───────────────────────────────────────────────
    to_hydrate, skipped_rows = [], []
    for j in job_list:
        reason = _title_skip_reason(j.get("title", ""))
        if reason:
            skipped_rows.append((reason, j.get("title", ""), j.get("sourcePool", "?")))
        else:
            to_hydrate.append(j)
    print(
        f"\n[*] Starting Stage 2: Deep hydrating detailed profiles for "
        f"{len(to_hydrate)} / {len(job_list)} harvested rows "
        f"({len(skipped_rows)} skipped by title pre-filter, "
        f"concurrency={HYDRATION_CONCURRENCY})..."
    )
    # Log every skip WITH the rule that fired it. These jobs never reach
    # hydration, so without this they leave no trace anywhere and an over-broad
    # word would quietly delete good roles for months (this has happened before).
    # Sorted by how often each rule fires: a rule at the top of this list that
    # you do not recognise is the one to go audit.
    if skipped_rows:
        counts = {}
        for reason, _, _ in skipped_rows:
            counts[reason] = counts.get(reason, 0) + 1
        print("[*] Title pre-filter — rules that fired (most frequent first):")
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"      {n:4} x  {reason}")
        print("[*] Title pre-filter — skipped titles:")
        for reason, title, pool in sorted(skipped_rows):
            print(f"      [{pool:8}] {reason:28} | {title[:70]}")

    # ── Step 2: Async concurrent hydration ───────────────────────────────────
    async def _run_all():
        semaphore = asyncio.Semaphore(HYDRATION_CONCURRENCY)
        async with httpx.AsyncClient(
            headers=BASE_HEADERS, timeout=20.0, follow_redirects=True
        ) as client:
            with tqdm(total=len(to_hydrate), desc="Hydrating System Profiles") as pbar:
                tasks = [
                    _hydrate_one(semaphore, client, job, detail_base_url, pbar)
                    for job in to_hydrate
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r for r in results
            if isinstance(r, dict) and r.get("descriptionText")
        ]

    hydrated = asyncio.run(_run_all())

    # Hydration yield summary. Detail-page fetches that 404, get rate-limited, or
    # parse empty are silently dropped by _hydrate_one; surface how many so a bad
    # run (e.g. a soft-block on the *detail* endpoint) is visible in the log
    # instead of just showing a small final dataset.
    attempted = len(to_hydrate)
    succeeded = len(hydrated)
    dropped = attempted - succeeded
    rate = (succeeded / attempted * 100.0) if attempted else 0.0
    print(
        f"[*] Stage 2 complete: hydrated {succeeded}/{attempted} rows "
        f"({dropped} dropped, {rate:.0f}% success)."
    )
    if attempted >= 20 and rate < 60.0:
        print(
            f" [!] LOW HYDRATION YIELD ({rate:.0f}%): most detail fetches failed. "
            f"Likely an IP rate-limit / soft-block on the jobPosting endpoint — "
            f"the saved dataset is degraded. Rotate IP before the next run."
        )

    return hydrated

# Safe segmented matrix targeting distinct transaction pools with standard URL parameters.
# NOTE: Each geoId must be its own URL entry. LinkedIn's guest API ignores comma-encoded
# multi-geoId params and falls back to a global/US result set.
#
# Keywords are split into two targeted groups to maximise coverage within the 150-job-per-
# profile UI cap. Each group gets its own pass over every geo, doubling the daily ceiling
# from ~900 to ~1800 raw jobs before deduplication.
#
# IMPORTANT ARCHITECTURAL NOTE: Precision vs Recall
# We intentionally use UNQUOTED strings and DO NOT use `NOT` negations.
# LinkedIn's guest API breaks on complex NOT logic, and strict exact match quotes ("%22")
# drop highly relevant roles (like "Senior Platform Engineer (Kubernetes)" or "Software
# Engineer - Platform"). By running broad, unquoted queries, we maximize recall and let
# the downstream `filter_jobs.py` and `semantic_job_analyzer.py` handle precision and noise reduction.
#
# Group A — Core Infrastructure roles (primary target):
#   Platform Engineer, Platform Engineering, Site Reliability Engineer, SRE, DevOps,
#   Infrastructure Engineer
from . import user_config  # noqa: E402

_KW_CORE = (
    "%28Platform+Engineer+OR+Platform+Engineering"
    "+OR+Site+Reliability+Engineer+OR+SRE+OR+DevOps"
    "+OR+Infrastructure+Engineer%29"
)
#
# Group B — AI & Emerging MLOps (aspirational / learning track):
#   Combines AI Infra, MLOps, AI Engineer, AI Architect, Agentic, GenAI
_KW_AI = (
    "%28AI+Infrastructure+OR+AI+Platform+OR+AI+Platform+Engineer"
    "+OR+MLOps+OR+ML+Platform"
    "+OR+AI+Engineer+OR+AI+Architect"
    "+OR+AI+Reliability+OR+Agentic"
    "+OR+AI+Agent+OR+GenAI+Engineer%29"
)
#
# Group C — Cloud & Cloud-Native (direct CV headline coverage):
#   Cloud Engineer, Cloud Infrastructure Engineer, Kubernetes, DevSecOps.
#   These families are central to the CV (CKA/CKAD, "Cloud Infrastructure",
#   dedicated DevSecOps track) but are NOT substrings of any Group A/B term.
_KW_CLOUD = (
    "%28Cloud+Engineer+OR+Cloud+Infrastructure+Engineer"
    "+OR+Kubernetes+Engineer+OR+Kubernetes+Administrator"
    "+OR+DevSecOps%29"
)
#
# Group D (Developer Platform / DevEx / Observability) — TRIED AND REMOVED
# 2026-07-31, after a live A/B on one 24h window. Do not re-add without reading
# this, because the offline case for it looked strong.
#   Hypothesis: DevEx / Developer Platform / Observability / Forward Deployed
#   roles were unreachable because their distinctive tokens (developer,
#   experience, productivity, observability) appear in no other pool, and titles
#   carrying those phrases had the highest median CV fit of any family.
#   Result: the pool surfaced 20 unique jobs and ZERO on-target ones. The 10
#   genuine DevEx/Observability/FDE postings that day were all claimed first by
#   A-core (8), B-ai (1) and C-cloud (1).
#   Why the offline analysis was wrong: it measured title families in ISOLATION,
#   but real titles are COMPOUND — "Platform Engineer - Developer experience L4",
#   "Site Reliability Engineer - Observability & Internal Tooling". The niche
#   phrase is a QUALIFIER on a core title the A/B/C pools already match, so a
#   dedicated pool has nothing left to claim and paginates into the long tail of
#   weak single-token matches (it returned "Backend Developer", "Senior Design
#   Engineer, Brand & Creative", and a French-language posting).
#   LESSON: validate a candidate pool by checking whether its target titles are
#   already claimed by an earlier pool — `sourcePool` on the output makes that a
#   one-query check — not by measuring a phrase's fit in isolation.
#
# Group E (Staff / Principal × Platform/SRE/Infrastructure/Cloud/DevOps) — TRIED
# AND REMOVED 2026-07-31, same live run as Group D. The user targets Staff infra
# roles, so this is worth understanding before anyone re-adds it.
#   It was NOT removed because Staff roles are unwanted, and NOT on the original
#   (wrong) reason for skipping them — that reason was that Staff/Principal
#   titles scored at or below the corpus median on FitScore, which only measures
#   similarity to the CV as it reads TODAY (a Senior Platform Engineer CV). An
#   aspirational tier scoring lower is expected and says nothing about intent.
#   It was removed because it is STRUCTURALLY REDUNDANT, by the same
#   compound-title logic that killed Group D: a genuine "Staff Platform
#   Engineer" or "Principal SRE" contains a core term, so A-core matches and
#   claims it FIRST (A-core runs first and the dedup dict is global). Group E can
#   therefore only *uniquely* surface staff-titled roles that lack an infra
#   term — which are, by definition, not infra roles. The live run confirmed it:
#   29 unique jobs, 12 staff/principal-titled, ~0 on-target; the uniques were
#   "Principal Engineer im Bereich Oberleitung" (railway overhead lines),
#   "Principal Engineer LCC HVDC systems", "Principal Compiler Architect" and
#   three Product Managers, while A-core independently found the one relevant
#   staff infra role that day ("Staff System Engineer - DNS / DHCP / Automation").
#   Quoting the phrases would not help — it would find exactly what A-core finds.
#   Staff-tier coverage is therefore an ANALYTICS concern, not a scraping one:
#   see `classify_role()` in jobs_analytics/update_learning_plan.py.

# Maps a keyword group to a short label recorded on every harvested job as
# `sourcePool`, so "is this pool earning its keep?" is answerable from the output
# instead of only from the run log. First pool to surface an id wins, which is
# exactly the attribution we want: what a pool finds that no earlier pool did.
_POOL_LABELS = [
    ("A-core", _KW_CORE), ("B-ai", _KW_AI), ("C-cloud", _KW_CLOUD),
]


def _pool_label(target_url: str) -> str:
    """Short group label ('A-core', 'B-ai', 'C-cloud') for a search URL."""
    for label, kw in _POOL_LABELS:
        if kw in target_url:
            return label
    return "unknown"
# WARNING regarding f_WT filters:
# Do NOT use comma-separated arrays for Workplace Type (e.g. f_WT=1,3) on city pools.
# LinkedIn's guest API often chokes on multiple f_WT flags and silently returns 0 results,
# destroying the yield. Always use 'ALL WORK TYPES' (no f_WT) for city pools and let
# the local deduplication logic handle overlapping remote jobs.
# Search profiles come from the gitignored config.json (see user_config.py) so
# the repository carries no personal search parameters. The hand-written list
# that used to live here is preserved verbatim in config.example.json, and
# tests/test_apify_replica.py asserts the rendered URLs are byte-identical to it.
#
# Falls back to the previous hard-coded list ONLY if config yields nothing, so a
# broken config can never silently produce a zero-pool run.
TARGET_PROFILES = user_config.target_profiles() or [
    # ── GROUP A: Core Infrastructure (Platform / DevOps / SRE) ──────────────────
    # Pool A1: Germany-Wide - REMOTE ONLY
    f"https://www.linkedin.com/jobs/search/?f_WT=2&f_JT=F&geoId=101282230&keywords={_KW_CORE}",
    # Pool A2: Berlin - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=106967730&keywords={_KW_CORE}",
    # Pool A3: Munich - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=101356337&keywords={_KW_CORE}",
    # Pool A4: Frankfurt am Main - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=100477049&keywords={_KW_CORE}",
    # Pool A5: Cologne/Duesseldorf (Rhine-Ruhr) - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=106430557&keywords={_KW_CORE}",
    # Pool A6: Hamburg - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=105347383&keywords={_KW_CORE}",
    # ── GROUP B: AI Infrastructure & Emerging (aspirational) ──────────────────────
    # Pool B1: Germany-Wide - REMOTE ONLY
    f"https://www.linkedin.com/jobs/search/?f_WT=2&f_JT=F&geoId=101282230&keywords={_KW_AI}",
    # Pool B2: Berlin - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=106967730&keywords={_KW_AI}",
    # Pool B3: Munich - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=101356337&keywords={_KW_AI}",
    # Pool B4: Frankfurt am Main - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=100477049&keywords={_KW_AI}",
    # Pool B5: Cologne/Duesseldorf (Rhine-Ruhr) - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=106430557&keywords={_KW_AI}",
    # Pool B6: Hamburg - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=105347383&keywords={_KW_AI}",
    # ── GROUP C: Cloud & Cloud-Native (CV headline coverage) ──────────────────────
    # Pool C1: Germany-Wide - REMOTE ONLY
    f"https://www.linkedin.com/jobs/search/?f_WT=2&f_JT=F&geoId=101282230&keywords={_KW_CLOUD}",
    # Pool C2: Berlin - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=106967730&keywords={_KW_CLOUD}",
    # Pool C3: Munich - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=101356337&keywords={_KW_CLOUD}",
    # Pool C4: Frankfurt am Main - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=100477049&keywords={_KW_CLOUD}",
    # Pool C5: Cologne/Duesseldorf (Rhine-Ruhr) - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=106430557&keywords={_KW_CLOUD}",
    # Pool C6: Hamburg - ALL WORK TYPES
    f"https://www.linkedin.com/jobs/search/?f_JT=F&geoId=105347383&keywords={_KW_CLOUD}",
]


if __name__ == "__main__":
    raw_jobs = extract_job_ids(TARGET_PROFILES, max_results_per_url=2000)
    if raw_jobs:
        completed_dataset = populate_job_details(raw_jobs)
        time_range_str = (
            "24h"
            if TIME_RANGE == "r86400"
            else "7d"
            if TIME_RANGE == "r604800"
            else "1m"
            if TIME_RANGE == "r2592000"
            else TIME_RANGE
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_filename = f"jobs_output/jobs_{time_range_str}_{timestamp}.json"

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(completed_dataset, f, ensure_ascii=False, indent=2)

        print(
            f"\n[+] Consolidated Master Dataset saved cleanly as: '{output_filename}'"
        )

    if _SOFTBLOCK_DETECTED:
        print(
            "\n[!] Run finished with a CONFIRMED soft-block (SOFTBLOCK_MODE=continue): "
            "the data above is PARTIAL. Rotate IP before the next run. Exiting non-zero "
            "so the launchd log flags this."
        )
        sys.exit(1)
