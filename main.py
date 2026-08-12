import os
import subprocess
import glob
import sys
import time

# Ensure the 'src' directory is in PYTHONPATH so python -m scraper.module works
os.environ["PYTHONPATH"] = os.path.abspath("src")

def main():
    print("=" * 60)
    print(" 🚀 STARTING JOB SCRAPING AND ANALYTICS PIPELINE")
    print("=" * 60)

    run_start = time.time()

    # 1. Run Scraper.
    # NOTE: intentionally NOT check=True. In SOFTBLOCK_MODE=continue the scraper
    # exits non-zero *after* saving a PARTIAL raw dataset (confirmed IP soft-block).
    # We still want to filter/score that partial data, so we tolerate a non-zero
    # exit here and decide what to do based on whether a *fresh* raw file was
    # actually produced by this run (guards against reprocessing yesterday's file
    # if the scraper crashed before writing anything).
    print("\n[1/6] Running Apify Replica Scraper...")
    scraper = subprocess.run(["uv", "run", "python", "-m", "scraper.apify_replica"])
    degraded = scraper.returncode != 0

    # Find raw json files (exclude derived _filtered / _semantic outputs)...
    raw_files = [
        f for f in glob.glob("jobs_output/jobs_*_*.json")
        if not f.endswith("_filtered.json") and not f.endswith("_semantic.json")
    ]
    # ...and keep only those written by THIS run.
    fresh_raw = [f for f in raw_files if os.path.getctime(f) >= run_start]

    if not fresh_raw:
        if degraded:
            print(
                "[-] Scraper exited non-zero and produced NO new raw file this run "
                "(soft-block before any data was saved, or a crash). Nothing to process."
            )
            sys.exit(1)
        print("[-] Error: No new raw data file was produced in jobs_output/.")
        return

    latest_file = max(fresh_raw, key=os.path.getctime)
    if degraded:
        print(
            f"[!] Scraper exited non-zero (soft-block continue mode). "
            f"Processing PARTIAL dataset: {latest_file}"
        )
    else:
        print(f"[+] Found fresh raw dataset: {latest_file}")

    # 2. Run Filter
    print(f"\n[2/6] Running Filter against {latest_file}...")
    subprocess.run(["uv", "run", "python", "-m", "scraper.filter_jobs", latest_file], check=True)

    # 3. Run Semantic Analyzer
    filtered_file = latest_file.replace(".json", "_filtered.json")
    print(f"\n[3/6] Running Semantic Analyzer against {filtered_file}...")
    subprocess.run(["uv", "run", "python", "-m", "scraper.semantic_job_analyzer", filtered_file], check=True)

    # 4. Update Learning Plan
    print("\n[4/6] Updating Learning Plan Markdown...")
    subprocess.run(["uv", "run", "python", "update_learning_plan.py"], cwd="jobs_analytics", check=True)

    # 5. Publish to Notion (best-effort). A Notion/API failure (rate limit,
    # missing NOTION_API_KEY, network) must NOT fail the pipeline — the plan and
    # data are already updated locally, so we tolerate a non-zero exit here.
    # Informational, never fatal. Deliberately WITHOUT --baseline: re-baselining
    # on every run would zero the week-over-week diff and destroy the very drift
    # signal the report exists to surface. Re-baseline by hand, only after
    # acting on a change.
    print("\n[5/6] Skill-gap check (read-only, informational)...")
    gap = subprocess.run(["uv", "run", "python", "jobs_analytics/skill_gap_report.py"])
    if gap.returncode != 0:
        print("[!] Skill-gap report failed (non-fatal) — plan/data are still up to date.")

    print("\n[6/6] Publishing to Notion (best-effort)...")
    notion = subprocess.run(["uv", "run", "python", "jobs_analytics/publish_to_notion.py"])
    if notion.returncode != 0:
        print("[!] Notion publish failed (non-fatal) — local plan/data are still up to date.")

    print("\n" + "=" * 60)
    print(" ✨ PIPELINE COMPLETE! Check the jobs_output/ directory.")
    print("=" * 60)

    if degraded:
        print(
            "\n[!] NOTE: this run was DEGRADED by a CONFIRMED soft-block — the dataset "
            "is PARTIAL. Rotate IP before the next run. Exiting non-zero."
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
