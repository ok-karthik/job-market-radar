#!/usr/bin/env python3
"""Backfill RoleFamily / RoleFamilyReason into existing *_filtered_semantic.json.

classify_role_family() is pure regex, so this needs no models and no re-scoring --
it rewrites the two fields in place across the whole corpus in seconds. Use this
instead of re-running the semantic stage when only the role-family patterns have
changed. Re-run it any time those patterns are edited.

Safe by design: only ever ADDS/overwrites the two RoleFamily* keys, never removes
a job, never touches raw scrape files (only *_filtered_semantic.json).

    uv run python jobs_analytics/backfill_role_family.py            # all files
    uv run python jobs_analytics/backfill_role_family.py --dry-run  # report only
"""
import argparse, glob, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scraper.semantic_job_analyzer import classify_role_family, ENABLE_GLINER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="jobs_output/*_filtered_semantic.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-off-target", action="store_true",
                    help="Label only; do not remove off-target rows.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        sys.exit(f"no files matched {args.glob!r}")

    grand, changed_files = Counter(), 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            jobs = json.load(f)
        before = [j.get("RoleFamily") for j in jobs]
        for j in jobs:
            j["RoleFamily"], j["RoleFamilyReason"] = classify_role_family(
                j.get("title", ""), j.get("descriptionText", "")
            )
        counts = Counter(j["RoleFamily"] for j in jobs)
        grand.update(counts)
        changed = sum(1 for b, j in zip(before, jobs) if b != j["RoleFamily"])

        # Clear stale EmergingSkills left by cache entries written while GLiNER
        # was enabled (the flag was not part of the score-cache fingerprint
        # until 2026-08-09).
        stale_em = 0
        if not ENABLE_GLINER:
            for j in jobs:
                if j.get("EmergingSkills"):
                    j["EmergingSkills"] = []
                    stale_em += 1

        # Apply the same drop main() applies, so the corpus can be corrected
        # after a pattern change WITHOUT re-running the embedding model — which
        # is the expensive, laptop-heating part. The input _filtered.json still
        # holds every row, so this stays reversible.
        kept = jobs if args.keep_off_target else [j for j in jobs if j["RoleFamily"] != "off-target"]
        print(f"{os.path.basename(path):50} {len(jobs):4} -> {len(kept):4}  "
              f"on={counts['on-target']:4} unclear={counts['unclear']:4} "
              f"off={counts['off-target']:4}  ({changed} reclassified"
              + (f", {stale_em} stale emerging cleared" if stale_em else "") + ")")
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(kept, f, ensure_ascii=False, indent=2)
            changed_files += 1

    total = sum(grand.values())
    print(f"\n{'TOTAL':55} {total:4} rows  "
          f"on={grand['on-target']} unclear={grand['unclear']} off={grand['off-target']}")
    print(f"({'dry run — nothing written' if args.dry_run else f'{changed_files} files rewritten'})")


if __name__ == "__main__":
    main()
