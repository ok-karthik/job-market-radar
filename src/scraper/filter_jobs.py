import json
import re
import sys

import pandas as pd
from langdetect import DetectorFactory, detect

# Set seed for consistent language detection
DetectorFactory.seed = 0


def is_german_nlp(text):
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False
    try:
        return detect(text[:500]) == "de"
    except Exception:
        return False


def clean_ui_artifacts(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'(?i)\n*\s*show more\s*\n*', '\n', text)
    text = re.sub(r'(?i)\n*\s*show less\s*\n*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── German-language requirement detection ────────────────────────────────────
# Rewritten 2026-08-09 after 17 of 163 jobs in a single run were found to have
# slipped through. TWO separate bugs, and the second is the structural one:
#
#   1. The requirement patterns were too narrow. All of these were missed:
#      "German from C1 for confident client communication", "German (min. B2
#      level)", "German and English are both mandatory requirements",
#      "Native-level German proficiency is required", "German & English at C1",
#      "communication skills in both German and English (C1/C2)".
#
#   2. The optional check was DOCUMENT-WIDE: `not any(opt_pattern in text)`.
#      A single "German is a plus" anywhere in a posting cancelled a hard C1
#      requirement stated elsewhere in the same posting. This is the same
#      proximity bug that `skill_modality()` in update_learning_plan.py was
#      built to solve -- a modality cue belongs to the mention NEAREST it, not
#      to the whole document.
#
# The rewrite scores each German mention inside its own segment (bullet or
# sentence). Optional beats required WITHIN a segment, because "German is a
# plus, but not required" legitimately contains both kinds of cue; but an
# optional cue in one bullet can no longer cancel a requirement in another.
#
# `\bgerman\b` deliberately does NOT match "Germany" (the trailing 'y' defeats
# the word boundary), so "Right to work in Germany ... is required" is not a
# language requirement. It DOES match "German-speaking", which is one.
_GERMAN_MENTION = re.compile(
    r"\b(german|deutsch|deutschkenntnisse|deutschsprachig\w*)\b", re.IGNORECASE
)
# The German-language cues matter: a posting written in German states its own
# language requirement in German ("Gute Deutschkenntnisse in Wort und Schrift"),
# and an English-only cue list misses those entirely.
_GERMAN_REQ_CUE = re.compile(
    r"\b(required|require|requirement|mandatory|must|necessary|essential|"
    r"fluent|fluency|proficien\w*|native|"
    r"advanced|excellent|strong|good command|business[- ]level|"
    r"[bc][12]|"
    # German-language cues
    r"verhandlungssicher\w*|flie(?:ss|ß)end\w*|muttersprach\w*|"
    r"gute|sehr gute|erforderlich|vorausgesetzt|zwingend|sicher\w*)\b",
    re.IGNORECASE,
)
_GERMAN_OPT_CUE = re.compile(
    r"\b(nice[- ]to[- ]have|is a plus|a plus|beneficial|advantageous|advantage|"
    r"bonus|welcome|desirable|preferred|optional|helpful|"
    r"not (?:required|mandatory|necessary|a must)|no german|"
    r"basic|beginner|willing(?:ness)? to learn|"
    # German-language optional cues
    r"von vorteil|wünschenswert|wuenschenswert|nicht erforderlich|"
    r"nicht zwingend|grundkenntnisse)\b",
    re.IGNORECASE,
)
# Bullet, newline and sentence boundaries. A JD's language line is almost always
# its own bullet, so this is a tight and reliable unit.
_SEGMENT_SPLIT = re.compile(r"[\n•;]|(?<=[a-z0-9)])\.\s")

# Phrases that ARE the requirement, with no separate cue needed. Deliberately
# requires "german" ADJACENT to speaking/speaker -- a bare `speaking` cue would
# fire on "a German company with an English-speaking environment", which states
# the opposite of a German requirement.
_GERMAN_DIRECT_REQ = re.compile(
    r"\bgerman[- ]speak(?:ing|er)\b|\bdeutschsprachig\w*\b|\bgerman[- ]native\b",
    re.IGNORECASE,
)


def _segments(text):
    """Yield (start, end) spans of the bullet/sentence-level segments of `text`."""
    pos, out = 0, []
    for m in _SEGMENT_SPLIT.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return out


def requires_german(text):
    """True when the posting states a German-language requirement.

    Per-segment so an optional mention in one bullet cannot cancel a hard
    requirement in another. Returns False for "German is a plus", "beneficial
    but not mandatory", "nice to have", and for mentions of Germany the country.
    """
    if not isinstance(text, str) or not text:
        return False
    if _GERMAN_DIRECT_REQ.search(text):
        return True
    for start, end in _segments(text):
        seg = text[start:end]
        if not _GERMAN_MENTION.search(seg):
            continue
        # Optional wins inside its own segment: "German is a plus, but not
        # required" carries both kinds of cue and is plainly not a requirement.
        if _GERMAN_OPT_CUE.search(seg):
            continue
        if _GERMAN_REQ_CUE.search(seg):
            return True
    return False


def is_contract_or_part_time(row):
    emp_type = str(row.get("employmentType", "")).lower()
    desc_text = str(row.get("descriptionText", "")).lower()
    if any(
        b in emp_type
        for b in ["contract", "part-time", "freelance", "temporary", "internship"]
    ):
        return True
    contract_patterns = [
        r"type:\s*contract",
        r"\$\d+(?:,\d+)?\s*/\s*hour",
        r"hourly\s+rate",
        r"part-time",
    ]
    return any(re.search(p, desc_text) for p in contract_patterns)





def is_target_location(row):
    """
    Checks if the location is definitively in Germany, or if it is a remote role in the UK/EU.
    Returns False if the location string looks like a US state or non-target country.
    """
    location = row.get("location", "")
    desc_text = str(row.get("descriptionText", "")).lower()

    if not isinstance(location, str):
        return False
    loc = location.lower()

    # 1. Obvious German city/country indicators
    german_markers = [
        "germany",
        "deutschland",
        "berlin",
        "munich",
        "münchen",
        "hamburg",
        "cologne",
        "köln",
        "frankfurt",
        "stuttgart",
        "düsseldorf",
        "dusseldorf",
        "dortmund",
        "essen",
        "hannover",
        "nuremberg",
        "nürnberg",
        "augsburg",
        "wiesbaden",
        "heidelberg",
        "mannheim",
        "karlsruhe",
        "freiburg",
        "leipzig",
        "dresden",
        "potsdam",
        "münster",
        "bonn",
        "mainz",
        "bochum",
        "aachen",
        "constance",
    ]

    # 2. Obvious US state indicators (common abbreviations like ', WA' or ', NY')
    us_state_markers = [
        ", al", ", ak", ", az", ", ar", ", ca", ", co", ", ct",
        ", fl", ", ga", ", hi", ", id", ", il", ", in", ", ia",
        ", ks", ", ky", ", la", ", me", ", md", ", ma", ", mi",
        ", mn", ", ms", ", mo", ", mt", ", ne", ", nv", ", nh",
        ", nj", ", nm", ", ny", ", nc", ", nd", ", oh", ", ok",
        ", or", ", pa", ", ri", ", sc", ", sd", ", tn", ", tx",
        ", ut", ", vt", ", va", ", wa", ", wv", ", wi", ", wy",
    ]

    if any(loc.endswith(us_state) or f"{us_state}," in loc for us_state in us_state_markers):
        if "germany" in loc or "deutschland" in loc:
            return True
        return False

    # 3. ISO country suffix: ends in ', de' (e.g. 'Frankfurt, DE')
    if loc.endswith(", de") or loc == "de":
        return True

    # 4. Direct city/country name match
    if any(marker in loc for marker in german_markers):
        return True

    # 5. German metropolitan / regional area strings that don't contain a city name
    #    e.g. "Stuttgart Region", "Rhine-Ruhr", "Cologne/Bonn Region"
    german_region_patterns = [
        r"rhine.ruhr",
        r"cologne.?bonn",
        r"neckar",
        r"bavarian",
        r"rhineland",
        r"westphalia",
        r"palatinate",
        r"hesse",
        r"saxony",
        r"thuringia",
        r"mecklenburg",
        r"saarland",
        r"swabia",
        r"lower saxony",
        r"north rhine",
        r"baden.w\w+temberg",
    ]
    if any(re.search(p, loc) for p in german_region_patterns):
        return True

    # 6. EU/UK Remote Check
    # Allow EU/UK locations ONLY if they are explicitly remote
    eu_uk_markers = [
        "united kingdom", "uk", "england", "london", "ireland", "dublin",
        "netherlands", "amsterdam", "europe", "eu", "spain", "france", "italy", "poland",
        "sweden", "denmark", "norway", "finland"
    ]
    is_eu_uk = any(marker in loc for marker in eu_uk_markers)
    
    if is_eu_uk:
        if "remote" in loc or "remote" in desc_text or "anywhere" in desc_text:
            return True

    return False


# ── Cross-post de-duplication (additive; NOT part of the stable core filters) ──
# LinkedIn frequently posts the SAME role once per city, each with its own job id
# and (usually) an identical description. Our id-based dedup can't catch these, so
# they survive as near-identical rows and get their skills counted multiple times
# downstream. This collapses them to a single survivor per (title, company).

def _location_priority(location) -> int:
    """Rank a location for choosing which cross-post to KEEP (higher = preferred).

    Berlin (the user's home city) wins; then any German-located listing; then
    everything else (EU/UK-remote, unrecognised).
    """
    loc = str(location).lower()
    if "berlin" in loc:
        return 3
    german_markers = (
        "germany", "deutschland", "münchen", "munich", "hamburg", "frankfurt",
        "cologne", "köln", "düsseldorf", "dusseldorf", "stuttgart", "leipzig",
        "hannover", "nuremberg", "nürnberg", "rhine", "ruhr", "bonn", "essen",
        "dortmund", "mannheim", "karlsruhe", "dresden", "bremen", ", de",
    )
    if any(m in loc for m in german_markers):
        return 2
    return 1


def _desc_tokens(text) -> set:
    """Lowercased word/number token SET of a description (order-insensitive)."""
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two token sets: |A∩B| / |A∪B|."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_cross_posts(df, desc_similarity_threshold: float = 0.90):
    """Collapse the same role cross-posted to multiple cities.

    Two rows are duplicates when they share a normalised (title, companyName)
    AND — if descriptions are available — their descriptions are near-identical
    (word-set Jaccard >= ``desc_similarity_threshold``). This guard prevents two
    genuinely different roles that merely share a generic title at one company
    from being wrongly merged. Within a duplicate cluster we keep one survivor,
    preferring a Berlin listing, then any German-located listing, then whatever
    remains — deterministically (highest priority first, original order among
    equals). Rows with a missing/'N/A' title or company are never collapsed.

    Set ``desc_similarity_threshold=0`` to fall back to pure title+company
    collapsing (ignore descriptions); 1.0 requires an identical token set.
    """
    if df.empty or "title" not in df.columns or "companyName" not in df.columns:
        return df

    df = df.copy()
    title_norm = df["title"].astype(str).str.strip().str.lower()
    company_norm = df["companyName"].astype(str).str.strip().str.lower()

    # Rows lacking a real title/company get a per-row unique key so they never
    # collapse into each other.
    valid = (
        title_norm.ne("") & title_norm.ne("n/a")
        & company_norm.ne("") & company_norm.ne("n/a")
    )
    df["_dup_key"] = title_norm + " @@ " + company_norm
    df.loc[~valid, "_dup_key"] = ["__unique_" + str(i) for i in df.index[~valid]]

    df["_prio"] = (
        df["location"].apply(_location_priority) if "location" in df.columns else 1
    )

    has_desc = "descriptionText" in df.columns
    keep_indices = []

    for key, group in df.groupby("_dup_key", sort=False):
        if len(group) == 1 or str(key).startswith("__unique_"):
            keep_indices.extend(group.index.tolist())
            continue

        # Highest priority first (Berlin > German > other); stable => original
        # order among equals, so the survivor of each cluster is deterministic.
        group_sorted = group.sort_values("_prio", ascending=False, kind="stable")

        survivors = []  # list of (index, token_set) — one per description cluster
        for idx, row in group_sorted.iterrows():
            toks = _desc_tokens(row.get("descriptionText", "")) if has_desc else set()
            matched = False
            for _, s_toks in survivors:
                # No descriptions available => collapse purely on title+company.
                # Otherwise only collapse when descriptions are near-identical.
                if not has_desc or _jaccard(toks, s_toks) >= desc_similarity_threshold:
                    matched = True
                    break
            if not matched:
                survivors.append((idx, toks))
        keep_indices.extend(i for i, _ in survivors)

    result = df.loc[sorted(keep_indices)].drop(columns=["_dup_key", "_prio"])
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_jobs.py <jobs_raw_extract_YYYY-MM-DD.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"Loading raw dataset: {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    initial_count = len(df)

    print("Applying filters...")
    df["descriptionText"] = df["descriptionText"].apply(clean_ui_artifacts)
    df["is_german"] = df["descriptionText"].apply(is_german_nlp)
    df["req_german"] = df["descriptionText"].apply(requires_german)
    df["is_contract"] = df.apply(is_contract_or_part_time, axis=1)
    df["is_wrong_location"] = df.apply(lambda row: not is_target_location(row), axis=1)

    # Final filter: Exclude German/Contract
    df_clean = df[
        ~(
            df["is_german"]
            | df["req_german"]
            | df["is_contract"]
            | df["is_wrong_location"]
        )
    ].copy()

    # Clean up
    cols_to_drop = ["is_german", "req_german", "is_contract", "is_wrong_location"]
    if "descriptionHtml" in df_clean.columns:
        cols_to_drop.append("descriptionHtml")

    df_clean = df_clean.drop(columns=cols_to_drop)

    # Collapse the same role cross-posted across multiple cities (keeps Berlin).
    pre_dedup = len(df_clean)
    df_clean = deduplicate_cross_posts(df_clean)
    collapsed = pre_dedup - len(df_clean)
    if collapsed:
        print(
            f"[*] Collapsed {collapsed} cross-posted duplicate(s) "
            f"(same title + company across cities)."
        )

    # 1. Save as CSV
    output_csv = input_file.replace(".json", "_filtered.csv")
    df_clean.to_csv(output_csv, index=False)

    # 2. Save as JSON
    output_json = input_file.replace(".json", "_filtered.json")
    df_clean.to_json(output_json, orient="records", indent=2)

    print(
        f"\nDone! Filtered {initial_count} down to {len(df_clean)} high-quality roles."
    )
    print(f"\n[+] Results generated:")
    print(f" -> CSV: {output_csv}")
    print(f" -> JSON: {output_json}")
