#!/usr/bin/env python3
"""Publish the generated learning-plan markdown to a Notion page.

Replace-body mirror: archives the target page's existing blocks and rewrites
them from ``your learning plan (`profile.plan_markdown`)``. The noisy day-wise pipeline
The page is split so the MAIN page opens on the actionable plan: focus areas →
what is already done → the active learning plan and its tracks → milestones.
Market tables and written reasoning move to a "📊 Market Data & Analysis"
sub-page, with a compact Req% digest generated back onto the main page. Daily
stats are moved into a "📅 Daily Pipeline Stats" sub-page (one collapsible
toggle per date) to keep the main page clean.

The Notion API accepts block objects, NOT markdown (the app can import .md; the
API cannot), so this carries a small markdown->blocks converter for exactly the
structures the plan uses: headings, paragraphs, quotes, dividers, bullet/
numbered lists (one level of nesting), tables, <details> toggles, and inline
**bold** / `code` / [links](url).

Usage:
    uv run python jobs_analytics/publish_to_notion.py            # publish
    uv run python jobs_analytics/publish_to_notion.py --dry-run  # parse only, no writes

Requires ``NOTION_API_KEY`` in the environment or repo-root ``.env``, and the
Notion integration must be shared with the target page (··· → Connections).
Override the target page with ``NOTION_PAGE_ID``.
"""
import argparse
import glob
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
from scraper import user_config as _cfg  # noqa: E402
MARKDOWN_PATH = os.path.join(SCRIPT_DIR, "..", _cfg.PLAN_MARKDOWN)
JOBS_GLOB = os.path.join(SCRIPT_DIR, "../jobs_output/jobs_24h_*_filtered_semantic.csv")
PAGE_ID = os.environ.get("NOTION_PAGE_ID", _cfg.NOTION_PAGE_ID)
NOTION_VERSION = "2022-06-28"
API = "https://api.notion.com/v1"
DAILY_SUBPAGE_TITLE = "📅 Daily Pipeline Stats"
JOBS_SUBPAGE_TITLE = "🎯 Jobs to Apply"
ANALYSIS_SUBPAGE_TITLE = "📊 Market Data & Analysis"

# ── Main-page / sub-page split (added 2026-08-10) ────────────────────────────
# The plan opens with ~500 lines of market tables and written reasoning before
# the first actionable line, so opening the page meant scrolling past the
# analysis every time. Everything between these two headings moves to the
# "📊 Market Data & Analysis" sub-page, leaving the main page as: focus areas →
# what's done → the active plan and its tracks. A compact Req% digest is
# generated back onto the main page so the headline numbers stay one glance away.
#
# Matched on heading TEXT rather than HTML markers because these are
# hand-written sections; update_learning_plan.py owns the marker pairs and must
# keep writing into the same file, so adding markers here would risk a conflict.
# Level-2 sections moved off the main page, in the order given. Each entry is a
# (start, resume) pair: everything from `start` up to but NOT including `resume`.
# Non-contiguous by design — reference material is scattered through the plan.
ANALYSIS_SECTIONS = [
    ("## 📊 MARKET DATA", "## ✅ WHAT YOU HAVE ALREADY COMPLETED"),
    ("## ✅ WHAT YOU HAVE ALREADY COMPLETED", "## 📅 ACTIVE LEARNING PLAN"),
    ("## 🗑️ PERMANENTLY REMOVED FROM THIS PLAN", None),   # None = to end of file
]

# Level-3 sections that stay on the main page but render COLLAPSED. These are
# the verbatim-JD evidence blocks under each track: worth keeping one click away
# (they are why the track says what it says) but they push the actual topic
# tables below the fold when expanded by default.
# NOTE: no "### " prefix — parse_blocks() stores the heading text only, and
# markdown emphasis is stripped into separate rich_text spans, so match the
# flattened plain text ("Go topics nobody asked for", not "**nobody**").
# Only the EVIDENCE blocks collapse — the verbatim-JD quotes and corrections
# that justify each track. Short negative-guidance sections ("Go topics nobody
# asked for", "Explicitly NOT asked for") stay expanded on purpose: they are
# what stops time being spent on the wrong material, and hiding them behind a
# click defeats the point.
COLLAPSE_HEADINGS = (
    "The finding that changes this track",
    "⚠️ Why this track changed",
    "⚠️ Correction — Tier 1",
    "Your question, answered with the JD text",
    "Read this first:",
)
# Applied to FitScore (the analyzer's composite), which is built on a semantic
# score that is min-max normalized per day — so this reads as "today's strongest
# matches" rather than an absolute bar. Override with the env var.
JOBS_SCORE_THRESHOLD = int(os.environ.get("JOBS_SCORE_THRESHOLD", "70"))
JOBS_CAP = 60  # keep the Notion table (and one request) well under limits
# Scrape days rendered as date sub-pages under "🎯 Jobs to Apply".
JOBS_DAYS = int(os.environ.get("JOBS_DAYS", "7"))
CV_PATH = os.path.join(SCRIPT_DIR, "../CV.pdf")
PACE = 0.34  # ~3 req/s, Notion's soft limit
DELETE_WORKERS = 4  # concurrent block deletes; api()'s 429 retry keeps this within Notion's limit

# ML-research / post-training role signals. These roles score high on semantic
# similarity (they share the CV's AI vocabulary) but demand skills that live
# OUTSIDE the infra tool taxonomy (SFT/DPO/GRPO, reward modeling, Megatron/TRL,
# research benchmarks), so neither the cosine score nor tool-overlap flags them.
# We detect them by keyword and sink them to the bottom of the apply list.
_RESEARCH_SIGNALS = re.compile(
    r"\b(?:post-?training|\bsft\b|\bdpo\b|\bgrpo\b|\brlhf\b|reward model\w*|"
    r"fine-?tun\w+|pre-?training|megatron|deepspeed|\btrl\b|\bverl\b|"
    r"research scientist|\bphd\b|peer[- ]review\w*|publications?|"
    r"reinforcement learning|model training|training pipeline)\b")


def role_fit_flag(title, desc):
    """Return '⚠️ ML-research' when a posting looks like an ML-research /
    post-training role outside the user's platform/SRE/infra track."""
    t, d = title.lower(), desc.lower()
    hits = len(set(_RESEARCH_SIGNALS.findall(d)))
    research_title = "research" in t or "scientist" in t
    if (research_title and hits >= 1) or hits >= 2:
        return "⚠️ ML-research"
    return ""


# CV skills, extracted once with the same regex taxonomy the analyzer uses
# (model-free). Empty set if CV.pdf or the taxonomy import is unavailable.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../src"))
_CV_SKILLS = None


def cv_skills():
    global _CV_SKILLS
    if _CV_SKILLS is not None:
        return _CV_SKILLS
    _CV_SKILLS = set()
    try:
        import fitz
        from scraper.semantic_job_analyzer import (
            TECH_SKILLS_PATTERNS, CASE_SENSITIVE_SKILLS)
        text = "".join(p.get_text() for p in fitz.open(CV_PATH))
        for name, pat in TECH_SKILLS_PATTERNS.items():
            flags = 0 if name in CASE_SENSITIVE_SKILLS else re.IGNORECASE
            if re.search(pat, text, flags):
                _CV_SKILLS.add(name)
    except Exception as e:
        print(f"[!] CV skills unavailable ({e}); skill-match column will show '—'")
    return _CV_SKILLS


# ── token loading ─────────────────────────────────────────────────────────────
def load_token():
    tok = os.environ.get("NOTION_API_KEY")
    if tok:
        return tok.strip()
    env = os.path.join(SCRIPT_DIR, "../.env")
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line.startswith("NOTION_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("NOTION_API_KEY not found (checked env and repo-root .env)")


# ── inline rich text ──────────────────────────────────────────────────────────
_INLINE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<bold>\*\*[^*]+\*\*)"
    r"|(?P<link>\[[^\]]+\]\([^)]+\))"
    r"|(?P<ital_u>(?<!\w)_[^_\n]+?_(?!\w))"
    r"|(?P<ital_a>(?<![\w*])\*[^*\n]+?\*(?![\w*]))"
)


# Notion rejects the whole request with "Invalid URL for link" if any span
# carries a non-URL. Job descriptions contain markdown-shaped text that is not
# markdown — one JD has "[...](Optional)", which the inline parser happily turned
# into a link with url="Optional" and 400'd a 31-job page.
_VALID_URL = re.compile(r"^(https?://|mailto:|tel:)", re.I)


def _txt(content, bold=False, italic=False, code=False, link=None):
    t = {"type": "text", "text": {"content": content}}
    if link and _VALID_URL.match(str(link).strip()):
        t["text"]["link"] = {"url": str(link).strip()}
    t["annotations"] = {
        "bold": bold, "italic": italic, "strikethrough": False,
        "underline": False, "code": code, "color": "default",
    }
    return t


def rich_text(s):
    """Parse a line of inline markdown into a list of Notion rich_text objects.

    Handles `code`, **bold**, [text](url), and _italic_ / *italic*. The italic
    lookarounds require a non-word boundary so intra-word underscores (e.g.
    ``jobs_24h``) and URLs are never treated as emphasis.
    """
    s = "" if s is None else s
    out, pos = [], 0
    for m in _INLINE.finditer(s):
        if m.start() > pos:
            out.append(_txt(s[pos:m.start()]))
        g = m.lastgroup
        tok = m.group()
        if g == "code":
            out.append(_txt(tok[1:-1], code=True))
        elif g == "bold":
            out.append(_txt(tok[2:-2], bold=True))
        elif g == "link":
            lm = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            if _VALID_URL.match(lm.group(2).strip()):
                out.append(_txt(lm.group(1), link=lm.group(2)))
            else:
                # Not a link — keep the literal text so nothing is lost.
                out.append(_txt(tok))
        else:  # ital_u or ital_a
            out.append(_txt(tok[1:-1], italic=True))
        pos = m.end()
    if pos < len(s):
        out.append(_txt(s[pos:]))
    return _fit_rich_text(out) or [_txt("")]


# Notion hard limits on a single rich_text array. Exceeding either is a 400
# validation_error, and the message names the offending block index only — so
# these are enforced here, at the one place every block's rich_text is built.
RICH_TEXT_MAX_ITEMS = 100
RICH_TEXT_MAX_CHARS = 2000


def _fit_rich_text(spans):
    """Collapse a rich_text list so Notion accepts it.

    Two rules, in order:

    1. **Merge adjacent spans with identical annotations.** Markdown like
       ``**a** **b**`` yields three spans (bold, plain space, bold) where one or
       two would do; a heavily formatted paragraph can therefore blow the
       100-item cap while carrying very little text. Merging is lossless.
    2. **If still over the cap, flatten the tail into one plain span.** Losing
       bold on the overflow is strictly better than losing the block: the run
       that hit this produced a 159-span quote and aborted the whole publish
       after the sub-page had already been created.

    Also splits any span whose content exceeds the 2,000-character limit.
    """
    merged = []
    for sp in spans:
        if (merged and sp.get("annotations") == merged[-1].get("annotations")
                and "link" not in sp["text"] and "link" not in merged[-1]["text"]):
            merged[-1]["text"]["content"] += sp["text"]["content"]
        else:
            merged.append(sp)

    chunked = []
    for sp in merged:
        c = sp["text"]["content"]
        while len(c) > RICH_TEXT_MAX_CHARS:
            head, c = c[:RICH_TEXT_MAX_CHARS], c[RICH_TEXT_MAX_CHARS:]
            cp = dict(sp); cp["text"] = dict(sp["text"], content=head)
            chunked.append(cp)
        sp["text"]["content"] = c
        chunked.append(sp)

    if len(chunked) <= RICH_TEXT_MAX_ITEMS:
        return chunked
    keep = chunked[:RICH_TEXT_MAX_ITEMS - 1]
    tail = "".join(sp["text"]["content"] for sp in chunked[RICH_TEXT_MAX_ITEMS - 1:])
    keep.append(_txt(tail[:RICH_TEXT_MAX_CHARS]))
    return keep


# ── block constructors ────────────────────────────────────────────────────────
def _block(t, **body):
    return {"object": "block", "type": t, t: body}


def heading(level, text):
    return _block(f"heading_{min(level, 3)}", rich_text=rich_text(text))


def paragraph(text):
    return _block("paragraph", rich_text=rich_text(text))


def paragraph_rt(spans):
    """Paragraph from pre-built rich_text spans (e.g. a bare link)."""
    return _block("paragraph", rich_text=_fit_rich_text(spans))


def quote(text):
    return _block("quote", rich_text=rich_text(text))


def divider():
    return _block("divider")


def _list_item(kind, text, children=None):
    body = {"rich_text": rich_text(text)}
    if children:
        body["children"] = children
    return _block(kind, **body)


def toggle(text, children):
    return _block("toggle", rich_text=rich_text(text), children=children or [])


def table(rows):
    """Build a table block. A cell may be a plain string (parsed as inline
    markdown) or an already-built rich_text list (e.g. a linked job title)."""
    width = max(len(r) for r in rows)
    trows = []
    for r in rows:
        cells = [c if isinstance(c, list) else rich_text(c) for c in r]
        while len(cells) < width:
            cells.append(rich_text(""))
        trows.append(_block("table_row", cells=cells))
    return _block("table", table_width=width, has_column_header=True,
                  has_row_header=False, children=trows)


# ── markdown parsing helpers ──────────────────────────────────────────────────
def _is_comment(line):
    return line.strip().startswith("<!--")


def _table_cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_sep_row(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|?\s*$", line)) and "-" in line


def _parse_details(buf):
    text = "".join(buf)
    ms = re.search(r"<summary>(.*?)</summary>", text, re.S)
    summary = re.sub(r"<[^>]+>", "", ms.group(1)).strip() if ms else "Details"
    children = []
    for li in re.findall(r"<li>(.*?)</li>", text, re.S):
        s = re.sub(r"<code>(.*?)</code>", r"`\1`", li, flags=re.S)
        s = re.sub(r"<[^>]+>", "", s).strip()
        if s:
            children.append(_list_item("bulleted_list_item", s))
    return toggle(summary, children)


def _parse_list(lines, i, n):
    """A run of consecutive list lines; indented items nest one level."""
    out = []
    while i < n:
        raw = lines[i].rstrip("\n")
        if not raw.strip():
            break
        mb = re.match(r"^(\s*)[-*]\s+(.*)$", raw)
        mn = re.match(r"^(\s*)\d+\.\s+(.*)$", raw)
        if not mb and not mn:
            break
        indent = len(mb.group(1) if mb else mn.group(1))
        text = mb.group(2) if mb else mn.group(2)
        kind = "numbered_list_item" if mn else "bulleted_list_item"
        node = _list_item(kind, text)
        if indent >= 2 and out:
            parent = out[-1]
            parent[parent["type"]].setdefault("children", []).append(node)
        else:
            out.append(node)
        i += 1
    return out, i


def parse_blocks(lines):
    blocks, i, n = [], 0, len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or _is_comment(raw):
            i += 1
            continue
        if re.match(r"^-{3,}$", stripped):
            blocks.append(divider()); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocks.append(heading(len(m.group(1)), m.group(2))); i += 1; continue
        if stripped.startswith("<details>"):
            j, buf = i, []
            while j < n and "</details>" not in lines[j]:
                buf.append(lines[j]); j += 1
            if j < n:
                buf.append(lines[j]); j += 1
            blocks.append(_parse_details(buf)); i = j; continue
        if stripped.startswith("|"):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                if not _is_sep_row(lines[i]):
                    tbl.append(_table_cells(lines[i]))
                i += 1
            if tbl:
                blocks.append(table(tbl))
            continue
        if stripped.startswith(">"):
            qlines = []
            while i < n and lines[i].strip().startswith(">"):
                qlines.append(re.sub(r"^\s*>\s?", "", lines[i].rstrip("\n")).rstrip())
                i += 1
            blocks.append(quote("\n".join(qlines)))
            continue
        if re.match(r"^\s*[-*]\s+", raw) or re.match(r"^\s*\d+\.\s+", raw):
            items, i = _parse_list(lines, i, n)
            blocks.extend(items)
            continue
        blocks.append(paragraph(stripped)); i += 1
    return blocks


def parse_daily_toggles(daily_lines):
    """Each '### <date>' heading becomes a toggle; its bullets become children."""
    toggles, title, kids = [], None, []
    for raw in daily_lines:
        line = raw.rstrip("\n")
        s = line.strip()
        if _is_comment(line) or not s:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", s)
        if m:
            if title is not None:
                toggles.append(toggle(title, kids))
            title, kids = m.group(1), []
            continue
        mb = re.match(r"^\s*[-*]\s+(.*)$", line)
        if mb and title is not None:
            kids.append(_list_item("bulleted_list_item", mb.group(1)))
    if title is not None:
        toggles.append(toggle(title, kids))
    return toggles


# ── "Jobs to Apply" page (from the latest filtered_semantic.csv) ──────────────
def load_apply_jobs(threshold, cap):
    """Newest scrape only — kept for the single-day path and as a fallback."""
    files = sorted(glob.glob(JOBS_GLOB))
    if not files:
        return None, threshold, "FitScore", []
    return _load_one_csv(files[-1], threshold, cap)


def _load_one_csv(path, threshold, cap):
    """Read the most recent filtered_semantic.csv, keep rows scoring >= threshold,
    best first, capped. Returns (date_str, actual_threshold, score_col, [jobs]).

    Ranks on the analyzer's composite ``FitScore`` (0.7 semantic + 0.3 CV-skill
    overlap) when the column is present, falling back to ``SemanticMatchScore``
    for older CSVs. Ranking on the raw cosine score put jobs like a "Senior
    Backend Engineer" (100 semantic, 4 CV tools) above a Platform role with 13 —
    which is exactly the failure FitScore was introduced to fix.
    """
    m = re.search(r"jobs_24h_(\d{8})_", os.path.basename(path))
    date = m.group(1) if m else "?"
    if len(date) == 8:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    df = pd.read_csv(path)
    score_col = "FitScore" if "FitScore" in df.columns else "SemanticMatchScore"
    if score_col not in df.columns:
        return date, threshold, score_col, []
    df = df[df[score_col] >= threshold].copy()
    df = df.sort_values(score_col, ascending=False).head(cap)
    # Prefer the overlap the analyzer already computed against the same CV; only
    # re-extract from CV.pdf when reading a CSV that predates the column.
    cvs = cv_skills() if "CVSkillOverlap" not in df.columns else set()
    jobs = []
    for _, r in df.iterrows():
        sal = ""
        if pd.notna(r.get("SalaryMin")):
            mn = r["SalaryMin"]
            mx = r["SalaryMax"] if pd.notna(r.get("SalaryMax")) else mn
            sal = f"€{mn / 1000:.0f}k–€{mx / 1000:.0f}k"
        link = str(r.get("link", "") or "")
        title = str(r.get("title", "N/A"))
        desc = str(r.get("descriptionText", "") or "")
        job_tools = [s.strip() for s in str(r.get("TechSkills", "") or "").split(",") if s.strip()]
        if pd.notna(r.get("CVSkillOverlap")):
            n_have = int(r["CVSkillOverlap"])
        else:
            n_have = len([s for s in job_tools if s in cvs])
        skill = f"{n_have}/{len(job_tools)}" if job_tools else "—"
        jobs.append({
            "score": int(round(r[score_col])),
            "sem": int(round(r["SemanticMatchScore"])) if pd.notna(r.get("SemanticMatchScore")) else None,
            "skill": skill,
            "flag": role_fit_flag(title, desc),
            "title": title,
            "company": str(r.get("companyName", "N/A")),
            "location": str(r.get("location", "") or ""),
            "salary": sal or "—",
            "link": link if link.startswith("http") else "",
            "id": str(r.get("id", "") or ""),
            "desc": desc,
        })
    # Genuine matches first (by score); ⚠️ ML-research roles sink to the bottom.
    jobs.sort(key=lambda j: (j["flag"] != "", -j["score"]))
    return date, threshold, score_col, jobs


def jobs_page_blocks(date, threshold, jobs, score_col="FitScore"):
    flagged = sum(1 for j in jobs if j["flag"])
    what = ("Fit = 0.7 × semantic similarity + 0.3 × CV-tool overlap"
            if score_col == "FitScore" else "raw semantic similarity")
    intro = paragraph(
        f"Top matches from the {date} scrape — {score_col} ≥ {threshold}, "
        f"best first ({len(jobs)} shown"
        + (f", {flagged} flagged" if flagged else "") + "). "
        f"Ranked on {what}. Click a title to open the posting. "
        "'Skill' = your CV tools ∩ the job's tools. "
        "⚠️ ML-research roles (post-training/eval/research) are sunk to the bottom — "
        "they score high on vocabulary overlap but sit outside your infra track. "
        "Tailor the top few; fast-apply the rest.")
    rows = [["Fit", "Skill", "Title", "Company", "Location", "Salary"]]
    for j in jobs:
        display = (j["flag"] + " " + j["title"]) if j["flag"] else j["title"]
        title_cell = [_txt(display, link=j["link"])] if j["link"] else rich_text(display)
        rows.append([str(j["score"]), j["skill"], title_cell,
                     j["company"], j["location"], j["salary"]])
    return [intro, table(rows)]


def load_apply_jobs_by_date(threshold, cap, days):
    """Return [(date, [jobs])] newest date first, over the last *days* scrapes.

    **Deduplicated across the window, keeping the EARLIEST appearance.** A
    posting that straddles midnight is re-scraped by the next 24h window — 6% of
    rows over the last 7 days — so without this the same job repeats across
    consecutive date pages. Keeping the first sighting makes each page read as
    "what was new that day", which is the useful stream when you are working
    through a backlog.
    """
    files = sorted(glob.glob(JOBS_GLOB))[-days:]
    seen, out = set(), []
    for path in files:                                   # oldest first
        m = re.search(r"jobs_24h_(\d{8})_", os.path.basename(path))
        date = m.group(1) if m else "?"
        if len(date) == 8:
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        _d, _t, score_col, jobs = _load_one_csv(path, threshold, cap)
        fresh = [j for j in jobs if j["id"] and j["id"] not in seen]
        for j in fresh:
            seen.add(j["id"])
        if not fresh:
            continue
        # Some days have two scrapes (e.g. 20260731_0938 and _2110). Merge them
        # into one page rather than emitting two pages with the same title.
        if out and out[-1][0] == date:
            out[-1][2].extend(fresh)
            out[-1][2].sort(key=lambda j: (j["flag"] != "", -j["score"]))
        else:
            out.append([date, score_col, fresh])
    return [tuple(x) for x in reversed(out)]             # newest first


def _desc_blocks(text, limit=6000):
    """Description as paragraph blocks, bullet lines preserved, length-capped.

    6,000 chars covers the full text of ~90% of postings; beyond that a JD is
    almost entirely boilerplate (benefits, EEO statements, company history) and
    the link is one click away.
    """
    text = (text or "").strip()
    if not text:
        return [paragraph("_No description captured._")]
    truncated = len(text) > limit
    text = text[:limit]
    out = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("•"):
            out.append(_list_item("bulleted_list_item", line.lstrip("• ").strip()))
        else:
            out.append(paragraph(line))
        if len(out) >= 90:        # keep under Notion's 100-children cap
            truncated = True
            break
    if truncated:
        out.append(paragraph("… truncated — open the posting for the full text."))
    return out


def jobs_date_page_blocks(date, threshold, jobs, score_col="FitScore"):
    """One date page: the ranked table, then a collapsed description per job."""
    blocks = jobs_page_blocks(date, threshold, jobs, score_col)
    blocks.append(heading(2, "Job descriptions"))
    blocks.append(quote("One toggle per job, same order as the table above. "
                        "Collapsed so the page stays scannable."))
    for j in jobs:
        head = f"{j['flag'] + ' ' if j['flag'] else ''}{j['score']} · {j['title']} — {j['company']}"
        kids = []
        if j["link"]:
            kids.append(paragraph_rt([_txt("Open posting ↗", link=j["link"])]))
        kids.append(paragraph(f"{j['location']} · {j['salary']} · CV tools {j['skill']}"))
        kids += _desc_blocks(j["desc"])
        blocks.append(toggle(head, kids[:100]))
    return blocks


def jobs_index_blocks(by_date, threshold):
    """Parent 'Jobs to Apply' page: a summary; the date pages are its children."""
    total = sum(len(js) for _d, _s, js in by_date)
    out = [paragraph(
        f"{total} distinct postings scoring ≥ {threshold}, across the last "
        f"{len(by_date)} scrape days. Each date page below holds the jobs that "
        f"FIRST appeared that day — a posting is never repeated across pages, so "
        f"working top-down covers everything exactly once.")]
    rows = [["Date", "New jobs"]] + [[d, str(len(js))] for d, _s, js in by_date]
    out.append(table(rows))
    return out


def _publish_date_pages(client, jobs_page_id, by_date):
    """Create/refresh one child page per scrape date under the Jobs page.

    Date pages are matched on a title PREFIX (the date) so the job count in the
    title can change between runs without orphaning the page — same in-place
    reuse the top-level sub-pages get, which is what preserves their URLs.
    """
    existing = {}
    for b in get_all_children(client, jobs_page_id):
        if b["type"] == "child_page":
            t = b["child_page"]["title"]
            m = re.match(r"(\d{4}-\d{2}-\d{2})", t)
            if m:
                existing[m.group(1)] = b
    wanted = {d for d, _s, _j in by_date}
    for date, score_col, jobs in by_date:
        title = f"{date} · {len(jobs)} jobs"
        blocks = jobs_date_page_blocks(date, JOBS_SCORE_THRESHOLD, jobs, score_col)
        try:
            _id, how = upsert_subpage(client, jobs_page_id, existing.get(date), title, blocks)
            print(f"      {title} {how}")
        except Exception as e:                            # noqa: BLE001
            print(f"      [!] {title} FAILED: {e}")

    # Prune date pages that have fallen out of the window, or they accumulate
    # one per day forever. These are a working shortlist, not an archive — the
    # postings behind them are closed, and the CSVs in jobs_output/ remain the
    # real history.
    stale = [d for d in existing if d not in wanted]
    for d in stale:
        try:
            api(client, "DELETE", f"/blocks/{existing[d]['id']}")
            time.sleep(PACE)
            print(f"      {d} pruned (outside the {JOBS_DAYS}-day window)")
        except Exception as e:                            # noqa: BLE001
            print(f"      [!] pruning {d} failed: {e}")


# ── Notion API ────────────────────────────────────────────────────────────────
def api(client, method, path, **kw):
    """Notion request with retry on 429 and on transient transport failures.

    **Retry safety is method-dependent, deliberately:**

    * A **connect** failure means the request never reached Notion, so any
      method is safe to retry.
    * A **read timeout** means the request may well have been processed and only
      the response was lost. Retrying a GET is harmless; retrying a PATCH that
      appended 90 blocks would append them **twice**. So read timeouts are
      retried only for GET, and surface loudly for writes.

    Added 2026-08-11 after the nightly run died on `httpx.ReadTimeout` inside
    `get_all_children` — a plain network blip with no retry path, the same class
    of failure that once killed the whole pipeline via the CV download.
    """
    last = None
    for attempt in range(6):
        try:
            r = client.request(method, f"{API}{path}", **kw)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            last = e                                   # never reached the server
        except httpx.TimeoutException as e:
            if method.upper() != "GET":
                raise RuntimeError(
                    f"{method} {path}: read timed out — NOT retried, because the "
                    f"write may have been applied and retrying could duplicate it"
                ) from e
            last = e
        else:
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", "1")))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
            return r.json() if r.text else {}
        # full-jitter capped backoff, matching the scraper's _get_with_retry
        cap = min(30, 2 ** attempt * 2)
        time.sleep(random.uniform(cap / 2, cap))
        print(f"      [~] {method} {path} transient failure ({type(last).__name__}), "
              f"retry {attempt + 1}/6")
    raise RuntimeError(f"{method} {path}: retries exhausted ({type(last).__name__ if last else 'rate limited'})")


def get_all_children(client, block_id):
    out, cur = [], None
    while True:
        q = "?page_size=100" + (f"&start_cursor={cur}" if cur else "")
        j = api(client, "GET", f"/blocks/{block_id}/children{q}")
        out += j["results"]
        if j.get("has_more"):
            cur = j["next_cursor"]; time.sleep(PACE)
        else:
            return out


def delete_all_children(client, block_id, keep_child_pages=False):
    """Clear a block's children.

    `keep_child_pages` preserves nested pages. Without it, refreshing the
    "🎯 Jobs to Apply" page deleted its own date sub-pages, so _publish_date_pages
    found nothing to reuse and recreated all six every run — churning their URLs
    (breaking bookmarks) and resetting the full-width setting each time. The
    symptom was every date page logging "created" instead of "updated in place".
    """
    kids = get_all_children(client, block_id)
    if keep_child_pages:
        kids = [b for b in kids if b["type"] != "child_page"]
    if not kids:
        return 0
    # Deletes are independent, so fan them out instead of one-request-then-sleep;
    # api()'s 429 retry (Retry-After) keeps aggregate throughput within Notion's limit.
    with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as pool:
        list(pool.map(lambda b: api(client, "DELETE", f"/blocks/{b['id']}"), kids))
    return len(kids)


def append_children(client, block_id, blocks):
    """Append in chunks, counting NESTED children toward the budget.

    Notion caps a request at 100 top-level children, but nested blocks (a toggle
    holding 40 description paragraphs) count toward the request's overall size
    too. A flat chunk of 100 toggles was ~1,300 blocks in one call. Chunking on
    the *total* block count keeps every request well inside the limit.
    """
    BUDGET = 90
    chunk, used = [], 0
    def _flush():
        nonlocal chunk, used
        if not chunk:
            return
        api(client, "PATCH", f"/blocks/{block_id}/children", json={"children": chunk})
        time.sleep(PACE)
        chunk, used = [], 0
    for b in blocks:
        cost = 1 + len(b.get(b["type"], {}).get("children", []) or [])
        if chunk and used + cost > BUDGET:
            _flush()
        chunk.append(b)
        used += cost
    _flush()


def create_subpage(client, parent_id, title, blocks):
    j = api(client, "POST", "/pages", json={
        "parent": {"page_id": parent_id},
        "properties": {"title": [{"type": "text", "text": {"content": title}}]},
    })
    sub_id = j["id"]
    append_children(client, sub_id, blocks)
    return sub_id


def update_page_title(client, page_id, title):
    api(client, "PATCH", f"/pages/{page_id}",
        json={"properties": {"title": [{"type": "text", "text": {"content": title}}]}})


def upsert_subpage(client, parent_id, existing_block, title, blocks, keep_child_pages=False):
    """Refresh a sub-page IN PLACE if it already exists (a child_page block whose
    id IS the page id), else create it. Reusing the page keeps its URL stable and
    preserves UI-only settings the API can't set — full-width layout and any
    'Publish to web' link the user enabled — across re-runs.
    """
    if existing_block is None:
        return create_subpage(client, parent_id, title, blocks), "created"
    pid = existing_block["id"]
    if existing_block["child_page"]["title"] != title:
        update_page_title(client, pid, title)
    delete_all_children(client, pid, keep_child_pages)  # clear content, keep the page
    append_children(client, pid, blocks)
    return pid, "updated in place"


# ── markdown endpoint (hybrid path, added 2026-08-12) ────────────────────────
# PATCH /v1/pages/{id}/markdown replaces a whole page from markdown text in ONE
# call, instead of ~40 chunked block appends. Verified by
# scratch/scratch_notion_markdown_spike.py: headings, tables, nested lists,
# quotes, dividers and links all survive, and the `[Name](Optional)` string that
# once 400'd the block path is handled safely.
#
# ⚠️ TOGGLES DO NOT SURVIVE. `<details>` is not parsed — it renders as VISIBLE
# raw HTML ("<details><summary>...</summary>" as literal body text). So:
#   * main page + analysis sub-page  -> markdown endpoint (prose and tables)
#   * Jobs date pages + Daily Stats  -> keep the block builder, because a toggle
#     per job IS the product there; without it the page is 120 full JDs in a wall.
# `_details_to_markdown` strips the HTML for the markdown path so nothing leaks.
_DETAILS_RE = re.compile(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", re.S | re.I)


def _details_to_markdown(md):
    """Flatten <details> into a bold lead-in plus its list items.

    The collapse is lost — that is the accepted trade for this path — but the
    CONTENT must not be, and the raw tags must never reach the page.
    """
    def repl(m):
        summary = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        items = re.findall(r"<li>(.*?)</li>", m.group(2), re.S | re.I)
        body = "\n".join(
            "- " + re.sub(r"<code>(.*?)</code>", r"`\1`", re.sub(r"<[^>]+>", "", i)).strip()
            for i in items if re.sub(r"<[^>]+>", "", i).strip()
        )
        return f"**{summary}**\n\n{body}\n"
    md = _DETAILS_RE.sub(repl, md)
    # Belt and braces: any stray tag left over would render as literal text.
    return re.sub(r"</?(details|summary|ul|li)>", "", md, flags=re.I)


# 🚫 UNUSED — kept only as the record of a rejected approach. DO NOT RE-ENABLE
# without re-reading this. `PATCH /pages/{id}/markdown` was trialled on
# 2026-08-12 and reverted the same day. Three independent blockers, found in
# order, each after the previous was "fixed":
#
#   1. It REFUSES on any page with child pages — 400 "would delete N child
#      page(s)" — so the main page can never use it. `allow_deleting_content`
#      would destroy the nav cards and every Jobs date page.
#   2. `<details>` is not parsed; it renders as VISIBLE raw HTML.
#   3. And the one the spike missed: **markdown nested inside a blockquote is
#      not parsed.** This plan puts tables and headings inside `>` callouts, and
#      they came out as literal `| Table | Answers |` and `### Heading` text.
#      HTML comments (`<!-- ROI_TABLE_START -->`) leak as visible text too — the
#      block path strips those via _is_comment, this path does not.
#
# The spike tested constructs in isolation and passed. The document uses them
# NESTED, which is where it fails. Test with the real document, not a sample.
def replace_page_markdown(client, page_id, md, label=""):
    """Replace a page's whole body from markdown in one request.

    `allow_deleting_content` is deliberately NOT set: Notion then refuses to
    delete nested child pages, which is exactly the protection we hand-rolled as
    keep_child_pages after the date sub-pages were wiped. The nav cards survive.
    """
    md = _details_to_markdown(md)
    resp = api(client, "PATCH", f"/pages/{page_id}/markdown",
               json={"type": "replace_content", "replace_content": {"new_str": md}})
    if resp.get("truncated"):
        print(f"  [!] '{label}' TRUNCATED by Notion — content was cut. "
              f"Falling back is safer than shipping a half page.")
        return False
    unknown = resp.get("unknown_block_ids") or []
    print(f"  '{label}' written via markdown endpoint (1 call"
          + (f", {len(unknown)} unknown blocks" if unknown else "") + ")")
    return True


# ── orchestration ─────────────────────────────────────────────────────────────
def _split_analysis(lines):
    """Return (main_lines, analysis_lines).

    Moves each (start, resume) range in ANALYSIS_SECTIONS off the main page.
    A range whose START heading is missing is skipped with a warning rather
    than guessed — a wrong cut would silently drop part of the plan from the
    published page, which is far worse than publishing one section too many.
    """
    def _find(prefix, frm=0):
        for i in range(frm, len(lines)):
            if lines[i].startswith(prefix):
                return i
        return None

    ranges = []
    for start_h, resume_h in ANALYSIS_SECTIONS:
        a = _find(start_h)
        if a is None:
            print(f"  [!] section not found, left on main page: {start_h}")
            continue
        b = len(lines) if resume_h is None else _find(resume_h, a + 1)
        if b is None:
            print(f"  [!] resume heading not found for {start_h} — section skipped")
            continue
        ranges.append((a, b))

    if not ranges:
        return lines, []

    ranges.sort()
    analysis, main_lines, prev = [], [], 0
    for a, b in ranges:
        chunk = lines[prev:a]
        # Drop a '---' / blank tail left dangling above a removed section.
        while chunk and chunk[-1].strip() in ("", "---"):
            chunk.pop()
        if chunk:
            chunk.append("\n")
        main_lines += chunk
        analysis += lines[a:b]
        prev = b
    main_lines += lines[prev:]
    return main_lines, analysis


def _rt_text(block):
    """Flatten a block's rich_text back to plain text."""
    body = block.get(block["type"], {})
    return "".join(x.get("text", {}).get("content", "") for x in body.get("rich_text", []))


def _collapse_sections(blocks, prefixes=COLLAPSE_HEADINGS):
    """Wrap the body of matching heading_3 sections in a collapsed toggle.

    A section runs until the next heading of ANY level. Notion accepts at most
    100 children in one create call, so an over-long section is left expanded
    rather than sent and rejected — the same failure that emptied the main page
    on 2026-08-10.
    """
    HEADINGS = {"heading_1", "heading_2", "heading_3"}
    out, i = [], 0
    while i < len(blocks):
        b = blocks[i]
        if b["type"] == "heading_3" and _rt_text(b).startswith(prefixes):
            j = i + 1
            kids = []
            while j < len(blocks) and blocks[j]["type"] not in HEADINGS:
                kids.append(blocks[j])
                j += 1
            if kids and len(kids) <= 100:
                out.append(toggle(_rt_text(b), kids))
                i = j
                continue
        out.append(b)
        i += 1
    return out


def _digest_markdown(analysis_lines):
    """The Req% digest as markdown, for the endpoint path.

    Same source as _req_digest() — parsed back out of the NOW market table so it
    cannot drift — just rendered as text instead of block objects.
    """
    blocks = _req_digest(analysis_lines)
    if not blocks:
        return ""
    rows = blocks[-1]["table"]["children"]
    def cells(r):
        return ["".join(x["text"]["content"] for x in c) for c in r["table_row"]["cells"]]
    head, *body = [cells(r) for r in rows]
    out = ["## 📊 What actually gates an application", "",
           "> " + _rt_text(blocks[1]), "",
           "| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out) + "\n"


def _req_digest(analysis_lines):
    """Compact 'what actually gates an application' table for the main page,
    parsed back out of the NOW market table so it cannot drift from the source."""
    rows, in_now, header = [], False, None
    for ln in analysis_lines:
        if ln.startswith("#### 🎯 NOW"):
            in_now = True
            continue
        if in_now and ln.startswith("####"):
            break
        if in_now and ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if header is None:
                header = cells
                continue
            if set("".join(cells)) <= set("-: "):
                continue
            if "**Req%**" in header:
                k = header.index("**Req%**")
                if k < len(cells):
                    rows.append((cells[0].strip("*"), cells[k]))
    if not rows:
        return []
    def _num(v):
        m = re.search(r"(\d+)", v)
        return int(m.group(1)) if m else -1
    rows.sort(key=lambda r: _num(r[1]), reverse=True)
    out = [heading(2, "📊 What actually gates an application"),
           quote("Top of the NOW track by **Req%** — the share of postings where the "
                 "sentence around the mention reads as a requirement, not one of several "
                 "alternatives. Full tables, trends, the Senior→Staff gap and the salary "
                 "analysis are in the 📊 Market Data & Analysis sub-page."),
           table([["Skill", "Required in"]] + [[n, v] for n, v in rows[:6]])]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="parse the markdown and report block counts; no API calls")
    args = ap.parse_args()

    lines = open(MARKDOWN_PATH, encoding="utf-8").readlines()

    start = end = None
    for idx, line in enumerate(lines):
        if "DAILY_STATS_TABLE_START" in line:
            start = idx
        elif "DAILY_STATS_TABLE_END" in line:
            end = idx
    if start is not None and end is not None:
        body_lines = lines[:start] + lines[end + 1:]
        daily_lines = lines[start + 1:end]
    else:
        body_lines, daily_lines = lines, []

    # The daily stats now live in a sub-page (rendered as a nav card at the top),
    # so drop the now-empty "## 📈 DAILY PIPELINE STATS" heading + its divider
    # from the body.
    for cut in range(len(body_lines) - 1, -1, -1):
        if "DAILY PIPELINE STATS" in body_lines[cut] and body_lines[cut].lstrip().startswith("#"):
            body_lines = body_lines[:cut]
            while body_lines and body_lines[-1].strip() in ("", "---"):
                body_lines.pop()
            break

    body_lines, analysis_lines = _split_analysis(body_lines)
    # Markdown text for the endpoint path, blocks kept as the fallback.
    body_md = "".join(body_lines)
    analysis_md = "".join(analysis_lines)
    body_blocks = _collapse_sections(parse_blocks(body_lines))
    if analysis_lines:
        body_blocks += _req_digest(analysis_lines)
    analysis_blocks = parse_blocks(analysis_lines) if analysis_lines else []
    # The page's own title already shows the H1, so drop a leading duplicate H1.
    if body_blocks and body_blocks[0]["type"] == "heading_1":
        body_blocks = body_blocks[1:]
    daily_toggles = parse_daily_toggles(daily_lines)
    by_date = load_apply_jobs_by_date(JOBS_SCORE_THRESHOLD, JOBS_CAP, JOBS_DAYS)
    n_jobs = sum(len(js) for _d, _s, js in by_date)
    jobs_title = f"{JOBS_SUBPAGE_TITLE} ({n_jobs} · ≥{JOBS_SCORE_THRESHOLD})"

    print(f"Parsed: {len(body_blocks)} body blocks | {len(analysis_blocks)} analysis blocks "
          f"| {len(daily_toggles)} daily toggles "
          f"| {n_jobs} apply-jobs >={JOBS_SCORE_THRESHOLD} across {len(by_date)} date page(s)")
    for _d, _s, _js in by_date:
        print(f"    {_d}: {len(_js)} new")
    print("Body block types:", dict(Counter(b["type"] for b in body_blocks)))
    if args.dry_run:
        print("(dry run — no API calls made)")
        return

    tok = load_token()
    headers = {"Authorization": f"Bearer {tok}", "Notion-Version": NOTION_VERSION,
               "Content-Type": "application/json"}
    # Granular timeouts: appending ~90 blocks legitimately takes longer than a
    # read, and a single flat 30s was tight enough to trip on a slow night.
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
    with httpx.Client(headers=headers, timeout=timeout) as client:
        print(f"Target page: {PAGE_ID}")
        children = get_all_children(client, PAGE_ID)

        # Locate existing sub-pages by title prefix so we can reuse them in place
        # (preserving their URL, full-width setting, and any public link).
        existing = {"daily": None, "jobs": None, "analysis": None}
        for b in children:
            if b["type"] == "child_page":
                t = b["child_page"]["title"]
                if t.startswith("📅 Daily Pipeline Stats"):
                    existing["daily"] = b
                elif t.startswith("🎯 Jobs to Apply"):
                    existing["jobs"] = b
                elif t.startswith("📊 Market Data"):
                    existing["analysis"] = b
        keep = {b["id"] for b in existing.values() if b}

        # Replace only the body; the sub-page cards are preserved (they shift to
        # the top of the page and act as a persistent nav / page tree).
        # Replace only the body; sub-page cards are preserved.
        body_kids = [b for b in children if b["id"] not in keep]
        for b in body_kids:
            try:
                api(client, "DELETE", f"/blocks/{b['id']}")
            except RuntimeError as e:
                # A DELETE that timed out is NOT retried (it may have applied).
                # Tolerate it: the block is either gone or will be orphaned, and
                # aborting here is what emptied the page on 2026-08-12.
                print(f"  [~] skipped a block delete: {str(e)[:80]}")
            time.sleep(PACE)
        print(f"  archived {len(body_kids)} body block(s); kept {len(keep)} sub-page(s)")

        # Each sub-page write is isolated. The body is archived BEFORE this
        # point, so an exception escaping here leaves the main page EMPTY —
        # which is exactly what happened on 2026-08-10 when a 159-span quote
        # tripped Notion's 100-item rich_text cap: the analysis sub-page was
        # created, appending its children 400'd, and the body was never
        # rewritten. A failed sub-page must never cost the main page.
        failures = []
        for cond, key, title, blocks, label in (
            (daily_toggles, "daily", DAILY_SUBPAGE_TITLE, daily_toggles,
             f"{len(daily_toggles)} dates"),
            (by_date, "jobs", jobs_title,
             jobs_index_blocks(by_date, JOBS_SCORE_THRESHOLD) if by_date else [],
             f"{n_jobs} jobs / {len(by_date)} dates"),
            (analysis_blocks, "analysis", ANALYSIS_SUBPAGE_TITLE, analysis_blocks,
             f"{len(analysis_blocks)} blocks"),
        ):
            # Analysis page is prose + tables with no load-bearing toggles, so it
            # takes the one-call markdown path; everything else keeps blocks.
            if not cond:
                continue
            try:
                _id, how = upsert_subpage(client, PAGE_ID, existing[key], title, blocks,
                                          keep_child_pages=(key == "jobs"))
                print(f"  '{title}' ({label}) {how}: {_id}")
                if key == "jobs":
                    _publish_date_pages(client, _id, by_date)
            except Exception as e:                      # noqa: BLE001
                failures.append(title)
                print(f"  [!] '{title}' FAILED: {e}")
                print("      continuing so the main page body is still written.")

        # Main page body via markdown. `allow_deleting_content` is NOT set, so
        # Notion refuses to remove the three nav sub-pages — the same protection
        # keep_child_pages gives on the block path.
        # ⚠️ The MAIN page CANNOT use the markdown endpoint. `replace_content`
        # hard-refuses when a page has child pages:
        #   400 "This operation would delete 5 child page(s) or database(s)"
        # and the only way past it is `allow_deleting_content: true`, which would
        # destroy the three nav sub-pages plus every Jobs date page. So the main
        # page stays on the block builder. Only the analysis sub-page — which has
        # no children — takes the one-call markdown path.
        append_children(client, PAGE_ID, body_blocks)
        print(f"  wrote {len(body_blocks)} body block(s)")
        if failures:
            print(f"  [!] {len(failures)} sub-page(s) failed: {', '.join(failures)}")
    print("✅ Notion page updated (sub-pages preserved in place).")


if __name__ == "__main__":
    main()
