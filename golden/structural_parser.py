"""Structural parser for editorial board markdown files.

Extracts Editor(name, role, affiliations) from recognizable publisher patterns
without LLM calls.  Falls back to the LLM extraction agent when a format is
unrecognized.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from models.journal import Editor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEGREE_PATTERN = re.compile(
    r'(?<!\w)(?:PhD|MD|Dr\.?|Prof\.?|Professor|BSc|MSc|DSc|Dr\. med\.?|Dr\. rer\. nat|'
    r'Dr\. habil|Dr\. Sci\.|FDS RCPs|FICD|FADM|FIMMM|FCG Dent|FHEA|'
    r'Habilitation|Dr\. Habil|Full professor|B\.Ch\.D\.|MBBS|FFICM|SFHEA)(?!\w)|'
    r'(?<=, )MA(?!\w)',
    re.IGNORECASE,
)

# Section headings that are never person names.  Used to filter out
# metadata sections (biography, expertise, etc.) that get split as
# individual sections by split_sections.
_HEADING_NOT_A_PERSON = frozenset({
    "subject areas",
    "expertise",
    "biography",
    "specialisms",
    "research website",
    "orcid profile",
    "editorial board",
    "editorial board by country/region",
    "gender diversity of editors and editorial board members",
    "editorial advisory board",
    "editors-in-chief",
    "assisted by",
    "contact",
    "email",
})


def _is_section_heading(name: str) -> bool:
    """Return True if *name* looks like a section heading rather than a person."""
    n = name.strip().strip("#").strip().lower()
    if n in _HEADING_NOT_A_PERSON:
        return True
    # Heuristic: if it contains no lowercase letter after stripping, it's
    # likely an abbreviation or code (e.g. "COSSEE", "USA", "ITA")
    if not re.search(r"[a-z]", n):
        return True
    return False


def strip_degrees(name: str) -> str:
    """Remove academic degrees and titles from editor name."""
    name = DEGREE_PATTERN.sub("", name)
    name = re.sub(r",\s*,", ",", name)  # collapse double commas
    name = re.sub(r"\s*,\s*$", "", name)  # trailing comma
    name = re.sub(r"^\s*,\s*", "", name)  # leading comma
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Heading-level detection + section splitting
# ---------------------------------------------------------------------------

def _heading_counts(text: str) -> dict[int, int]:
    """Return {level: count} for # / ## / ### headings."""
    pattern = re.compile(r"^(#{1,3})\s+")
    counts = {1: 0, 2: 0, 3: 0}
    for line in text.split("\n"):
        m = pattern.match(line)
        if m:
            counts[len(m.group(1))] += 1
    return counts


def _split_on(text: str, level: int) -> list[tuple[str, str]]:
    """Split *text* on headings of *level*, returning [(heading, body), ...].

    The body includes everything up to the next heading of the same level
    (exclusive).  Pre-heading content is returned as section "intro".
    """
    title_pattern = re.compile(r"^#\s+(.+)$", re.MULTILINE)
    pattern = rf"^{'#' * level}\s+"
    parts = re.split(pattern, text, flags=re.MULTILINE)

    if len(parts) < 2:
        title_m = title_pattern.search(text)
        heading = title_m.group(1).strip() if title_m else "editors"
        return [(heading, text)]

    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        title_m = title_pattern.search(parts[0])
        heading = title_m.group(1).strip() if title_m else "intro"
        sections.append((heading, parts[0]))

    for i in range(1, len(parts)):
        heading = parts[i].split("\n")[0].strip()
        sections.append((heading, parts[i]))

    return sections


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections.

    Detects the file's heading level automatically:
    - Pick the heading level with count >= 2 and highest count.
    - If no level qualifies (count < 2), the entire text is one section.
    """
    title_pattern = re.compile(r"^#\s+(.+)$", re.MULTILINE)
    counts = _heading_counts(text)

    # Find best level: count >= 2, highest count wins
    candidates = {lvl: c for lvl, c in counts.items() if c >= 2}
    if not candidates:
        # No structural headings — entire text is one section
        title_m = title_pattern.search(text)
        heading = title_m.group(1).strip() if title_m else "editors"
        return [(heading, text)]

    level = max(candidates, key=candidates.get)
    return _split_on(text, level)


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _extract_springer_inline(body: str) -> list[Editor]:
    """Format A: ### Role: Name, PhD; Affiliation, Country

    Source: springer_nature/srep (In-house Editors section)
    """
    pattern = re.compile(
        r"^###\s+(?P<role>.+?):\s+(?P<name>.+?),\s+[^;]+;\s+(?P<aff>.+?)\s*,\s*(?P<country>.+)$",
        re.MULTILINE,
    )
    editors: list[Editor] = []
    for m in pattern.finditer(body):
        editors.append(
            Editor(
                name=strip_degrees(m.group("name")),
                role=m.group("role").strip(),
                affiliations=[f"{m.group('aff').strip()}, {m.group('country').strip()}"],
            )
        )
    return editors


def _extract_springer_single(body: str) -> list[Editor]:
    """Format B: **Name**, PhD, Affiliation, Country - Subject
    
    Two-pass: extract name from **...** markers, then parse remainder loosely.
    Handles missing degrees, unrecognized degrees, mid-line **, and optional subject.
    
    Source: springer_nature/srep (Senior Editorial Board section)
    """
    bold_name_pattern = re.compile(r"\*\*(?P<name>.+?)\*\*")
    editors: list[Editor] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Extract name from **...** markers (handles start-of-line and mid-line **)
        bold_m = bold_name_pattern.search(line)
        if not bold_m:
            continue

        name = bold_m.group("name").strip().rstrip(",").strip()
        after = line[bold_m.end():].strip().lstrip(",").strip()

        # Extract optional subject (last " - Subject" separator)
        subject = None
        if " - " in after:
            idx = after.rfind(" - ")
            subject = after[idx + 3:].strip()
            after = after[:idx].strip()

        # Strip degrees from the affiliation part to avoid "PhD, University..."
        after = strip_degrees(after)

        # Split by comma; last token is country
        parts = [p.strip() for p in after.split(",") if p.strip()]
        if not parts:
            continue

        country = parts[-1].strip()
        rest = ", ".join(parts[:-1])
        aff_str = f"{rest}, {country}" if rest else country

        editors.append(
            Editor(
                name=name,
                role=subject or "Senior Editorial Board Member",
                affiliations=[aff_str],
            )
        )
    return editors


def _extract_bold_two_line(
    body: str, section_heading: str | None = None
) -> list[Editor]:
    """Format C: **Name**, PhD on one line, Affiliation on next

    Source: springer_nature/srep (Editorial Board Members, subject-area sections)
    """
    pattern = re.compile(
        r"^\*\*(?P<name>.+?)(?:,\s*)?(?:PhD|Dr\.?|MD|Professor|Prof\.?|BSc|MSc|Dr\. med\.?)?\*\*\s*$"
    )
    editors: list[Editor] = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        m = pattern.match(lines[i])
        if m:
            name = strip_degrees(m.group("name"))
            aff = None
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].strip():
                    aff = lines[j].strip()
                    i = j + 1
                    break
            if aff:
                role = (
                    section_heading
                    if section_heading
                    and section_heading not in ("Editorial Board Members", "editors")
                    else None
                )
                editors.append(Editor(name=name, role=role, affiliations=[aff]))
        i += 1
    return editors


def _extract_npj_bold_italic(body: str) -> list[Editor]:
    """Format D: **Name**, PhD, *Affiliation, Country*

    Source: springer_nature/npjclimataction
    """
    role_pattern = re.compile(r"^#{1,3}\s+(?P<role>.+?):\s*$")
    editor_pattern = re.compile(
        r"^\*\*(?P<name>.+?)\*\*,?\s*(?:PhD|Dr\.?|MD|Professor)?\s*,?\s*\*(?P<aff>.+?)\*$"
    )
    editors: list[Editor] = []
    current_role: str | None = None

    for line in body.split("\n"):
        role_m = role_pattern.match(line)
        if role_m:
            current_role = role_m.group("role").strip()
            continue

        m = editor_pattern.match(line)
        if m:
            name = strip_degrees(m.group("name"))
            affiliation = m.group("aff").strip().strip("* ,")
            editors.append(
                Editor(name=name, role=current_role, affiliations=[affiliation])
            )
    return editors


def _extract_nmeth_inline(heading: str, body: str) -> list[Editor]:
    """Format K: ## Role: Name, PhD, Affiliation, Country ORCiD
    
    The heading itself IS the editor entry.
    
    Source: springer_nature/nmeth
    """
    pattern = re.compile(
        r"^(?:#{1,3}\s+)?(?P<role>.+?):\s+(?P<name>.+?),\s+(?:PhD|Dr\.?|MD|Professor|MA),\s+(?P<aff>.+?)\s*,\s*(?P<country>.+?)(?:\s+ORCiD)?$"
    )
    editors: list[Editor] = []
    # "Chief Editor: Allison Doerr, PhD, Springer Nature, United States of America ORCiD"
    m = pattern.match(heading)
    if m:
        editors.append(
            Editor(
                name=strip_degrees(m.group("name")),
                role=m.group("role").strip(),
                affiliations=[f"{m.group('aff').strip()}, {m.group('country').strip()}"],
            )
        )
    return editors


def _extract_acs_plain(body: str) -> list[Editor]:
    """Format E: Name / Affiliation / Country on separate lines, blank-separated.

    Each editor is a 4-line record: Name, Affiliation, Country, E-mail.
    Fields separated by single blank lines (\n\n).

    Source: acs/es-and-t, acs-nano, acs-energy-letters (partially)
    """
    editors: list[Editor] = []
    # Split on blank lines to get individual fields
    fields = [l.strip() for l in body.strip().split("\n\n") if l.strip()]
    # Group fields into 4-per-editor records: Name, Affiliation, Country, E-mail
    i = 0
    while i < len(fields):
        field = fields[i]
        # Skip headings, list items, and noise
        if field.startswith(("-", "#", "*")) or field.startswith("E-mail"):
            i += 1
            continue
        # Treat as potential name; look ahead for affiliation + country
        if i + 2 < len(fields) and not fields[i + 1].startswith(("-", "#", "*", "E-mail")):
            name = field
            if _is_section_heading(name):
                i += 1
                continue
            aff = fields[i + 1]
            country = fields[i + 2] if i + 2 < len(fields) else None
            aff_str = f"{aff}, {country}" if country else aff
            editors.append(Editor(name=strip_degrees(name), role=None, affiliations=[aff_str]))
            i += 4  # skip name, aff, country, email
        else:
            i += 1
    return editors


def _extract_sage_table(body: str) -> list[Editor]:
    """Format F: | Name | Affiliation, Country |

    Source: sage/sgo, sage/smsa, sage/smo
    """
    heading_pattern = re.compile(r"^#{1,3}\s")
    table_line_pattern = re.compile(r"^\s*\|")
    entry_pattern = re.compile(r"^\s*\|\s*(?P<name>.+?)\s*\|\s*(?P<aff>.+?)\s*\|")
    editors: list[Editor] = []
    current_role: str | None = None

    for line in body.split("\n"):
        if heading_pattern.match(line):
            continue
        if not table_line_pattern.match(line) and line.strip():
            current_role = line.strip()
            continue

        m = entry_pattern.match(line)
        if m:
            name = strip_degrees(m.group("name").strip())
            affiliation = m.group("aff").strip()
            editors.append(
                Editor(name=name, role=current_role, affiliations=[affiliation])
            )
    return editors


def _extract_frontiers_block(body: str) -> list[Editor]:
    """Format G: ### name → Affiliation → City, Country → Role
    
    Used for Copernicus sections that contain ### sub-headings
    (e.g. "Subject areas", "Expertise" — filtered by _is_section_heading).
    """
    name_pattern = re.compile(r"^###\s+(?P<name>.+)$")
    editors: list[Editor] = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        m = name_pattern.match(lines[i])
        if m:
            name = strip_degrees(m.group("name").strip())
            if _is_section_heading(name):
                i += 1
                continue
            
            aff_parts: list[str] = []
            role: str | None = None
            last_j = i
            
            for j in range(i + 1, min(i + 8, len(lines))):
                line = lines[j].strip()
                if not line:
                    continue
                if line.startswith("###"):
                    break
                
                # If line looks like a role, stop collecting affiliation and set role
                if any(r in line.lower() for r in ["chief editor", "section editor", "specialty", "associate", "deputy"]):
                    role = line
                    last_j = j
                    break
                
                # Otherwise, collect as affiliation
                aff_parts.append(line)
                last_j = j
            
            if aff_parts:
                editors.append(Editor(name=name, role=role, affiliations=[", ".join(aff_parts)]))
            i = last_j
        i += 1
    return editors


def _extract_heading_name(heading: str, body: str) -> list[Editor]:
    """Format Q: Name is the section heading, affiliation is in the body.

    Used when `split_sections` splits on `### Name` headings, making the name
    the section heading and the body just the affiliation text.

    Source: springer_link (bmc-biology, epj-data-science, j-solid-state-electrochem)
    """
    editors: list[Editor] = []
    name = strip_degrees(heading.strip())
    if not name or len(name) < 2:
        return editors
    if _is_section_heading(name):
        return editors

    # Extract affiliation from body - first non-empty non-heading line.
    # Skip the first line if it's the heading repeated (split_on includes
    # the heading text as the first line of the body part).
    heading_stripped = strip_degrees(heading.strip()).lower()
    aff = None
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith(("#", "-", "*")):
            continue
        # Skip the heading repeated as first body line
        if strip_degrees(line).lower() == heading_stripped:
            continue
        aff = line
        break

    editors.append(Editor(name=name, role=None, affiliations=[aff] if aff else []))
    return editors


def _extract_elsevier_bold_section(body: str) -> list[Editor]:
    """Format L: ### Editor-in-Chief / #### Name, PhD → Affiliation, Country

    Source: elsevier/information-sciences, elsevier/alexandria-engineering-journal
    """
    role_pattern = re.compile(r"^###\s+(?P<role>.+)$")
    name_pattern = re.compile(r"^####\s+(?P<name>.+)$")
    editors: list[Editor] = []
    current_role: str | None = None
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        # Detect role from ### heading
        hm = role_pattern.match(lines[i])
        if hm and "editor" in hm.group("role").lower():
            current_role = hm.group("role").strip()
            i += 1
            continue

        # Detect #### Name, optional degrees
        nm = name_pattern.match(lines[i])
        if nm:
            raw = nm.group("name").strip().rstrip("*")
            if "editor" in raw.lower():
                i += 1
                continue
            name = strip_degrees(raw)
            if _is_section_heading(name):
                i += 1
                continue
            aff = None
            for j in range(i + 1, min(i + 6, len(lines))):
                line = lines[j].strip()
                if line.startswith("#"):
                    break
                if not line:
                    continue
                if not aff:
                    aff = line
            if aff:
                editors.append(
                    Editor(name=name, role=current_role, affiliations=[aff])
                )
            i += 1
            continue
        i += 1
    return editors


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(body: str) -> str:
    """Return format label based on heuristics of first 500 chars.

    Order matters: more specific patterns first to avoid false positives.
    """
    # Pre-compile heuristics for this call
    det_springer_inline = re.compile(r"^###\s+\w+.*?:\s+\w+.*?;\s+\w+", re.MULTILINE)
    det_springer_single = re.compile(r"^\*\*\w.+?\*\*,\s*(?:PhD|Dr|MD|Professor).+?\s*-\s*\w", re.MULTILINE)
    det_npj_bold_italic = re.compile(r"^\*\*\w.+?\*\*,\s*(?:PhD|Dr|MD|Professor)?\s*,?\s*\*", re.MULTILINE)
    det_bold_two_line = re.compile(r"^\*\*\w.+?\*\*\s*$", re.MULTILINE)
    det_sage_table = re.compile(r"^\|", re.MULTILINE)
    det_frontiers_block = re.compile(r"^###\s+\w", re.MULTILINE)
    det_elsevier_bold = re.compile(r"^####\s+\w", re.MULTILINE)

    sample = body[:500]

    # Springer inline: ### Role: Name, PhD; Affiliation, Country
    if det_springer_inline.search(sample):
        return "springer_inline"
    # Springer single-line: **Name**, PhD, Affiliation, Country - Subject
    if det_springer_single.search(sample):
        return "springer_single"
    # npj bold-italic: **Name**, PhD, *Affiliation, Country*
    if det_npj_bold_italic.search(sample):
        return "npj_bold_italic"
    # Bold two-line: **Name**, PhD (no - or * on same line)
    if det_bold_two_line.search(sample) and not re.search(r"\|", sample):
        return "bold_two_line"
    # Sage table: | Name | Affiliation |
    if det_sage_table.search(sample):
        return "sage_table"
    # Elsevier #### Name heading
    if det_elsevier_bold.search(sample):
        # Elsevier has #### Name followed by affiliation on next non-blank line
        lines = sample.split("\n")
        for line in lines:
            if det_elsevier_bold.match(line):
                idx = lines.index(line)
                for k in range(idx + 1, min(idx + 4, len(lines))):
                    if lines[k].strip():
                        if not lines[k].strip().startswith("#"):
                            return "elsevier_bold_section"
                        break
    # Frontiers block: ### name (no colon on first 200 chars)
    if det_frontiers_block.search(sample) and ":" not in sample[:200]:
        return "frontiers_block"
    # ACS plain: no ** or | markers
    if "**" not in sample and "|" not in sample:
        return "acs_plain"
    return "unknown"


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def parse_editors(text: str) -> list[Editor]:
    """Parse editors from markdown text using format-specific extractors.

    Returns an empty list if no format could be detected (caller should
    fall back to LLM extraction).
    """
    # Detection pattern for nmeth-style inline headings
    nmeth_inline_re = re.compile(
        r"^(.+?):\s+(.+?),\s+(?:PhD|Dr\.?|MD|Professor|MA),\s+\w+"
    )
    sections = split_sections(text)
    all_editors: list[Editor] = []

    # Check for nmeth-style inline headings before processing sections
    has_nmeth_inline = any(nmeth_inline_re.search(h) for h, _ in sections)
    if has_nmeth_inline:
        # nmeth: each ## heading IS an editor entry
        for heading, body in sections:
            editors = _extract_nmeth_inline(heading, body)
            all_editors.extend(editors)
        return all_editors

    # Check for heading-name format: most section headings are person names
    # (contain degrees, multi-word, not metadata headings).
    # Source: springer_link (bmc-biology, epj-data-science, j-solid-state-electrochem)
    person_headings = [
        h for h, _ in sections
        if not _is_section_heading(h) and DEGREE_PATTERN.search(h)
    ]
    if len(person_headings) > 1 and len(person_headings) / len(sections) >= 0.5:
        for heading, body in sections:
            editors = _extract_heading_name(heading, body)
            all_editors.extend(editors)
    else:
        for heading, body in sections:
            fmt = detect_format(body)

            if fmt == "springer_inline":
                all_editors.extend(_extract_springer_inline(body))
            elif fmt == "springer_single":
                all_editors.extend(_extract_springer_single(body))
            elif fmt == "bold_two_line":
                all_editors.extend(_extract_bold_two_line(body, heading))
            elif fmt == "npj_bold_italic":
                all_editors.extend(_extract_npj_bold_italic(body))
            elif fmt == "acs_plain":
                all_editors.extend(_extract_acs_plain(body))
            elif fmt == "sage_table":
                all_editors.extend(_extract_sage_table(body))
            elif fmt == "frontiers_block":
                all_editors.extend(_extract_frontiers_block(body))
            elif fmt == "elsevier_bold_section":
                all_editors.extend(_extract_elsevier_bold_section(body))
            elif fmt == "unknown":
                pass

    # Deduplicate
    seen = set()
    unique: list[Editor] = []
    for ed in all_editors:
        key = (ed.name.lower().strip(), tuple(a.lower() for a in ed.affiliations))
        if key not in seen:
            seen.add(key)
            unique.append(ed)
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the structural editor parser")
    parser.add_argument("file", type=Path, help="Path to the editors markdown file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print all extracted editors")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File {args.file} not found")
        exit(1)

    text = args.file.read_text(encoding="utf-8", errors="ignore")
    editors = parse_editors(text)

    print(f"Extracted {len(editors)} editors from {args.file.name}")
    if args.verbose:
        for ed in editors:
            print(f"- {ed.name} ({ed.role}) | {', '.join(ed.affiliations)}")
