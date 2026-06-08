# Structural Editor Parser

## Motivation

12 of 38 journals have editors files > 100K chars (25K tokens). The largest (`srep`) is 1M chars (250K tokens). Giving full editors text to the LLM extraction agent overflows context windows. A structural parser extracts editors from recognizable patterns, eliminating the need for LLM extraction on structured data.

## Architecture

```
parse_editors(text: str) -> list[Editor]
    │
    ├─ split_sections(text) → list[(heading, body)]
    │
    ├─ For each section:
    │   ├─ detect_format(body) → EditorFormat
    │   ├─ extract_by_format(format, body) → list[Editor]
    │   └─ Normalize: strip degrees, clean affiliations
    │
    └─ Deduplicate by (name_normalized, affiliation_normalized)
```

## Heading-Level Variability

Heading levels are **not** consistent across publishers. The parser must detect the file's heading structure before splitting.

| Publisher | `#` | `##` | `###` | Split level | Format |
|---|---|---|---|---|---|
| **srep** | 1 | 56 | 49 | `##` (56 sections) | A, B, C |
| **nmeth** | 1 | 9 | 0 | `##` (9 inline editors) | K |
| **npj** | 1 | 4 | 0 | `##` (4 sections) | D |
| **Sage** | 0 | 2 | 0 | `##` (2 sections) | F |
| **Wiley** | 0 | 2 | 1 | `##` (2 sections) | H |
| **Frontiers** | 0 | 0 | 1 | none (1 section) | G |
| **ACS** | 0 | 0 | 0 | none (1 section) | E |

**Strategy**: `detect_file_structure()` counts `#`, `##`, `###` heading occurrences. Picks the level with count >= 2 and highest count. For files with no headings or only 1 heading at any level, the entire text is one section.

**Special case — nmeth**: `##` headings contain inline editor data (`## Role: Name, PhD, Affiliation, Country ORCiD`). After splitting on `##`, each section heading is the editor entry. `detect_format()` returns `NMETH_INLINE` when the heading itself matches the inline pattern.

## Detected Formats

### Format A: Springer Inline

**Source**: `springer_nature/srep` (In-house Editors section only)

```markdown
### Chief Editor: Rafal Marszalek, PhD; Springer Nature, UK

Rafal's background is analytical and biological chemistry...

ORCID 0000-0003-0316-1363
```

**Pattern**: `### (Role): (Name), (degree); (Affiliation), (Country)`

**Regex**: `^###\s+(.+?):\s+(.+?),\s+[^;]+;\s+(.+?)\s*,\s*(.+)$`

**Groups**: role, name, affiliation, country

**Extraction**:
- `role`: Group 1 as-is (e.g. "Chief Editor")
- `name`: Group 2, strip trailing degrees via `strip_degrees()`
- `affiliations`: `[f"{group3}, {group4}"]`

**Coverage**: ~49 editors in srep's in-house section. Also matches `springer_link/bmc-biology` if formatted similarly.

---

### Format B: Springer Single-Line

**Source**: `springer_nature/srep` (Senior Editorial Board section)

```markdown
**Guido Caldarelli,** PhD, Ca' Foscari University of Venice, Italy - Networks and Complex Systems
```

**Pattern**: `**Name**, (degree), (Affiliation), (Country) - (Subject)`

**Regex**: `^\*\*(.+?)\*\*,?\s*(?:PhD|Dr\.?|MD|Professor|Prof\.?)[^,]*,\s*(.+?)\s*,\s*(.+?)\s*-\s*(.+)$`

**Groups**: name, affiliation, country, subject

**Extraction**:
- `role`: "Senior Editorial Board Member"
- `name`: Group 1, strip degrees
- `affiliations`: `[f"{group2}, {group3}"]`
- Subject area is not stored in `Editor` model but could be used for debugging

**Coverage**: ~200 editors in srep's Senior Editorial Board

---

### Format C: Bold-Name Two-Line

**Source**: `springer_nature/srep` (Editorial Board Members, subject-area sections)

```markdown
## REGISTERED REPORTS

**Deborah Apthorp, **

University of New England, NSW, Australia

**Stepan Bahnik, Dr.**

Prague University of Economics and Business, Czech Republic
```

**Pattern**: `**Name**, (degree)` on one line, affiliation on next non-empty line

**Regex** (name line): `^\*\*(.+?)(?:,\s*)?(?:PhD|Dr\.?|MD|Professor|Prof\.?|BSc|MSc|Dr\. med\.?)?\*\*$`
**Regex** (affiliation line): `^([A-Z][^\n]*?)\s*,\s*(?:USA|UK|China|Germany|France|Japan|Canada|Australia|Italy|Spain|India|Brazil|[^,]+)$`

**Extraction** (paired via line iteration):
- `role`: Section heading (e.g. "REGISTERED REPORTS")
- `name`: Group 1 from name line, strip degrees
- `affiliations`: `[affiliation line text]`

**Coverage**: ~500 editors across ~50 subject-area sections in srep

---

### Format D: npj Bold-Italic

**Source**: `springer_nature/npjclimataction`

```markdown
## Editor-in-Chief:

**Jale Tosun, PhD**, *Ruprecht Karl University of Heidelberg, Germany*

## Associate Editors:

**Paul Almeida**, PhD, *University of California, Merced, CA, USA*
```

**Pattern**: `**Name**, (degree)`, `*Affiliation, Country*`

**Regex**: `^\*\*(.+?)\*\*,?\s*(?:PhD|Dr\.?|MD|Professor)[^*]*\*\*(.+?)\*\*$`

**Groups**: name, affiliation+country

**Extraction**:
- `role`: From section heading (e.g. "Editor-in-Chief", "Associate Editor")
- `name`: Group 1, strip degrees
- `affiliations`: `[group2.strip("* ,")]`

**Coverage**: ~15 editors in npjclimataction. Also applies to `springer_nature/nmeth` if similar.

---

### Format E: ACS Plain Triple

**Source**: `acs/es-and-t`

```markdown
Joshua Apte

University of California Berkeley

United States

Win Cowger

The Moore Institute for Plastic Pollution Research

United States
```

**Pattern**: Three lines per entry: name, affiliation, country. Separated by blank lines.

**Detection**: Section has no `**` bold markers, no `###` headings, no tables. Lines alternate between name-like and affiliation-like.

**Extraction** (line iteration):
- Split on blank lines to get groups of 3 lines
- Line 1 = name, Line 2 = affiliation, Line 3 = country
- `role`: None (ACS doesn't specify roles in this file)
- `affiliations`: `[f"{line2}, {line3}"]`

**Coverage**: ~15 editors in es-and-t. Likely applies to `acs/acs-nano`, `acs/acs-energy-letters`, `acs/jacs`.

---

### Format F: Sage Markdown Table

**Source**: `sage/sgo`, `sage/smsa`, `sage/smo`

```markdown
## Editorial board

Editor-in-Chief

| Rory Magrath | SAGE Publications, UK |

Business and Management

| Innocent Senyo Kwasi Acquah | University of Cape Coast, Ghana |
```

**Pattern**: `| Name | Affiliation, Country |` with role from preceding heading

**Regex**: `^\|\s*(.+?)\s*\|\s*(.+?)\s*\|`

**Groups**: name, affiliation+country

**Extraction**:
- `role`: From preceding non-table, non-heading line (e.g. "Business and Management")
- `name`: Group 1
- `affiliations`: `[group2]`

**Coverage**: ~100 editors across sage journals

---

### Format G: Frontiers Block

**Source**: `frontiers/public-health`, `frontiers/energy-research`, `frontiers/microbiology`

```markdown
### paolo vineis

Imperial College London

London, United Kingdom

Field Chief Editor

Frontiers in Public Health
```

**Pattern**: `### name` → affiliation → city, country → role

**Detection**: Lines start with `### `, followed by 2-3 lines of affiliation/location, then a role line.

**Extraction** (line iteration):
- `### Name` → name
- Next non-empty line → affiliation
- Next line with `,` → city, country (skip or merge with affiliation)
- Role line → role
- `affiliations`: `[affiliation line]`

**Coverage**: ~50-100 editors per frontiers journal. These are the largest files (2.8M+ tokens).

---

### Format H: Wiley Dash-Separated List

**Source**: `wiley/advanced-functional-materials`

```markdown
### Advisory Board

- Khalil Amine,
*Argonne National Laboratory, Lemont* - Thomas D. Anthopoulos,
*King Abdullah University of Science and Technology, Thuwal* - ...
```

**Pattern**: `- Name, *Affiliation* - Name, *Affiliation* - ...` (multiple editors per line, separated by ` - `)

**Regex**: `(?:^|\s*-\s*)(\w[\w\s\.]+?),?\s*\*(.+?)\*`

**Groups**: name, affiliation

**Extraction**:
- `role`: From section heading (e.g. "Advisory Board")
- `name`: Group 1
- `affiliations`: `[group2]`

**Coverage**: ~40 editors in Wiley AFM. Likely applies to other Wiley journals.

---

### Format I: IEEE Scraped Form

**Source**: `ieee/taes`, `ieee/tbdata`, `ieee/tmrb`

```markdown
Position(s)

Contact

Country

USA

Affiliation

Air Force Research Laboratory

IEEE Region

Region 2 (Eastern U.S.)
```

**Pattern**: Form field labels followed by values. No name visible in text — names are in HTML `img` tags or alt text that didn't scrape.

**Status**: **Cannot parse** — name field is missing from markdown. Fall back to LLM extraction.

**Workaround**: If the original HTML is available, extract from `<h3>` or `<img alt="...">` tags. Otherwise, skip IEEE editors or re-scrape with better name extraction.

---

### Format J: MDPI Interests-First

**Source**: `mdpi/sustainability`, `mdpi/energies`, `mdpi/buildings`

```markdown
**Interests:**hydrology and water resources; sustainable agriculture

Special Issues, Collections and Topics in MDPI journals

**Interests:**sustainability; sustainable development; energy

* Section: Sustainable Materials
```

**Pattern**: `**Interests:**...` blocks with optional `* Section: ...`

**Status**: **Cannot parse** — no name/affiliation in the visible text. Names are likely in images or links that didn't scrape.

**Workaround**: Re-scrape MDPI editorial board pages with name extraction, or fall back to LLM if names are present in other files (e.g. `about.md`).

---

## Format Detection

```python
def detect_format(body: str) -> EditorFormat:
    """Heuristic detection based on first 500 chars of section body."""
    sample = body[:500]

    if re.search(r'^###\s+\w+:', sample, re.MULTILINE):
        return EditorFormat.SPRINGER_INLINE
    if re.search(r'^\*\*.+?\*\*,?.+?-.+$', sample, re.MULTILINE):
        return EditorFormat.SPRINGER_SINGLE
    if re.search(r'^\*\*.+?\*\*$', sample, re.MULTILINE) and not re.search(r'\|', sample):
        return EditorFormat.BOLD_TWO_LINE
    if re.search(r'^\*\*.+?\*\*,?\s*\*', sample, re.MULTILINE):
        return EditorFormat.NPJ_BOLD_ITALIC
    if re.search(r'^\|', sample, re.MULTILINE):
        return EditorFormat.SAGE_TABLE
    if re.search(r'^###\s+\w', sample, re.MULTILINE) and not re.search(r':', sample[:200]):
        return EditorFormat.FRONTIERS_BLOCK
    if re.search(r'^-\s+\w+,\s*\*', sample, re.MULTILINE) or re.search(r'\w+,\s*\*.*?-\s*\w+,\s*\*', sample):
        return EditorFormat.WILEY_DASH
    if not re.search(r'\*\*', sample) and not re.search(r'\|', sample):
        return EditorFormat.ACS_PLAIN
    return EditorFormat.UNKNOWN
```

## Implementation

File: `scripts/editor_parser.py`

```python
import re
from dataclasses import dataclass, field
from typing import Optional

from models.journal import Editor


@dataclass
class EditorFormat:
    SPRINGER_INLINE = "springer_inline"
    SPRINGER_SINGLE = "springer_single"
    BOLD_TWO_LINE = "bold_two_line"
    NPJ_BOLD_ITALIC = "npj_bold_italic"
    ACS_PLAIN = "acs_plain"
    SAGE_TABLE = "sage_table"
    FRONTIERS_BLOCK = "frontiers_block"
    WILEY_DASH = "wiley_dash"
    UNKNOWN = "unknown"


DEGREE_PATTERN = re.compile(
    r'(?:PhD|MD|Dr\.?|Prof\.?|Professor|BSc|MSc|DSc|Dr\. med\.?|Dr\. rer\. nat|'
    r'Dr\. habil|Dr\. Sci\.|FDS RCPs|FICD|FADM|FIMMM|FCG Dent|FHEA|'
    r'Habilitation|Dr\. Habil|Full professor|B\.Ch\.D\.|MBBS|FFICM|SFHEA)',
    re.IGNORECASE
)


def strip_degrees(name: str) -> str:
    """Remove academic degrees and titles from editor name."""
    name = DEGREE_PATTERN.sub('', name)
    name = re.sub(r'\s*,\s*$', '', name)  # trailing comma
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def detect_file_structure(text: str) -> int:
    """Detect the heading level used for section splitting.

    Counts #, ##, ### heading occurrences. Returns the level (1, 2, or 3)
    that appears most frequently as a structural divider. Returns 0 if
    no headings are found (entire text is one section).
    """
    counts = {1: 0, 2: 0, 3: 0}
    for line in text.split('\n'):
        m = re.match(r'^(#{1,3})\s+', line)
        if m:
            level = len(m.group(1))
            counts[level] = counts.get(level, 0) + 1
    # Pick the level with most occurrences, minimum 2 to be structural
    best = max(((lvl, c) for lvl, c in counts.items() if c >= 2), default=None)
    if best:
        return best[0]
    return 0


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections.

    Detects the file's heading level automatically:
    - If ## appears most: split on ## (srep, npj, sage)
    - If ### appears most: split on ### (frontiers)
    - If no headings: entire text is one section (acs)
    """
    level = detect_file_structure(text)

    if level == 0:
        # No structural headings — entire text is one section
        # Try to find a title from first # heading or use "editors"
        title_m = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        heading = title_m.group(1).strip() if title_m else "editors"
        return [(heading, text)]

    # Split on the detected heading level
    pattern = r'^#{' + str(level) + r'}\s+'
    parts = re.split(pattern, text, flags=re.MULTILINE)

    if len(parts) < 2:
        return [("editors", text)]

    sections = []
    # parts[0] is content before first heading of detected level
    if parts[0].strip():
        # Pre-heading content — may contain a title
        title_m = re.search(r'^#\s+(.+)$', parts[0], re.MULTILINE)
        heading = title_m.group(1).strip() if title_m else "intro"
        sections.append((heading, parts[0]))

    for i in range(1, len(parts)):
        body = parts[i]
        heading = body.split('\n')[0].strip()
        sections.append((heading, body))

    return sections


def extract_springer_inline(body: str) -> list[Editor]:
    """Format A: ### Role: Name, PhD; Affiliation, Country"""
    editors = []
    for m in re.finditer(
        r'^###\s+(.+?):\s+(.+?),\s+[^;]+;\s+(.+?)\s*,\s*(.+)$',
        body, re.MULTILINE
    ):
        role, name, affiliation, country = m.groups()
        editors.append(Editor(
            name=strip_degrees(name),
            role=role.strip(),
            affiliations=[f"{affiliation.strip()}, {country.strip()}"],
        ))
    return editors


def extract_springer_single(body: str) -> list[Editor]:
    """Format B: **Name**, PhD, Affiliation, Country - Subject"""
    editors = []
    for m in re.finditer(
        r'^\*\*(.+?)\*\*,?\s*(?:PhD|Dr\.?|MD|Professor|Prof\.?)[^,]*,\s*(.+?)\s*,\s*(.+?)\s*-\s*(.+)$',
        body, re.MULTILINE
    ):
        name, affiliation, country, subject = m.groups()
        editors.append(Editor(
            name=strip_degrees(name),
            role="Senior Editorial Board Member",
            affiliations=[f"{affiliation.strip()}, {country.strip()}"],
        ))
    return editors


def extract_bold_two_line(body: str, section_heading: str = None) -> list[Editor]:
    """Format C: **Name**, PhD on one line, Affiliation on next"""
    editors = []
    lines = body.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'^\*\*(.+?)(?:,\s*)?(?:PhD|Dr\.?|MD|Professor|Prof\.?|BSc|MSc|Dr\. med\.?)?\*\*\s*$', lines[i])
        if m:
            name = strip_degrees(m.group(1))
            # Find next non-empty line as affiliation
            aff = None
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].strip():
                    aff = lines[j].strip()
                    i = j + 1
                    break
            if aff:
                editors.append(Editor(
                    name=name,
                    role=section_heading if section_heading and section_heading not in ("Editorial Board Members",) else None,
                    affiliations=[aff],
                ))
        i += 1
    return editors


def extract_npj_bold_italic(body: str) -> list[Editor]:
    """Format D: **Name**, PhD, *Affiliation, Country*"""
    editors = []
    current_role = None

    for line in body.split('\n'):
        # Detect role from section heading (any level)
        role_m = re.match(r'^#{1,3}\s+(.+?):\s*$', line)
        if role_m:
            current_role = role_m.group(1).strip()
            continue

        # Detect editor entry
        m = re.match(r'^\*\*(.+?)\*\*,?\s*(?:PhD|Dr\.?|MD|Professor)?\s*,?\s*\*(.+?)\*$', line)
        if m:
            name = strip_degrees(m.group(1))
            affiliation = m.group(2).strip().strip('* ,')
            editors.append(Editor(
                name=name,
                role=current_role,
                affiliations=[affiliation],
            ))
    return editors


def extract_acs_plain(body: str) -> list[Editor]:
    """Format E: Name / Affiliation / Country on separate lines, blank-separated"""
    editors = []
    blocks = re.split(r'\n\n+', body.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) >= 2:
            name = lines[0]
            aff = lines[1]
            country = lines[2] if len(lines) > 2 else None
            aff_str = f"{aff}, {country}" if country else aff
            editors.append(Editor(
                name=name,
                role=None,
                affiliations=[aff_str],
            ))
    return editors


def extract_sage_table(body: str) -> list[Editor]:
    """Format F: | Name | Affiliation, Country |"""
    editors = []
    current_role = None

    for line in body.split('\n'):
        # Detect role from non-table, non-heading text
        if re.match(r'^#{1,3}\s', line):
            continue
        if not re.match(r'^\s*\|', line) and line.strip():
            current_role = line.strip()
            continue

        m = re.match(r'^\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|', line)
        if m:
            name = m.group(1).strip()
            affiliation = m.group(2).strip()
            editors.append(Editor(
                name=name,
                role=current_role,
                affiliations=[affiliation],
            ))
    return editors


def extract_frontiers_block(body: str) -> list[Editor]:
    """Format G: ### name → Affiliation → City, Country → Role"""
    editors = []
    lines = body.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'^###\s+(.+)$', lines[i])
        if m:
            name = m.group(1).strip()
            aff = None
            role = None
            for j in range(i + 1, min(i + 8, len(lines))):
                line = lines[j].strip()
                if not line:
                    continue
                if line.startswith('###'):
                    break
                if not aff:
                    aff = line
                elif role is None and any(r in line.lower() for r in
                    ['chief editor', 'section editor', 'specialty', 'associate', 'deputy']):
                    role = line
                # else: city/country line, skip
            if aff:
                editors.append(Editor(
                    name=name,
                    role=role,
                    affiliations=[aff],
                ))
            i = j
        i += 1
    return editors


def extract_wiley_dash(body: str) -> list[Editor]:
    """Format H: - Name, *Affiliation* - Name, *Affiliation*"""
    editors = []
    current_role = None

    # Find role from heading (any level)
    role_m = re.search(r'^#{1,3}\s+(.+)$', body, re.MULTILINE)
    if role_m:
        current_role = role_m.group(1).strip()

    # Find all name-affiliation pairs
    for m in re.finditer(r'(?:^|\s*-\s*)(\w[\w\s\.]+?),?\s*\*(.+?)\*', body, re.MULTILINE):
        name = m.group(1).strip()
        affiliation = m.group(2).strip()
        if len(name) > 1:  # Skip very short matches
            editors.append(Editor(
                name=name,
                role=current_role,
                affiliations=[affiliation],
            ))
    return editors


def parse_editors(text: str) -> list[Editor]:
    """Parse editors from markdown text using format-specific extractors."""
    sections = split_sections(text)
    all_editors: list[Editor] = []

    for heading, body in sections:
        fmt = detect_format(body)

        if fmt == EditorFormat.SPRINGER_INLINE:
            all_editors.extend(extract_springer_inline(body))
        elif fmt == EditorFormat.SPRINGER_SINGLE:
            all_editors.extend(extract_springer_single(body))
        elif fmt == EditorFormat.BOLD_TWO_LINE:
            all_editors.extend(extract_bold_two_line(body, heading))
        elif fmt == EditorFormat.NPJ_BOLD_ITALIC:
            all_editors.extend(extract_npj_bold_italic(body))
        elif fmt == EditorFormat.ACS_PLAIN:
            all_editors.extend(extract_acs_plain(body))
        elif fmt == EditorFormat.SAGE_TABLE:
            all_editors.extend(extract_sage_table(body))
        elif fmt == EditorFormat.FRONTIERS_BLOCK:
            all_editors.extend(extract_frontiers_block(body))
        elif fmt == EditorFormat.WILEY_DASH:
            all_editors.extend(extract_wiley_dash(body))
        # UNKNOWN → fall through, no extraction

    # Deduplicate
    seen = set()
    unique = []
    for ed in all_editors:
        key = (ed.name.lower().strip(), tuple(a.lower() for a in ed.affiliations))
        if key not in seen:
            seen.add(key)
            unique.append(ed)
    return unique
```

## Pipeline Integration

In `scripts/generate_golden_samples.py`, Phase 1 extraction for editors (pass 3) becomes:

```python
# Phase 1: Extract
for pass_config in PASSES:
    pass_index = pass_config["index"]  # 0-3

    if pass_index == 3:  # Editors pass
        # Try structural parser first
        editors_file = journal_dir / "editors.md"
        if not editors_file.exists():
            editors_file = journal_dir / "editorial_board.md"

        if editors_file.exists():
            text = editors_file.read_text(encoding="utf-8")
            parsed = parse_editors(text)
            if parsed:
                # Parser succeeded — use parsed editors, skip LLM agent
                results[pass_index] = EditorialExtraction(editors=parsed)
                continue

    # Fallback: LLM extraction agent
    agent = make_agent(pass_config, include_search=False)
    result = await run_extraction_pass(agent, journal_id, nodes)
    results[pass_index] = result
```

## Coverage Analysis

| Format | Journals | Est. Editors | Parseable? |
|---|---|---|---|
| A: Springer Inline | srep (in-house) | 49 | Yes |
| B: Springer Single | srep (senior board) | ~200 | Yes |
| C: Bold Two-Line | srep (subject areas) | ~500 | Yes |
| D: npj Bold-Italic | npjclimataction | ~15 | Yes |
| E: ACS Plain | acs/es-and-t, acs-nano, acs-energy-letters, jacs | ~60 | Partial |
| F: Sage Table | sage/sgo, sage/smsa, sage/smo, sage/jom | ~150 | Yes |
| G: Frontiers Block | frontiers/public-health, energy-research, microbiology | ~300 | Partial |
| H: Wiley Dash | wiley/advanced-functional-materials, others | ~60 | **No** |
| I: IEEE Form | ieee/taes, tbdata, tmrb | ~30 | **No** (names missing) |
| J: MDPI Interests | mdpi/sustainability, energies, buildings | ~150 | **No** (names missing) |
| K: nmeth Inline | nmeth | 9 | Yes |
| L: Elsevier Bold | the-lancet | ~40 | Yes |
| **Total parseable** | **18 of 38 journals** | **~1200** | |
| **Fallback to LLM** | **20 journals** | **~200** | |

## Token Savings

- **Editors pass LLM calls eliminated**: 18 journals × 1 call = 18 calls saved
- **Context tokens saved**: ~1.8M tokens (sum of all parseable editors files)
- **Output tokens saved**: ~18 × 2K = 36K tokens
- **srep alone**: Saves 250K input tokens (the entire editors file)
- **nmeth**: Saves 10K input tokens
- **sage journals**: Saves ~300K input tokens
- **frontiers journals**: Saves ~3M input tokens (largest files)

## Maintenance

New publisher format not in the parser → `detect_format()` returns `UNKNOWN` → falls back to LLM agent. No breakage, just an extra LLM call. Add new format extractor when a new publisher pattern is observed.

## Testing

Run parser on each journal's editors file and compare count/structure against LLM extraction output. First run:

```bash
python scripts/editor_parser.py --test journal-samples/springer_nature/extracted/srep/editors.md
# Expected: ~750 editors extracted
```
