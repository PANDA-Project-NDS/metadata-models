import pytest
from golden.structural_parser import parse_editors, detect_format
from models.journal import Editor

# --- Test Data ---

# Real-world anonymized snippets (~5 editors each) sourced from whitelisted
# journal files.  Names are pseudonyms; structure, degrees, affiliations and
# roles are faithful to the originals so the parser exercises the same logic.
#
# Source files:
#   springer_inline_real   -> srep/editors.md (In-house Editors)
#   springer_single_real   -> srep/editors.md (Senior Editorial Board)
#   acs_plain_real         -> acs/es-and-t/editors.md
#   sage_table_real        -> sage/smsa/editorial_board.md
#   npj_bold_italic_real   -> npjclimataction/editors.md
#   frontiers_block_real   -> frontiers/energy-research/editors.md
#   elsevier_bold_real     -> elsevier/information-sciences/editorial_board.md
#   heading_name_real      -> springer_link/epj-data-science/editorial-board.md

TEST_CASES = [
    # --- springer_inline (5 editors, srep In-house Editors) -------------------
    # Needs >= 5 ## headings so split_sections picks ## (not ###) as split level.
    (
        "springer_inline",
        "\n".join([
            "## In-house Editors",
            "",
            "### Chief Editor: A. J. Thornton, PhD; Springer Nature, UK",
            "",
            "A. J. Thornton's background is analytical chemistry...",
            "",
            "ORCID 0000-0003-0316-1363",
            "",
            "### Deputy Editor: B. C. Lin, PhD; Springer Nature, China",
            "",
            "B. C. Lin holds a PhD in organic chemistry...",
            "",
            "ORCID 0000-0001-6004-7087",
            "",
            "## Senior Editors",
            "",
            "### Deputy Editor: C. D. Evans, PhD; Springer Nature, UK",
            "",
            "C. D. Evans has a background in pharmacology...",
            "",
            "ORCID 0000-0003-2616-3193",
            "",
            "## Deputy Editors",
            "",
            "### Deputy Editor: D. E. Patel, PhD; Springer Nature, India",
            "",
            "D. E. Patel has a background in Nanochemistry...",
            "",
            "ORCID 0009-0008-1309-9050",
            "",
            "## Associate Editors",
            "",
            "### Deputy Editor: E. F. Rossi, PhD; Springer Nature, UK",
            "",
            "E. F. Rossi joined in 2019...",
            "",
            "ORCID 0000-0001-5083-5936",
            "",
            "## Editorial Support",
        ]),
        [
            Editor(name="A. J. Thornton", affiliations=["Springer Nature, UK"], role="Chief Editor"),
            Editor(name="B. C. Lin", affiliations=["Springer Nature, China"], role="Deputy Editor"),
            Editor(name="C. D. Evans", affiliations=["Springer Nature, UK"], role="Deputy Editor"),
            Editor(name="D. E. Patel", affiliations=["Springer Nature, India"], role="Deputy Editor"),
            Editor(name="E. F. Rossi", affiliations=["Springer Nature, UK"], role="Deputy Editor"),
        ],
    ),

    # --- springer_single (5 editors, srep Senior Editorial Board) -------------
    # Tests degree variants: PhD, DVM+MS+PhD, MSc+PhD, no degree, Prof. Dr
    # Note: DVM and MS are NOT in DEGREE_PATTERN, so they remain in affiliation.
    (
        "springer_single",
        "\n".join([
            "**A. B. Murray**, PhD, University of Adelaide, Australia - Biological Physics",
            "**B. C. Fernandez**, DVM, MS, PhD, Michigan State University, USA - Immunology",
            "**C. D. Wang**, MSc, PhD, Henan University of Science and Technology, China - Plants",
            "**D. E. Kim**, Chung-Ang University, South Korea - Cell Biology",
            "**E. F. Weber**, Prof. Dr, University of Twente, The Netherlands - Drug Discovery",
        ]),
        [
            Editor(name="A. B. Murray", affiliations=["University of Adelaide, Australia"], role="Biological Physics"),
            Editor(name="B. C. Fernandez", affiliations=["DVM, MS, Michigan State University, USA"], role="Immunology"),
            Editor(name="C. D. Wang", affiliations=["Henan University of Science and Technology, China"], role="Plants"),
            Editor(name="D. E. Kim", affiliations=["Chung-Ang University, South Korea"], role="Cell Biology"),
            Editor(name="E. F. Weber", affiliations=["University of Twente, The Netherlands"], role="Drug Discovery"),
        ],
    ),

    # --- acs_plain (5 editors, acs/es-and-t) ----------------------------------
    # Tests 4-field grouping (name / affiliation / country / email)
    (
        "acs_plain",
        "\n".join([
            "## Executive Editors",
            "",
            "A. B. Nguyen",
            "",
            "Rice University",
            "",
            "United States",
            "",
            "E-mail: alvarez-office@est.acs.org",
            "",
            "B. C. Thompson",
            "",
            "Stanford University",
            "",
            "United States",
            "",
            "E-mail: aboehm@stanford.edu",
            "",
            "C. D. Rivera",
            "",
            "University of California Irvine",
            "",
            "United States",
            "",
            "E-mail: carlton-office@uci.edu",
            "",
            "D. E. Okafor",
            "",
            "Georgia Institute of Technology",
            "",
            "United States",
            "",
            "E-mail: okafor-office@gatech.edu",
            "",
            "E. F. Johansson",
            "",
            "KTH Royal Institute of Technology",
            "",
            "Sweden",
            "",
            "E-mail: johansson-office@kth.se",
        ]),
        [
            Editor(name="A. B. Nguyen", affiliations=["Rice University, United States"], role=None),
            Editor(name="B. C. Thompson", affiliations=["Stanford University, United States"], role=None),
            Editor(name="C. D. Rivera", affiliations=["University of California Irvine, United States"], role=None),
            Editor(name="D. E. Okafor", affiliations=["Georgia Institute of Technology, United States"], role=None),
            Editor(name="E. F. Johansson", affiliations=["KTH Royal Institute of Technology, Sweden"], role=None),
        ],
    ),

    # --- sage_table (5 editors, sage/smsa) ------------------------------------
    # Tests role tracking across table blocks with section headings
    (
        "sage_table",
        "\n".join([
            "## Editorial board",
            "",
            "Editors",
            "",
            "| A. B. Carter | Sage Publishing, India |",
            "",
            "Allergy & Immunology",
            "",
            "| B. C. Rossi | Ospedali Riuniti Marche Nord, Italy |",
            "| C. D. Cohen | Sheba Medical Center, Israel |",
            "",
            "Cardiology",
            "",
            "| D. E. Novak | Nicolaus Copernicus University, Poland |",
            "",
            "Clinical Epidemiology",
            "",
            "| E. F. Idris | Kazan State Medical University, Russia |",
        ]),
        [
            Editor(name="A. B. Carter", affiliations=["Sage Publishing, India"], role="Editors"),
            Editor(name="B. C. Rossi", affiliations=["Ospedali Riuniti Marche Nord, Italy"], role="Allergy & Immunology"),
            Editor(name="C. D. Cohen", affiliations=["Sheba Medical Center, Israel"], role="Allergy & Immunology"),
            Editor(name="D. E. Novak", affiliations=["Nicolaus Copernicus University, Poland"], role="Cardiology"),
            Editor(name="E. F. Idris", affiliations=["Kazan State Medical University, Russia"], role="Clinical Epidemiology"),
        ],
    ),

    # --- npj_bold_italic (5 editors, npjclimataction) -------------------------
    # Tests role from heading, **Name**, PhD, *Affiliation* pattern
    (
        "npj_bold_italic",
        "\n".join([
            "## Associate Editors:",
            "",
            "**A. B. Sullivan**, PhD, *University of California, Merced, CA, USA*",
            "",
            "A. B. Sullivan is a Professor of Sociology...",
            "",
            "**B. C. Vargas**, PhD, *Instituto de Altos Estudios Nacionales, Ecuador*",
            "",
            "B. C. Vargas is an Associate Professor...",
            "",
            "**C. D. Rao**, PhD, *University of Cambridge, Cambridge, UK*",
            "",
            "C. D. Rao is an assistant professor...",
            "",
            "**D. E. Bergman**, PhD, *The Hebrew University, Jerusalem, Israel*",
            "",
            "D. E. Bergman's research sits at the intersection...",
            "",
            "**E. F. Mueller**, PhD, *ETH Zurich, Switzerland*",
            "",
            "E. F. Mueller is a Postdoctoral Researcher...",
        ]),
        [
            Editor(name="A. B. Sullivan", affiliations=["University of California, Merced, CA, USA"], role="Associate Editors"),
            Editor(name="B. C. Vargas", affiliations=["Instituto de Altos Estudios Nacionales, Ecuador"], role="Associate Editors"),
            Editor(name="C. D. Rao", affiliations=["University of Cambridge, Cambridge, UK"], role="Associate Editors"),
            Editor(name="D. E. Bergman", affiliations=["The Hebrew University, Jerusalem, Israel"], role="Associate Editors"),
            Editor(name="E. F. Mueller", affiliations=["ETH Zurich, Switzerland"], role="Associate Editors"),
        ],
    ),

    # --- frontiers_block (5 editors, frontiers/energy-research) ---------------
    # Needs >= 5 ## headings so split_sections picks ## (not ###) as split level.
    (
        "frontiers_block",
        "\n".join([
            "## Field Chief Editors",
            "",
            "### a. b. schrader",
            "",
            "Institute of Biochemistry, University of Greifswald",
            "",
            "Greifswald, Germany",
            "",
            "Field Chief Editor",
            "",
            "## Specialty Editors - Wave",
            "",
            "### b. c. moretti",
            "",
            "Mediterranea University of Reggio Calabria",
            "",
            "Reggio Calabria, Italy",
            "",
            "Specialty Chief Editor",
            "",
            "Wave and Tidal Energy",
            "",
            "## Specialty Editors - Hydrogen",
            "",
            "### c. d. huber",
            "",
            "Swiss Federal Laboratories for Materials Science and Technology",
            "",
            "Dubendorf, Switzerland",
            "",
            "Specialty Chief Editor",
            "",
            "Hydrogen Storage and Production",
            "",
            "## Specialty Editors - Energy",
            "",
            "### d. e. park",
            "",
            "Clemson University",
            "",
            "Clemson, United States",
            "",
            "Specialty Chief Editor",
            "",
            "Sustainable Energy Systems",
            "",
            "## Specialty Editors - Fuel Cells",
            "",
            "### e. f. zhang",
            "",
            "University of South Carolina",
            "",
            "Columbia, United States",
            "",
            "Specialty Chief Editor",
            "",
            "Fuel Cells, Electrolyzers and Membrane Reactors",
        ]),
        [
            Editor(name="a. b. schrader", affiliations=["Institute of Biochemistry, University of Greifswald, Greifswald, Germany"], role="Field Chief Editor"),
            Editor(name="b. c. moretti", affiliations=["Mediterranea University of Reggio Calabria, Reggio Calabria, Italy"], role="Specialty Chief Editor"),
            Editor(name="c. d. huber", affiliations=["Swiss Federal Laboratories for Materials Science and Technology, Dubendorf, Switzerland"], role="Specialty Chief Editor"),
            Editor(name="d. e. park", affiliations=["Clemson University, Clemson, United States"], role="Specialty Chief Editor"),
            Editor(name="e. f. zhang", affiliations=["University of South Carolina, Columbia, United States"], role="Specialty Chief Editor"),
        ],
    ),

    # --- elsevier_bold_section (5 editors, elsevier/information-sciences) -----
    # Needs ## count >= ### count so split_sections picks ## as split level.
    (
        "elsevier_bold_section",
        "\n".join([
            "## Editorial board",
            "",
            "### Editors-in-Chief",
            "",
            "#### A. B. Santoro, PhD",
            "",
            "University of Salerno, Department of Information and Electrical Engineering, Fisciano, Italy",
            "",
            "#### B. C. Liu, PhD",
            "",
            "Xidian University, Xi'an, China",
            "",
            "## Senior Editors",
            "",
            "### Senior Editors",
            "",
            "#### C. D. Pratama",
            "",
            "University of South Australia, UniSA STEM, Mawson Lakes, South Australia, Australia",
            "",
            "#### D. E. Qasim, PhD",
            "",
            "King Fahd University of Petroleum & Minerals, Dhahran, Saudi Arabia",
            "",
            "#### E. F. Tavana, PhD",
            "",
            "La Salle University, Philadelphia, Pennsylvania, United States",
        ]),
        [
            Editor(name="A. B. Santoro", affiliations=["University of Salerno, Department of Information and Electrical Engineering, Fisciano, Italy"], role="Editors-in-Chief"),
            Editor(name="B. C. Liu", affiliations=["Xidian University, Xi'an, China"], role="Editors-in-Chief"),
            Editor(name="C. D. Pratama", affiliations=["University of South Australia, UniSA STEM, Mawson Lakes, South Australia, Australia"], role="Senior Editors"),
            Editor(name="D. E. Qasim", affiliations=["King Fahd University of Petroleum & Minerals, Dhahran, Saudi Arabia"], role="Senior Editors"),
            Editor(name="E. F. Tavana", affiliations=["La Salle University, Philadelphia, Pennsylvania, United States"], role="Senior Editors"),
        ],
    ),

    # --- heading_name (5 editors, springer_link/epj-data-science) -------------
    # Detected at parse_editors level (not via detect_format), like nmeth_inline.
    # ### Name PhD -> affiliation on next line
    (
        "heading_name",
        "\n".join([
            "# Editorial board",
            "",
            "## Editors-in-Chief",
            "",
            "-",
            "### A. B. Garcia PhD",
            "",
            "University of Konstanz, Konstanz, Germany",
            "",
            "-",
            "### B. C. Mejova PhD",
            "",
            "Institute for Scientific Interchange, Turin, Italy",
            "",
            "## Advisory Editors",
            "",
            "-",
            "### C. D. Schweitzer PhD",
            "",
            "ETH Zurich, Zurich, Switzerland",
            "",
            "-",
            "### D. E. Strohmaier PhD",
            "",
            "Complexity Science Hub Vienna, Vienna, Austria",
            "",
            "-",
            "### E. F. Vespignani PhD",
            "",
            "Northeastern University, Boston, United States",
        ]),
        [
            Editor(name="A. B. Garcia", affiliations=["University of Konstanz, Konstanz, Germany"], role=None),
            Editor(name="B. C. Mejova", affiliations=["Institute for Scientific Interchange, Turin, Italy"], role=None),
            Editor(name="C. D. Schweitzer", affiliations=["ETH Zurich, Zurich, Switzerland"], role=None),
            Editor(name="D. E. Strohmaier", affiliations=["Complexity Science Hub Vienna, Vienna, Austria"], role=None),
            Editor(name="E. F. Vespignani", affiliations=["Northeastern University, Boston, United States"], role=None),
        ],
    ),
]

# --- Tests ---

@pytest.mark.parametrize("fmt, snippet, expected", TEST_CASES)
def test_detect_format(fmt, snippet, expected):
    """Verify that the correct format is detected for each snippet."""
    if fmt in ("heading_name",):
        pytest.skip("detected at parse_editors level, not via detect_format")
    assert detect_format(snippet) == fmt

@pytest.mark.parametrize("fmt, snippet, expected", TEST_CASES)
def test_parse_editors_formats(fmt, snippet, expected):
    """Verify that the correct Editor objects are extracted for each format."""
    result = parse_editors(snippet)
    assert result == expected

# --- Edge cases --------------------------------------------------------------

def test_parse_editors_unknown_format():
    """Verify that unknown formats are skipped (return empty list)."""
    snippet = "This is some random text that doesn't match any format."
    assert parse_editors(snippet) == []

def test_parse_editors_empty_input():
    """Verify that empty input returns an empty list."""
    assert parse_editors("") == []
