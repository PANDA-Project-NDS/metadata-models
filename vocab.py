from typing import Literal

publisher_examples = [
    "American Chemical Society",
    "Copernicus Publications",
    "Elsevier",
    "Emerald Publishing",
    "Frontiers Media S.A.",
    "IEEE",
    "MDPI AG",
    "SAGE Publishing",
    "Springer Nature",
    "Taylor & Francis",
    "Wiley",
]

type ReviewType = Literal["single-blind", "double-blind", "open-review"]
type SupportedCurrency = Literal["USD", "EUR", "CHF", "GBP", "JPY", "CAD", "AUD"]
type Frequency = Literal[
    "Annually",
    "Semiannually",
    "Quarterly",
    "Monthly",
    "Biweekly",
    "Weekly",
    "Daily",
    "Irregular",
]

# COARS Resource Type leaf nodes under "text" (c_18cf)
# Source: example_schema/resource_types_for_dspace_en.xml
# Regenerate with: python scripts/extract_coars.py
type ArticleTypeValue = Literal[
    "annotation",
    "bachelor thesis",
    "bibliography",
    "blog post",
    "book part",
    "book review",
    "clinical study",
    "commentary",
    "conference paper",
    "conference paper not in proceedings",
    "conference poster",
    "conference poster not in proceedings",
    "conference presentation",
    "corrigendum",
    "data management plan",
    "data paper",
    "doctoral thesis",
    "editorial",
    "knowledge synthesis protocol",
    "lecture",
    "letter",
    "letter to the editor",
    "magazine article",
    "manuscript",
    "master thesis",
    "memorandum",
    "musical notation",
    "newspaper article",
    "other periodical",
    "peer review",
    "policy report",
    "preprint",
    "project deliverable",
    "research article",
    "research proposal",
    "research protocol",
    "research report",
    "review article",
    "software paper",
    "technical documentation",
    "technical report",
    "transcription",
    "working paper",
]
# TODO: how many to list? Journals mention more.
type IndexingService = Literal[
    "Scopus",
    "Web of Science",
    "Google Scholar",
    "Dimensions",
    "PubMed",
    "MEDLINE",
    "Engineering Village",
    "IEEE Xplore",
    "Chemical Abstracts Service (CAS)",
    "PsycINFO",
    "Education Resources Information Center (ERIC)",
    "Directory of Open Access Journals (DOAJ)",
    "Sherpa Romeo",
    "OpenAlex",
    "CrossRef",
    "SCImago Journal Rank (SJR)",
    "Ei Compendex",
    "Cumulative Index to Nursing and Allied Health Literature (CINAHL)",
    "Embase",
]
type MembershipType = Literal["society", "institutional", "individual", "corporate"]
type DiscountType = Literal["waiver", "fixed", "percent", "unspecified"]
type LicenseType = Literal[
    "CC0",
    "CC BY",
    "CC BY-SA",
    "CC BY-ND",
    "CC BY-NC",
    "CC BY-NC-SA",
    "CC BY-NC-ND",
    "other",
]
