from typing import Literal

type ReviewType = Literal["single-blind", "double-blind", "open-review"]
type SupportedCurrency = Literal["USD", "EUR", "CHF", "GBP", "JPY", "CAD"]
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
type IndexingService = Literal[
    "Scopus",
    "Web of Science",
    "Google Scholar",
    "Dimensions",
    "PubMed",
    "MEDLINE",
    "Engineering Village",
    "IEEE Xplore",
    "Chemical Abstracts Service",
    "PsycINFO",
    "ERIC",
    "DOAJ",
    "Sherpa Romeo",
    "OpenAlex",
    "CrossRef",
]
type MembershipType = Literal["society", "institutional", "individual", "corporate"]
type DiscountType = Literal["waiver", "fixed", "percent"]
