from dataclasses import dataclass

from pydantic import BaseModel

from models.journal import (
    INCLUDE_EVIDENCE,
    BasicInfoExtraction,
    EditorialExtraction,
    FeesExtraction,
    PoliciesExtraction,
)


ISSN_RULES = """
## DOMAIN-SPECIFIC EXTRACTION RULES
### IDENTIFIERS (ISSN)
   - NORMALIZATION: Always output ISSNs in the 'NNNN-NNNN' format.
   - CLEANING: Remove prefixes such as 'eISSN:', 'pISSN:', or 'ISSN:'.
   - EXAMPLE: "eISSN 12345678" -> "1234-5678".
"""

MONETARY_RULES = """
## DOMAIN-SPECIFIC EXTRACTION RULES
### MONETARY VALUES (APC & Discounts)
   - CURRENCY MAPPING: Map symbols to ISO codes: '$' or 'USD' -> 'USD', '€' or 'EUR' -> 'EUR'.
   - PRICE RANGES: If a range is provided (e.g., "$1000 - $2000"), extract the MAXIMUM value.
   - ROUNDING: All monetary values must be integers. Round decimals to the nearest whole number.
   - DISCOUNTS:
     - 'Waiver' (full or partial) -> type: "waiver".
     - Percentage (e.g., "10% off") -> type: "percent", percentage: 10.0.
     - Fixed Amount (e.g., "500 EUR discount") -> type: "fixed", amount: {value: 500, currency: "EUR"}.
"""

EVIDENCE_RULES = """
### EVIDENCE & ELIGIBILITY
   - ELIGIBILITY: For discounts/waivers, the 'eligibility' field must be a verbatim quote of the criteria.
   - PRECISION: If a value is "approximately" or "about" a certain amount, extract the number and note the approximation in the quote.
"""


@dataclass
class PassConfig:
    """RAG Extraction pass config"""

    name: str
    output_type: type[BaseModel]
    queries: list[str]
    domain_guidelines: str = ""


PASSES: list[PassConfig] = [
    PassConfig(
        "Info Agent",
        BasicInfoExtraction,
        [
            "Journal title, publisher, about this journal, mission, scope, sections",
            "ISSN, print ISSN, online ISSN, indexed in, abstracting and indexing databases",
            "Impact factor, journal metrics, citation score, cite score",
        ],
        domain_guidelines=ISSN_RULES,
    ),
    PassConfig(
        "Policies Agent",
        PoliciesExtraction,
        [
            "Publication frequency, issues per year, submission guidelines, author instructions, article types accepted",
            "Peer review process, blind review, open access policy statement, copyright, quality assurance",
            "diamond open access, community owned, open to all authors",
            "publication languages, languages accepted",
        ],
    ),
    PassConfig(
        "Fees Agent",
        FeesExtraction,
        [
            "Article Processing Charge, APC, publication fees, cost, waivers, discounts, society membership, institutional membership",
        ],
        domain_guidelines=MONETARY_RULES + (EVIDENCE_RULES if INCLUDE_EVIDENCE else ""),
    ),
    PassConfig(
        "Editors Agent",
        EditorialExtraction,
        [
            "Editorial board, Editor in Chief, managing editor, editorial team",
        ],
    ),
]
