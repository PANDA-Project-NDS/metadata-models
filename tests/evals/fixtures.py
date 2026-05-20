"""Synthetic journal context snippets and helpers for eval tests."""


def is_empty(obj):
    """Treats None, empty lists, and all-subfields-empty containers as empty."""
    if obj is None:
        return True
    if isinstance(obj, (list, tuple, set, dict)):
        return len(obj) == 0
    if hasattr(obj, "model_dump"):  # Handle Pydantic v2 models
        return all(is_empty(v) for v in obj.model_dump().values())
    if hasattr(obj, "__dict__"):  # Fallback for standard objects
        return all(is_empty(v) for v in obj.__dict__.values())
    return False


ISSN_CONTEXT = """
Journal of Artificial Intelligence Research
Print ISSN: 1234-5678 | Online ISSN: 9876-5432
Publisher: JAIR Press
"""

ISSN_PRINT_ONLY_CONTEXT = """
Journal of Testing
Print ISSN: 1234-5678
"""

ISSN_BOTH_CONTEXT = """
Journal of Dual ISSN
Print ISSN: 1234-5678
Online ISSN: 9876-5432
"""

ISSN_FORMAT_VARIATIONS_CONTEXT = """
Journal Identifiers:
Print ISSN: 12345678
Online ISSN: eISSN 9876-5432
"""

ISSN_LINKING_CONTEXT = """
The journal is identified by Print ISSN 1234-5678, Online ISSN 9876-5432, and Linking ISSN (ISSN-L) 2468-1357.
"""

APC_SINGLE_CONTEXT = """
Article Processing Charge:
Research articles: $2000 USD
"""

APC_DOUBLE_CONTEXT = """
Article Processing Charges:
- Research articles: $2000 USD
- Short papers: $1500 USD
Waivers available for authors from low-income countries.
"""

APC_CURRENCY_MIXED_CONTEXT = """
Fees:
- Research articles: $2000 
- Review articles: 1500 €
"""

APC_DISCOUNTS_COMPLEX_CONTEXT = """
Waivers and Discounts:
- We provide a full waiver for authors from low-income countries.
- Members of the International Science Society receive a fixed discount of 500 EUR.
- Early-bird submissions receive a 15% discount on the total APC.
"""

EDITORIAL_CONTEXT = """
Editorial Board:
Editor-in-Chief: Dr. Jane Smith, MIT
Managing Editor: Dr. John Doe, Stanford
"""

POLICIES_CONTEXT = """
This journal uses a double-blind peer review process.
Publication frequency: Monthly, 12 issues per year.
Accepted article types: research article, review article, letter.
Languages accepted: eng.
"""

IRRELEVANT_CONTEXT = """
Contact the journal
The fastest way to find answers to questions about our journals and submissions is usually to search in our support portal.
You can track the progress of your submission in the 'Your research' section of your account.
If you cannot find the answer you need in our support portal or your account, you can request support.
"""
