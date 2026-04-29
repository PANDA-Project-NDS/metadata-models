from typing import Literal

ReviewType = Literal["single-blind", "double-blind", "open-review"]
SupportedCurrency = Literal["USD", "EUR"]
Frequency = Literal[
    "Annually",
    "Semiannually",
    "Quarterly",
    "Monthly",
    "Biweekly",
    "Weekly",
    "Daily",
    "Irregular",
]
ArticleTypeValue = Literal[
    "Research Article",
    "Review Article",
    "Brief Communication",
    "Case Study",
    "Perspective",
    "Editorial",
    "Mini-Review",
    "Technical Note",
    "Conference Paper",
    "Protocol",
    "Meta-Analysis",
    "Invited Paper",
    "Comment",
]
IndexingService = Literal[
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