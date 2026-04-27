from typing import Literal

ReviewTypeLiteral = Literal["single-blind", "double-blind", "open-review"]
SupportedCurrencyLiteral = Literal["USD", "EUR"]
FrequencyLiteral = Literal[
    "Annually",
    "Semiannually",
    "Quarterly",
    "Monthly",
    "Biweekly",
    "Weekly",
    "Daily",
    "Irregular",
]

ArticleTypeLiteral = Literal[
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

EditorRoleLiteral = Literal[
    "Editor-in-Chief",
    "Editor",
    "Associate Editor",
    "Section Editor",
    "Editorial Board Member",
    "Past Editor-in-Chief",
    "Honorary Editor",
    "Advisory Editor",
    "Guest Editor",
    "Deputy Editor",
    "Assistant Editor",
]