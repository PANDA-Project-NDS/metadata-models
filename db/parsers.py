import json

import trafilatura
from llama_index.core import Document

EXCEL_METADATA_FIELDS = [
    ("Journal", "title"),
    ("Subject", "subject_area"),
    ("ISSN", "online_issn"),
    ("Open access type", "open_access_type"),
    ("License", "license_types_offered"),
    ("APC", "full_price"),
    ("Blocked", "blocked"),
]


def _serialize_excel_doc(db_doc: dict, collection_name: str) -> Document:
    """Serialize a non-HTML (Excel/APC) document into a Document for embedding."""
    m = db_doc.get("metadata", {})
    parts = []
    for label, key in EXCEL_METADATA_FIELDS:
        val = m.get(key)
        if val is not None and val != "":
            parts.append(f"{label}: {json.dumps(val)}")
    header = m.get("header_footer")
    if header:
        parts.append(header)
    return Document(
        text="\n".join(parts),
        metadata={
            "source_uri": m.get("url", "unknown"),
            "journal_id": m.get("title", "unknown"),
            "publisher": collection_name,
            "scope": "excel",
        },
        excluded_embed_metadata_keys=["source_uri", "journal_id", "publisher", "scope"],
    )


def _serialize_html_doc(db_doc: dict, collection_name: str) -> Document | None:
    """Extract HTML content and serialize into a Document for embedding."""
    metadata = db_doc.get("metadata", {})
    html_content = metadata.get("html", "")
    if not html_content:
        return None

    extracted_text = trafilatura.extract(html_content)
    if not extracted_text:
        extracted_text = html_content

    source_url = metadata.get("url", "unknown")
    journal_id = metadata.get("title", "unknown")

    return Document(
        text=extracted_text,
        metadata={
            "source_uri": source_url,
            "journal_id": journal_id,
            "publisher": collection_name,
            "scope": "html",
        },
        excluded_embed_metadata_keys=[
            "source_uri",
            "journal_id",
            "publisher",
            "scope",
        ],
    )
