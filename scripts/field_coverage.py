"""Report field coverage percentages for the journal_metadata MongoDB collection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from db import MongoDBManager


def is_filled(value) -> bool:
    """Return True if a value is meaningfully present."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def _is_sourced_value(obj: dict) -> bool:
    """A SourcedValue has exactly two keys: 'value' and 'evidence'."""
    return set(obj.keys()) == {"value", "evidence"}


def flatten_doc(doc: dict, prefix: str = "") -> dict[str, bool]:
    """Flatten a journal_metadata document to leaf paths with filled status."""
    result = {}
    for key, val in doc.items():
        if key in ("evidence", "_id"):
            continue
        path = f"{prefix}.{key}" if prefix else key
        if val is None:
            continue
        if isinstance(val, dict):
            if _is_sourced_value(val):
                # Only track the "value" leaf, skip evidence sub-fields
                result[f"{path}.value"] = is_filled(val.get("value"))
            else:
                sub = flatten_doc(val, path)
                result.update(sub)
        elif isinstance(val, list):
            result[path] = is_filled(val)
            # Drill into first element to get nested field paths
            if len(val) > 0 and isinstance(val[0], dict):
                nested = flatten_doc(val[0], path)
                result.update(nested)
        else:
            # Leaf primitive (str, int, float, bool)
            result[path] = is_filled(val)
    return result


def pct_color(pct: float) -> str:
    """Return a background color for a given percentage."""
    if pct >= 90:
        return "#2d6a4f"
    if pct >= 70:
        return "#52b788"
    if pct >= 50:
        return "#95d5b2"
    if pct >= 30:
        return "#d8f3dc"
    if pct >= 10:
        return "#fde5cf"
    return "#fca311"


def main():
    load_dotenv()
    manager = MongoDBManager(os.environ["MONGODB_URI"])
    collection = manager.get_collection(os.environ["MONGO_METADATA_COLLECTION"])

    total = collection.count_documents({})
    if total == 0:
        print("No documents found in journal_metadata.")
        return

    # Accumulate: field_path -> count of filled
    field_counts: dict[str, int] = {}
    for doc in collection.find():
        leafs = flatten_doc(doc)
        for path, filled in leafs.items():
            field_counts[path] = field_counts.get(path, 0) + (1 if filled else 0)

    # Build sorted rows
    rows = []
    for path, count in field_counts.items():
        pct = count / total * 100
        rows.append((path, count, pct))

    rows.sort(key=lambda r: r[0])

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: monospace; margin: 20px; }",
        "table { border-collapse: collapse; }",
        "th, td { padding: 6px 12px; border: 1px solid #ccc; text-align: left; }",
        "th { background: #333; color: #fff; }",
        ".pct { text-align: center; font-weight: bold; }",
        "</style></head><body>",
        f"<h2>Field Coverage — {total} documents</h2>",
        "<table><tr><th>Field</th><th>Count</th><th>Percentage</th></tr>",
    ]
    for path, count, pct in rows:
        bg = pct_color(pct)
        html_parts.append(
            f"<tr><td>{path}</td><td>{count}</td>"
            f"<td class='pct' style='background:{bg}'>{pct:.1f}%</td></tr>"
        )
    html_parts.append("</table></body></html>")

    html = "\n".join(html_parts)
    output_path = "field_coverage.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
