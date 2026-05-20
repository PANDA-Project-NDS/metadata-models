#!/usr/bin/env python3
"""Generate HTML documentation for the JournalMetadata schema directly from Pydantic models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from journal import JournalMetadata


def inline_refs(schema: dict) -> dict:
    """Recursively inline all $ref references from $defs."""
    defs = schema.get("$defs", {})

    def _inline(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    resolved = _inline(defs[ref_name])
                    for k, v in obj.items():
                        if k != "$ref":
                            resolved[k] = v
                    return resolved
            return {k: _inline(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_inline(item) for item in obj]
        return obj

    result = _inline(schema)
    result.pop("$defs", None)
    return result


def type_badge(prop: dict | str) -> str:
    """Render a type badge for a JSON Schema property."""
    if isinstance(prop, str):
        return f'<span class="badge {prop}">{prop}</span>'
    if "enum" in prop:
        return '<span class="badge enum">enum</span> ' + ", ".join(
            f"<code>{v}</code>" for v in prop["enum"]
        )
    t = prop.get("type", "")
    if t == "string":
        return '<span class="badge string">string</span>'
    if t == "integer":
        return '<span class="badge number">integer</span>'
    if t == "number":
        return '<span class="badge number">number</span>'
    if t == "boolean":
        return '<span class="badge bool">boolean</span>'
    if t == "array":
        return '<span class="badge array">array</span>'
    if t == "object":
        return '<span class="badge object">object</span>'
    if t == "null":
        return '<span class="badge null">null</span>'
    return ""


def resolve_anyof(prop: dict) -> dict:
    """Extract the actual object type from anyOf [..., null]."""
    if "anyOf" in prop:
        for item in prop["anyOf"]:
            if isinstance(item, dict) and item.get("type") == "object":
                return item
        return prop
    return prop


def render_field(name: str, prop: dict, required: bool = False, depth: int = 0) -> str:
    """Render a single schema field as HTML."""
    desc = prop.get("description", "")
    default = prop.get("default")
    req_marker = '<span class="required">*</span>' if required else ""

    resolved = resolve_anyof(prop)
    is_object = resolved.get("type") == "object" or "properties" in resolved
    is_array = prop.get("type") == "array"
    nullable = "anyOf" in prop and any(
        isinstance(x, dict) and x.get("type") == "null" for x in prop["anyOf"]
    )

    # Build type string
    if is_array:
        items = resolved.get("items", {})
        if "enum" in items:
            type_str = (
                '<span class="badge array">array</span> of <span class="badge enum">enum</span> '
                + ", ".join(f"<code>{v}</code>" for v in items["enum"])
            )
        else:
            type_str = '<span class="badge array">array</span> of ' + type_badge(
                items.get("type", "object")
            )
    elif is_object:
        type_str = '<span class="badge object">object</span>'
    else:
        if "enum" in prop:
            type_str = type_badge(prop)
        elif "anyOf" in prop:
            types = []
            for item in prop["anyOf"]:
                if isinstance(item, dict) and item.get("type") != "null":
                    if "enum" in item:
                        types.append(type_badge(item))
                    else:
                        types.append(type_badge(item.get("type", "unknown")))
            type_str = " | ".join(types)
            if nullable:
                type_str += ' | <span class="badge null">null</span>'
        else:
            type_str = type_badge(prop.get("type", "unknown"))
            if nullable:
                type_str += ' | <span class="badge null">null</span>'

    html = f'<div class="field" style="margin-left:{depth * 20}px">\n'
    html += (
        f'  <div class="field-header">\n'
        f'    <span class="field-name"><code>{name}</code>{req_marker}</span>\n'
        f'    <span class="field-type">{type_str}</span>\n'
        f"  </div>\n"
    )
    if desc:
        html += f'  <div class="field-desc">{desc}</div>\n'
    if default is not None:
        html += f'  <div class="field-default">default: <code>{json.dumps(default)}</code></div>\n'

    # Nested object properties
    if is_object and "properties" in resolved:
        sub_required = resolved.get("required", [])
        html += '  <div class="field-children">\n'
        for sub_name, sub_prop in resolved["properties"].items():
            html += render_field(
                sub_name, sub_prop, sub_name in sub_required, depth + 1
            )
        html += "  </div>\n"

    # Array items with object structure
    if is_array and "properties" in resolved.get("items", {}):
        items = resolved["items"]
        sub_required = items.get("required", [])
        html += '  <div class="field-children"><div class="array-item-label">&lt;items&gt;</div>\n'
        for sub_name, sub_prop in items["properties"].items():
            html += render_field(
                sub_name, sub_prop, sub_name in sub_required, depth + 1
            )
        html += "  </div>\n"

    html += "</div>\n"
    return html


def main() -> None:
    schema = inline_refs(JournalMetadata.model_json_schema())
    props = schema.get("properties", {})

    fields_html = ""
    for name, prop in props.items():
        fields_html += render_field(name, prop)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JournalMetadata Schema</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f8f9fa;
    color: #24292e;
    line-height: 1.6;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
    background: #fff;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 40px;
  }}
  h1 {{
    font-size: 28px;
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 12px;
    margin-bottom: 8px;
  }}
  .schema-desc {{
    color: #586069;
    margin-bottom: 32px;
    font-size: 15px;
  }}
  .field {{
    padding: 12px 0;
    border-bottom: 1px solid #f0f0f0;
  }}
  .field:last-child {{ border-bottom: none; }}
  .field-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .field-name {{
    font-size: 16px;
    background: #f6f8fa;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid #e1e4e8;
  }}
  .field-type {{
    font-size: 13px;
    color: #586069;
  }}
  .field-desc {{
    margin-top: 4px;
    margin-left: 0;
    font-size: 14px;
    color: #586069;
  }}
  .field-default {{
    margin-top: 2px;
    font-size: 13px;
    color: #959da5;
  }}
  .field-children {{
    margin-top: 4px;
    border-left: 2px solid #e1e4e8;
    padding-left: 4px;
  }}
  .array-item-label {{
    font-size: 12px;
    color: #959da5;
    font-style: italic;
    padding: 4px 0;
  }}
  .required {{
    color: #d73a49;
    font-weight: bold;
  }}
  .badge {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 12px;
    font-weight: 600;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }}
  .badge.string {{ background: #dbedff; color: #0366d6; }}
  .badge.number {{ background: #e8f5e9; color: #2e7d32; }}
  .badge.bool {{ background: #fff3e0; color: #e65100; }}
  .badge.array {{ background: #f3e5f5; color: #7b1fa2; }}
  .badge.object {{ background: #eceff1; color: #455a64; }}
  .badge.enum {{ background: #fff8e1; color: #f57f17; }}
  .badge.null {{ background: #fafafa; color: #9e9e9e; }}
  code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12px;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>JournalMetadata</h1>
  <p class="schema-desc">{schema.get("description", "").replace(chr(10), " ")}</p>
{fields_html}</div>
</body>
</html>"""

    out = Path(__file__).resolve().parent.parent / "metadata_schema.html"
    with open(out, "w") as f:
        f.write(html)

    print(f"Written {out}")


if __name__ == "__main__":
    main()
