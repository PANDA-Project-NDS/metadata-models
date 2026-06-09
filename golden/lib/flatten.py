import re
import typing
from typing import Any, Type, get_origin, get_args, Union
from dataclasses import dataclass

from golden.lib._sourced import (
    _sourced_value_paths, 
    _sourced_model_paths, 
    _is_sourced_value, 
    _is_sourced_model
)
from pydantic import BaseModel


@dataclass(frozen=True)
class FlattenedField:
    path: str
    value: Any
    evidence: str | None

MAX_ROUNDS = 1
MAX_CORRECTION_ROUNDS = 2


def flatten_metadata(metadata):
    sv_paths = _sourced_value_paths(type(metadata))
    sm_paths = _sourced_model_paths(type(metadata))
    yield from _flatten_dict(
        metadata.model_dump(mode="json"), "", sv_paths, sm_paths, None
    )


def _flatten_dict(
    obj: Any,
    prefix: str,
    sv_paths: set[str],
    sm_paths: set[str],
    inherited_evidence: str | None,
):
    if isinstance(obj, dict):
        if prefix in sv_paths:
            val = obj.get("value")
            ev = obj.get("evidence")
            ev_str = _format_evidence(ev)
            if isinstance(val, (dict, list)):
                yield from _flatten_dict(val, prefix, sv_paths, sm_paths, ev_str)
            else:
                yield FlattenedField(prefix or "(root)", val, ev_str)
            return

        if prefix in sm_paths and "evidence" in obj:
            ev = obj["evidence"]
            ev_str = _format_evidence(ev) or inherited_evidence
            rest = {k: v for k, v in obj.items() if k != "evidence"}
            for k, v in rest.items():
                yield from _flatten_dict(
                    v, f"{prefix}.{k}", sv_paths, sm_paths, ev_str
                )
            return

        for k, v in obj.items():
            yield from _flatten_dict(
                v, f"{prefix}.{k}" if prefix else k, sv_paths, sm_paths,
                inherited_evidence
            )
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _flatten_dict(
                item, f"{prefix}[{i}]", sv_paths, sm_paths,
                inherited_evidence
            )
    else:
        yield FlattenedField(prefix, obj, inherited_evidence)


def _format_evidence(ev) -> str | None:
    if isinstance(ev, dict) and ev:
        return f"{ev.get('quote', '')} [{ev.get('source', '')}]"
    return ev if isinstance(ev, str) else None


def _split_path(path: str) -> list[str]:
    return re.split(r'\.(?![^\[]*\])', path)


def get_path(obj: Any, path: str) -> Any:
    current = obj
    for part in _split_path(path):
        if current is None:
            return None
        m = re.match(r"^(.+)\[(\d+)\]$", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if key:
                current = current.get(key) if isinstance(current, dict) else None
            if isinstance(current, list) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            current = current.get(part) if isinstance(current, dict) else None
    return current


def set_path(obj: dict, path: str, value: Any):
    parts = _split_path(path)
    current = obj
    for i, part in enumerate(parts[:-1]):
        m = re.match(r"^(.+)\[(\d+)\]$", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if key:
                current = current.setdefault(key, {})
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            elif isinstance(current, dict) and idx in current:
                current = current[idx]
            else:
                current = {}
        else:
            current = current.setdefault(part, {})
    last = parts[-1]
    m = re.match(r"^(.+)\[(\d+)\]$", last)
    if m:
        key, idx = m.group(1), int(m.group(2))
        existing = current.get(key) if key in current else None
        if not isinstance(existing, list) or len(existing) <= idx:
            current[key] = (existing or []) + [None] * (idx + 1 - len(existing or []))
        current[key][idx] = value
    else:
        current[last] = value


def merge_partial(draft_dump: dict, patch_dump: dict, dedup_key: str | None = None, force: bool = False) -> dict:
    for k, v in patch_dump.items():
        if v is None:
            continue
        if k not in draft_dump or draft_dump[k] is None or force:
            draft_dump[k] = v
        elif isinstance(draft_dump[k], list) and isinstance(v, list):
            if dedup_key and v and isinstance(v[0], dict):
                existing = {
                    item.get(dedup_key)
                    for item in draft_dump[k]
                    if isinstance(item, dict)
                }
                new_items = [
                    item for item in v
                    if item.get(dedup_key) not in existing
                ]
                draft_dump[k].extend(new_items)
            else:
                draft_dump[k].extend(v)
        elif isinstance(draft_dump[k], dict) and isinstance(v, dict):
            merge_partial(draft_dump[k], v, dedup_key=dedup_key, force=force)
    return draft_dump


def strip_evidence(obj):
    if isinstance(obj, dict):
        return {k: strip_evidence(v) for k, v in obj.items() if k != "evidence"}
    if isinstance(obj, list):
        return [strip_evidence(item) for item in obj]
    return obj


def get_slimmer_schema(model: Type[BaseModel]) -> str:
    """
    Flattens a Pydantic model into a concise list of field paths and descriptions.
    Paths follow the 'parent.child' format used by get_path/set_path.
    """
    lines = []

    def unwrap(ann):
        """Unwrap Optional and Union to get the inner type."""
        origin = get_origin(ann)
        if origin is None:
            return ann
        if origin is list:
            return ann
        if origin is Union or (hasattr(typing, "UnionType") and isinstance(ann, typing.UnionType)):
            args = get_args(ann)
            # Filter out NoneType
            non_none = [a for a in args if a is not type(None)]
            return non_none[0] if non_none else ann
        return ann

    def walk(m: Type[BaseModel], prefix: str):
        if not hasattr(m, "model_fields"):
            return

        for field_name, field_info in m.model_fields.items():
            ann = field_info.annotation
            unwrapped = unwrap(ann)
            
            # Check if the field is a Sourced wrapper
            is_sv = _is_sourced_value(unwrapped)
            is_sm = _is_sourced_model(unwrapped)
            
            path = f"{prefix}.{field_name}" if prefix else field_name
            
            if is_sv or is_sm:
                # If it's a SourcedValue[T], we want to recurse into T if T is a model
                origin = get_origin(unwrapped)
                if origin is not None:
                    # It's a Generic (like SourcedValue[T])
                    for arg in get_args(unwrapped):
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            walk(arg, path)
                            continue
                
                # If it's just a SourcedValue (scalar) or we didn't recurse
                # We only add as leaf if it's not a nested model we already walked
                # To avoid duplicates, we'll use a flag or just check if we walked
                # Actually, if it's a SourcedValue[str], it's a leaf.
                if not (origin is not None and any(isinstance(a, type) and issubclass(a, BaseModel) for a in get_args(unwrapped))):
                    description = field_info.description or "No description provided."
                    lines.append(f"- {path}: {description}")
                continue

            # Handle Lists
            origin = get_origin(unwrapped)
            if origin is list:
                inner_type = get_args(unwrapped)[0]
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    walk(inner_type, path)
                    continue
                description = field_info.description or "No description provided."
                lines.append(f"- {path}: {description}")
                continue

            # Handle Nested Models
            if isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel):
                walk(unwrapped, path)
                continue

            # Leaf Node
            description = field_info.description or "No description provided."
            lines.append(f"- {path}: {description}")

    walk(model, "")
    return "\n".join(lines)
