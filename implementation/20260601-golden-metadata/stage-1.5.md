# Stage 1.5 — Utility Functions (`golden/lib/flatten.py`)

**Goal**: Extract shared utility functions into `golden/lib/flatten.py` so both `golden/agents/golden.py` and `golden/main.py` can import them.

**Depends on**: Stage 1

## Files to Create/Modify

| File | Action |
|---|---|
| `golden/lib/__init__.py` | **Create** — empty |
| `golden/lib/flatten.py` | **Create** — all flatten, merge, path, strip utilities |
| `golden/lib/_sourced.py` | **Create** — `SourcedValue`/`SourcedModel` type detection helpers |

## `golden/lib/__init__.py`

Empty file to make `lib` a package.

## `golden/lib/flatten.py`

```python
from typing import Any, Generator
from dataclasses import dataclass

from golden.lib._sourced import _sourced_value_paths, _sourced_model_paths

@dataclass(frozen=True)
class FlattenedField:
    """A single leaf field with its path and evidence."""
    path: str
    value: Any
    evidence: str | None

MAX_ROUNDS = 1
MAX_CORRECTION_ROUNDS = 2


def flatten_metadata(metadata) -> Generator[FlattenedField, None, None]:
    """Recursively walk metadata to (path, value, evidence) triples.

    Walks the model dump (dict), yielding FlattenedField objects
    for every leaf field.  SourcedModel evidence propagates to all child leaf
    fields.  SourcedValue fields yield the .value and .evidence directly.

    Uses schema-aware path sets (_sourced_value_paths, _sourced_model_paths)
    to detect SourcedValue/SourcedModel wrappers — no key-name heuristics.
    """
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
) -> Generator[FlattenedField, None, None]:
    """Recursively flatten a dict into FlattenedField objects.

    Uses schema-derived path sets to identify SourcedValue and SourcedModel
    wrappers in the JSON dump, avoiding fragile key-name heuristics.
    """
    if isinstance(obj, dict):
        # SourcedValue wrapper: {"value": ..., "evidence": ...}
        if prefix in sv_paths:
            val = obj.get("value")
            ev = obj.get("evidence")
            ev_str = _format_evidence(ev)
            if isinstance(val, (dict, list)):
                yield from _flatten_dict(val, prefix, sv_paths, sm_paths, ev_str)
            else:
                yield FlattenedField(prefix or "(root)", val, ev_str)
            return

        # SourcedModel: dict whose fields include "evidence" + other data
        if prefix in sm_paths and "evidence" in obj:
            ev = obj["evidence"]
            ev_str = _format_evidence(ev) or inherited_evidence
            rest = {k: v for k, v in obj.items() if k != "evidence"}
            for k, v in rest.items():
                yield from _flatten_dict(
                    v, f"{prefix}.{k}", sv_paths, sm_paths, ev_str
                )
            return

        # Regular dict
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
    """Format an Evidence dict into 'quote [source]' string."""
    if isinstance(ev, dict) and ev:
        return f"{ev.get('quote', '')} [{ev.get('source', '')}]"
    return ev if isinstance(ev, str) else None


def _split_path(path: str) -> list[str]:
    """Split a dotted path while ignoring dots inside brackets.
    'pricing.apcs[0].value' -> ['pricing', 'apcs[0]', 'value']
    """
    return re.split(r'\.(?![^\[]*\])', path)


def get_path(obj: Any, path: str) -> Any:
    """Get a value from a nested dict/list using dotted bracket path.

    'pricing.article_processing_charges[0].fee.value' navigates the structure.
    """
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
    """Set a leaf value in a nested dict using dotted bracket path.

    Handles bracket notation for list indices, e.g.
    'pricing.article_processing_charges[0].fee.value'.
    """
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
    # Set the final part
    last = parts[-1]
    m = re.match(r"^(.+)\[(\d+)\]$", last)
    if m:
        key, idx = m.group(1), int(m.group(2))
        if key not in current or not isinstance(current[key], list):
            current[key] = [None] * (idx + 1)
        current[key][idx] = value
    else:
        current[last] = value


def merge_partial(draft_dump: dict, patch_dump: dict, dedup_key: str | None = None, force: bool = False) -> dict:
    """Deep merge *patch_dump* into *draft_dump* in-place.

    - `None` values in patch are skipped
    - Scalar: only set if draft is `None` (or if `force=True`)
    - List: extend draft with patch items (deduplicated if *dedup_key* is given)
    - Dict/model: recurse

    **Dedup strategy**: When merging lists of model dicts (e.g. editors, APCs),
    pass ``dedup_key`` to avoid duplicates if the completeness agent re-emits
    existing items.  Each candidate item is compared against existing items by
    the given key (e.g. ``"name"`` for editors, ``"article_type"`` for APCs).
    Items missing the key are always kept.  For lists without a natural key,
    omit *dedup_key* and accept the risk of duplicates.

    Example::

        merge_partial(draft, patch, dedup_key="name")  # editors
        merge_partial(draft, patch, dedup_key="article_type")  # APCs
        merge_partial(draft, patch, force=True)  # Correcting errors
        merge_partial(draft, patch)  # no dedup (e.g. languages[str])
    """
    for k, v in patch_dump.items():
        if v is None:
            continue
        if k not in draft_dump or draft_dump[k] is None or force:
            draft_dump[k] = v
        elif isinstance(draft_dump[k], list) and isinstance(v, list):
            if dedup_key and v and isinstance(v[0], dict):
                # Deduplicate: only extend with items whose dedup_key isn't
                # already represented in the draft list.
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
    """Recursively remove 'evidence' keys from model_dump output."""
    if isinstance(obj, dict):
        return {k: strip_evidence(v) for k, v in obj.items() if k != "evidence"}
    if isinstance(obj, list):
        return [strip_evidence(item) for item in obj]
    return obj
```

## `golden/lib/_sourced.py`

Helper module for SourcedValue/SourcedModel type detection. Uses `get_origin()`/`get_args()` to unwrap `Optional`/`Union` wrappers.

```python
from typing import get_origin, get_args, Callable
from typing import Type

from models.journal import SourcedValue, SourcedModel


def issubclass_safe(cls, base) -> bool:
    """issubclass that doesn't crash on non-class types."""
    try:
        return isinstance(cls, type) and issubclass(cls, base)
    except TypeError:
        return False


def _is_sourced_value(annotation: type) -> bool:
    """Check if a type annotation is SourcedValue[T], unwrapping Optional/Union."""
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            if _is_sourced_value(arg):
                return True
        return False
    return issubclass_safe(annotation, SourcedValue)


def _is_sourced_model(annotation: type) -> bool:
    """Check if a type annotation is a SourcedModel subclass, unwrapping Optional/Union."""
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            if _is_sourced_model(arg):
                return True
        return False
    return issubclass_safe(annotation, SourcedModel)


def _find_sourced_paths(model: type, predicate: Callable[[type], bool]) -> set[str]:
    """Walk a Pydantic model schema to find all field paths matching the predicate.
    
    Used to identify SourcedValue and SourcedModel wrappers in the schema.
    """
    paths: set[str] = set()
    if not hasattr(model, "model_fields"):
        return paths
    for name, field_info in model.model_fields.items():
        ann = field_info.annotation
        if predicate(ann):
            paths.add(name)
        elif hasattr(ann, "model_fields"):
            for sub in _find_sourced_paths(ann, predicate):
                paths.add(f"{name}.{sub}")
        else:
            # Unwrap Optional[SomeModel] to check inner type
            origin = get_origin(ann)
            if origin is not None:
                for arg in get_args(ann):
                    if hasattr(arg, "model_fields"):
                        for sub in _find_sourced_paths(arg, predicate):
                            paths.add(f"{name}.{sub}")
    return paths


def _sourced_value_paths(model: type) -> set[str]:
    """Walk a Pydantic model schema to find all SourcedValue-wrapped field paths."""
    return _find_sourced_paths(model, _is_sourced_value)


def _sourced_model_paths(model: type) -> set[str]:
    """Walk a Pydantic model schema to find all SourcedModel-wrapped field paths."""
    return _find_sourced_paths(model, _is_sourced_model)
```

## Acceptance Criteria

- `from golden.lib.flatten import flatten_metadata, merge_partial, get_path, set_path, strip_evidence, MAX_ROUNDS, MAX_CORRECTION_ROUNDS` works
- `_sourced_value_paths` and `_sourced_model_paths` correctly unwrap `Optional`/`Union` wrappers
- `flatten_metadata` uses schema-aware paths, not key-name heuristics
- `merge_partial(dedup_key="name")` deduplicates editor lists
