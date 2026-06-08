# Testing Strategy — Golden Metadata

This document defines the required test cases for the non-agent logic in the golden metadata pipeline.
Tests live in `golden/tests/` and run via `uv run pytest golden/tests`.

## 1. Path Navigation (`get_path` / `set_path`)

### Test Cases
- **Basic Access**
    - **Input**: `data={"name": "Journal A"}`, `path="name"`
    - **Expectation**: returns `"Journal A"`
    - **Assertion**: `assert get_path(data, "name") == "Journal A"`
- **Deep Nesting**
    - **Input**: `data={"a": {"b": {"c": 1}}}`, `path="a.b.c"`
    - **Expectation**: returns `1`
    - **Assertion**: `assert get_path(data, "a.b.c") == 1`
- **List Indexing**
    - **Input**: `data={"pricing": [{"fee": 100}, {"fee": 200}]}`, `path="pricing[1].fee"`
    - **Expectation**: returns `200`
    - **Assertion**: `assert get_path(data, "pricing[1].fee") == 200`
- **Boundary Conditions (Get)**
    - **Input**: `data={}`, `path="a.b"`
    - **Expectation**: returns `None`
    - **Assertion**: `assert get_path(data, "a.b") is None`
- **Boundary Conditions (Set - Missing Keys)**
    - **Input**: `data={}`, `path="a.b.c"`, `value=10`
    - **Expectation**: `data` becomes `{"a": {"b": {"c": 10}}}`
    - **Assertion**: `assert data == {"a": {"b": {"c": 10}}}`
- **Boundary Conditions (Set - List Expansion)**
    - **Input**: `data={"list": [1]}`, `path="list[2]"`, `value=3`
    - **Expectation**: `data` becomes `{"list": [1, None, 3]}` (or padded with None)
    - **Assertion**: `assert data["list"][2] == 3`

## 2. Schema Flattening (`flatten_metadata`)

### Test Cases
- **SourcedValue Propagation**
    - **Input**: `SourcedValue(value="Open Access", evidence="The journal is Open Access")` at path `access_type`
    - **Expectation**: Triple `("access_type", "Open Access", "The journal is Open Access")`
    - **Assertion**: `assert ("access_type", "Open Access", "...") in flatten_metadata(model)`
- **SourcedModel Propagation**
    - **Input**: `SourcedModel(evidence="Global Quote", field=SourcedValue(value="X", evidence="Local Quote"))`
    - **Expectation**: `field` uses `"Local Quote"` (more specific). If `field` was just a scalar, it would use `"Global Quote"`.
    - **Assertion**: `assert result_triple[2] == "Local Quote"`
- **Null Handling**
    - **Input**: `SourcedValue(value=None, evidence="No info found")`
    - **Expectation**: Triple `("path", None, "No info found")`
    - **Assertion**: `assert (path, None, "...") in flatten_metadata(model)`

## 3. Deep Merging (`merge_partial`)

### Test Cases
- **Scalar First-Win**
    - **Input**: `base={"a": 1}`, `patch={"a": 2}`
    - **Expectation**: `{"a": 1}`
    - **Assertion**: `assert merge_partial(base, patch)["a"] == 1`
- **Scalar Fill**
    - **Input**: `base={"a": None}`, `patch={"a": 2}`
    - **Expectation**: `{"a": 2}`
    - **Assertion**: `assert merge_partial(base, patch)["a"] == 2`
- **Scalar Overwrite (Force)**
    - **Input**: `base={"a": 1}`, `patch={"a": 2}`, `force=True`
    - **Expectation**: `{"a": 2}`
    - **Assertion**: `assert merge_partial(base, patch, force=True)["a"] == 2`
- **Recursive Object Merge**
    - **Input**: `base={"meta": {"a": 1}}`, `patch={"meta": {"b": 2}}`
    - **Expectation**: `{"meta": {"a": 1, "b": 2}}`
    - **Assertion**: `assert merge_partial(base, patch)["meta"] == {"a": 1, "b": 2}`
- **List Deduplication (Recursive)**
    - **Input**: 
        - `base={"pricing": [{"type": "APC", "val": 100}]}`
        - `patch={"pricing": [{"type": "APC", "val": 200}, {"type": "Page", "val": 10}]}`
        - `dedup_key={"pricing": "type"}`
    - **Expectation**: `{"pricing": [{"type": "APC", "val": 100}, {"type": "Page", "val": 10}]}`
    - **Assertion**: `assert len(result["pricing"]) == 2` and `result["pricing"][0]["val"] == 100`

## 4. Sourced Type Detection (`_sourced.py`)

### Test Cases
- **Type Discovery**
    - **Input**: Model with `field: SourcedValue[str]` and `nested: SourcedModel[NestedModel]`
    - **Expectation**: `_sourced_value_paths` contains `field`, `_sourced_model_paths` contains `nested`.
    - **Assertion**: `assert "field" in sourced_value_paths`
- **Wrapper Handling**
    - **Input**: `field: Optional[SourcedValue[str]]`
    - **Expectation**: Correctly identified as a sourced value path.
    - **Assertion**: `assert "field" in sourced_value_paths`

## 5. Utility & Script Helpers

### Test Cases
- **Evidence Stripping**
    - **Input**: `{"val": 1, "evidence": "...", "nested": {"val": 2, "evidence": "..."}}`
    - **Expectation**: `{"val": 1, "nested": {"val": 2}}`
    - **Assertion**: `assert "evidence" not in strip_evidence(data)` and `assert "evidence" not in strip_evidence(data)["nested"]`
- **Coverage Parsing**
    - **Input**: Markdown with `## Publisher A\nContent A\n## Publisher B\nContent B`
    - **Expectation**: `{"Publisher A": "Content A", "Publisher B": "Content B"}`
    - **Assertion**: `assert len(load_coverage_sections(text)) == 2`
- **Journal Discovery**
    - **Input**: Directory `journal-samples/` with `J1/extract.md`, `J2/extract.md`, `J3/` (empty)
    - **Expectation**: `["J1", "J2"]`
    - **Assertion**: `assert "J3" not in discover_journals()`
