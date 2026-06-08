import os

# Force evidence mode for flatten_metadata tests
os.environ["WITH_EVIDENCE"] = "1"

from golden.lib.flatten import (
    get_path,
    set_path,
    merge_partial,
    strip_evidence,
    flatten_metadata,
    FlattenedField,
)
from golden.lib._sourced import (
    _sourced_value_paths,
    _sourced_model_paths,
)
from models.journal import (
    SourcedValue,
    Evidence,
    BasicInfoExtraction,
    JournalMetadata,
)


# --- 1. Path Navigation (get_path / set_path) ---


class TestGetPath:
    def test_basic_access(self):
        assert get_path({"name": "Journal A"}, "name") == "Journal A"

    def test_deep_nesting(self):
        assert get_path({"a": {"b": {"c": 1}}}, "a.b.c") == 1

    def test_list_indexing(self):
        data = {"pricing": [{"fee": 100}, {"fee": 200}]}
        assert get_path(data, "pricing[1].fee") == 200

    def test_missing_keys_returns_none(self):
        assert get_path({}, "a.b") is None

    def test_missing_nested_key_returns_none(self):
        assert get_path({"a": 1}, "a.b.c") is None

    def test_list_out_of_bounds_returns_none(self):
        data = {"items": [1]}
        assert get_path(data, "items[5]") is None


class TestSetPath:
    def test_set_missing_keys_creates_nested_dict(self):
        data = {}
        set_path(data, "a.b.c", 10)
        assert data == {"a": {"b": {"c": 10}}}

    def test_set_existing_key_overwrites(self):
        data = {"a": 1}
        set_path(data, "a", 2)
        assert data["a"] == 2

    def test_list_expansion_pads_with_none(self):
        data = {"list": [1]}
        set_path(data, "list[2]", 3)
        assert data["list"][2] == 3
        assert data["list"][1] is None


# --- 2. Schema Flattening (flatten_metadata) ---


class TestFlattenMetadata:
    def test_sourced_value_propagation(self):
        """SourcedValue yields (path, value, evidence) triple."""
        model = BasicInfoExtraction(
            title=SourcedValue(
                value="Open Access Journal",
                evidence=Evidence(
                    quote="This is an Open Access Journal", source="about.html"
                ),
            )
        )
        fields = list(flatten_metadata(model))
        title_fields = [f for f in fields if f.path == "title"]
        assert len(title_fields) == 1
        assert title_fields[0].value == "Open Access Journal"
        assert "Open Access Journal" in title_fields[0].evidence
        assert "about.html" in title_fields[0].evidence

    def test_sourced_model_propagation(self):
        """SourcedModel evidence propagates to child fields; local evidence takes precedence."""
        from models.journal import Metrics, SourcedValue

        model = BasicInfoExtraction(
            metrics=Metrics(
                cite_score=SourcedValue(
                    value=3.5,
                    evidence=Evidence(
                        quote="Local Quote", source="metrics.html"
                    ),
                ),
                impact_factor=SourcedValue(
                    value=2.1,
                    evidence=Evidence(
                        quote="Local Quote IF", source="metrics.html"
                    ),
                ),
            )
        )
        fields = list(flatten_metadata(model))
        cite = [f for f in fields if f.path == "metrics.cite_score"]
        assert len(cite) == 1
        assert cite[0].value == 3.5
        assert "Local Quote" in cite[0].evidence

    def test_null_value_with_evidence(self):
        """Null SourcedValue yields (path, None, evidence)."""
        from pydantic import BaseModel, Field
        from typing import Optional

        class NullableModel(BaseModel):
            field: Optional[SourcedValue[Optional[str]]] = None

        model = NullableModel(
            field=SourcedValue(
                value=None,
                evidence=Evidence(
                    quote="No info found", source="about.html"
                ),
            )
        )
        fields = list(flatten_metadata(model))
        field_results = [f for f in fields if f.path == "field"]
        assert len(field_results) == 1
        assert field_results[0].value is None
        assert "No info found" in field_results[0].evidence


# --- 3. Deep Merging (merge_partial) ---


class TestMergePartial:
    def test_scalar_first_win(self):
        """Existing scalar value is preserved (first-wins)."""
        base = {"a": 1}
        patch = {"a": 2}
        result = merge_partial(base, patch)
        assert result["a"] == 1

    def test_scalar_fill(self):
        """None in base is filled from patch."""
        base = {"a": None}
        patch = {"a": 2}
        result = merge_partial(base, patch)
        assert result["a"] == 2

    def test_scalar_overwrite_force(self):
        """force=True overwrites existing scalar."""
        base = {"a": 1}
        patch = {"a": 2}
        result = merge_partial(base, patch, force=True)
        assert result["a"] == 2

    def test_recursive_object_merge(self):
        """Nested dicts are merged recursively."""
        base = {"meta": {"a": 1}}
        patch = {"meta": {"b": 2}}
        result = merge_partial(base, patch)
        assert result["meta"] == {"a": 1, "b": 2}

    def test_list_deduplication(self):
        """Lists are deduplicated by key; only new items are added."""
        base = {"pricing": [{"type": "APC", "val": 100}]}
        patch = {
            "pricing": [
                {"type": "APC", "val": 200},
                {"type": "Page", "val": 10},
            ]
        }
        result = merge_partial(base, patch, dedup_key="type")
        assert len(result["pricing"]) == 2
        assert result["pricing"][0]["val"] == 100
        assert result["pricing"][1]["type"] == "Page"

    def test_none_values_in_patch_skipped(self):
        """None values in patch are skipped entirely."""
        base = {"a": 1, "b": 2}
        patch = {"a": None, "c": 3}
        result = merge_partial(base, patch)
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3


# --- 4. Sourced Type Detection (_sourced.py) ---


class TestSourcedTypeDetection:
    def test_sourced_value_paths_discovered(self):
        """_sourced_value_paths finds SourcedValue fields in a model."""
        paths = _sourced_value_paths(BasicInfoExtraction)
        assert "title" in paths
        assert "publisher" in paths
        assert "scope" in paths

    def test_optional_sourced_value_unwrapped(self):
        """Optional[SourcedValue[T]] is correctly identified."""
        from typing import Optional
        from pydantic import BaseModel, Field

        class TestModel(BaseModel):
            field: Optional[SourcedValue[str]] = None

        paths = _sourced_value_paths(TestModel)
        assert "field" in paths

    def test_sourced_model_paths_discovered(self):
        """_sourced_model_paths finds SourcedModel subclasses."""
        paths = _sourced_model_paths(JournalMetadata)
        assert "publication_frequency" in paths


# --- 5. Utility Helpers ---


class TestStripEvidence:
    def test_removes_evidence_keys_recursively(self):
        data = {
            "val": 1,
            "evidence": "quote",
            "nested": {"val": 2, "evidence": "nested quote"},
            "list": [{"val": 3, "evidence": "list quote"}],
        }
        result = strip_evidence(data)
        assert "evidence" not in result
        assert "evidence" not in result["nested"]
        assert "evidence" not in result["list"][0]
        assert result["val"] == 1
        assert result["nested"]["val"] == 2
        assert result["list"][0]["val"] == 3

    def test_preserves_non_evidence_keys(self):
        data = {"name": "Test", "score": 42}
        result = strip_evidence(data)
        assert result == {"name": "Test", "score": 42}


class TestFlattenedField:
    def test_frozen_dataclass(self):
        """FlattenedField is immutable."""
        f = FlattenedField(path="title", value="Test", evidence="quote")
        assert f.path == "title"
        assert f.value == "Test"
        assert f.evidence == "quote"
