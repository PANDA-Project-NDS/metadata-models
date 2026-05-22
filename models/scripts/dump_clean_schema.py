import json, sys, copy, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from journal import JournalMetadata

schema = JournalMetadata.model_json_schema()

def strip_evidence(obj):
    """Recursively remove evidence fields from property dicts."""
    if isinstance(obj, dict):
        cleaned = {k: strip_evidence(v) for k, v in obj.items() if k != "evidence"}
        return cleaned
    elif isinstance(obj, list):
        return [strip_evidence(item) for item in obj]
    return obj

# Step 1: Strip evidence from everything
clean = strip_evidence(copy.deepcopy(schema))
defs = clean.get("$defs", {})

# Step 2: Build map of SourcedValue def -> unwrapped inner schema
sv_map = {}
sv_keys = [k for k in defs if "SourcedValue" in k]
for k in sv_keys:
    d = defs[k]
    props = d.get("properties", {})
    val = props.get("value")
    if val:
        sv_map[k] = val  # raw inner schema

# Step 3: Replace $ref to SourcedValue defs with inline unwrapped schema
def resolve_sv_refs(obj):
    if isinstance(obj, dict):
        ref = obj.get("$ref", "")
        # Strip #/$defs/ prefix to match sv_map keys
        key = ref.removeprefix("#/$defs/") if ref else ""
        if key in sv_map:
            return copy.deepcopy(sv_map[key])
        return {k: resolve_sv_refs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_sv_refs(item) for item in obj]
    return obj

clean = resolve_sv_refs(clean)

# Step 4: Remove SourcedValue / Evidence / SourcedModel defs from $defs
clean_defs = clean.get("$defs", {})
for k in list(clean_defs):
    if "SourcedValue" in k or k in ("Evidence", "SourcedModel"):
        clean_defs.pop(k, None)

json.dump(clean, sys.stdout, indent=2)
sys.stdout.write("\n")
