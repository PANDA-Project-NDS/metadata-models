"""
Dumps the JournalMetadata JSON schema to stdout, with all $ref references inlined.
Useful for providing a self-contained schema to LLMs or external validation tools.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from journal import JournalMetadata

def resolve_refs(obj):
    defs = obj.pop("$defs", {})

    def _inline(obj, defs):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/", 2)[2]
                    return _inline(defs.get(def_name, {}), defs)
            return {k: _inline(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_inline(item, defs) for item in obj]
        return obj

    return _inline(obj, defs)

schema = JournalMetadata.model_json_schema()
resolved_schema = resolve_refs(schema)

json.dump(resolved_schema, sys.stdout, indent=2)
sys.stdout.write("\n")
