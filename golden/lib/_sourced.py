from typing import get_origin, get_args
from typing import Type

from models.journal import SourcedValue, SourcedModel


def issubclass_safe(cls, base) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, base)
    except TypeError:
        return False


def _is_sourced_value(annotation: type) -> bool:
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            if _is_sourced_value(arg):
                return True
        return False
    return issubclass_safe(annotation, SourcedValue)


def _is_sourced_model(annotation: type) -> bool:
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            if _is_sourced_model(arg):
                return True
        return False
    return issubclass_safe(annotation, SourcedModel)


def _find_sourced_paths(model: type, predicate) -> set[str]:
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
            origin = get_origin(ann)
            if origin is not None:
                for arg in get_args(ann):
                    if hasattr(arg, "model_fields"):
                        for sub in _find_sourced_paths(arg, predicate):
                            paths.add(f"{name}.{sub}")
    return paths


def _sourced_value_paths(model: type) -> set[str]:
    return _find_sourced_paths(model, _is_sourced_value)


def _sourced_model_paths(model: type) -> set[str]:
    return _find_sourced_paths(model, _is_sourced_model)
