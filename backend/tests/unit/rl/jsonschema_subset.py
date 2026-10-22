"""A STRICT stdlib validator for the JSON Schema 2020-12 subset that
``docs/demo-bundle.schema.json`` uses (task-rl-12 acceptance test 1).

``jsonschema`` is not a dependency of this repo; rather than add one for a single test, this
implements exactly the keywords the bundle schema uses — and REFUSES any other keyword it meets
(:class:`UnsupportedKeywordError`), so a schema edit can never be silently half-checked.
Semantics follow the 2020-12 core/validation vocabulary: ``type`` (``integer`` = a number with
no fractional part, booleans are not numbers), ``properties`` / ``required`` /
``additionalProperties`` /
``propertyNames``, ``items`` / ``prefixItems`` / ``minItems`` / ``maxItems``, ``enum`` / ``const``
(JSON equality: ``1 == 1.0``, ``True != 1``), ``minimum`` / ``maximum`` / ``exclusiveMinimum``,
``minLength`` / ``pattern`` (``re.search``, applied to strings only), local ``$ref``
(``#/$defs/...``) and ``anyOf``. Annotation keywords are ignored.
"""

import math
import re
from collections.abc import Mapping
from typing import Any, Final

_ANNOTATIONS: Final = frozenset({"$schema", "$id", "title", "description", "$comment", "$defs"})
_SUPPORTED: Final = _ANNOTATIONS | frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "propertyNames",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "minLength",
        "pattern",
        "$ref",
        "anyOf",
    }
)


class UnsupportedKeywordError(ValueError):
    """The schema uses a keyword this subset does not implement (never silently skipped)."""


class SchemaViolationError(ValueError):
    """The instance does not conform."""


def _is_type(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if kind == "number":
        return True
    if kind == "integer":
        return isinstance(value, int) or (math.isfinite(value) and value == int(value))
    raise UnsupportedKeywordError(f"unknown type {kind!r}")


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_json_equal(a[k], b[k]) for k in a)
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return a is None and b is None


class Validator:
    def __init__(self, root: Mapping[str, Any]) -> None:
        self._root = root

    def validate(self, instance: Any) -> None:
        self._check(self._root, instance, "$")

    def _resolve(self, ref: str) -> Mapping[str, Any]:
        if not ref.startswith("#/"):
            raise UnsupportedKeywordError(f"only local $ref is supported, got {ref!r}")
        node: Any = self._root
        for part in ref[2:].split("/"):
            node = node[part]
        resolved: Mapping[str, Any] = node
        return resolved

    def _check(self, schema: Mapping[str, Any], value: Any, where: str) -> None:
        unknown = set(schema) - _SUPPORTED
        if unknown:
            raise UnsupportedKeywordError(f"{where}: unsupported keyword(s) {sorted(unknown)}")
        if "$ref" in schema:
            self._check(self._resolve(schema["$ref"]), value, where)
        if "anyOf" in schema:
            errors = []
            for option in schema["anyOf"]:
                try:
                    self._check(option, value, where)
                    break
                except SchemaViolationError as exc:
                    errors.append(str(exc))
            else:
                raise SchemaViolationError(f"{where}: matches no anyOf option ({errors})")
        if "type" in schema:
            kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not any(_is_type(value, kind) for kind in kinds):
                raise SchemaViolationError(f"{where}: {type(value).__name__} is not {kinds}")
        if "const" in schema and not _json_equal(value, schema["const"]):
            raise SchemaViolationError(f"{where}: {value!r} != const {schema['const']!r}")
        if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
            raise SchemaViolationError(f"{where}: {value!r} not in enum {schema['enum']!r}")
        if isinstance(value, int | float) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise SchemaViolationError(f"{where}: {value} < minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise SchemaViolationError(f"{where}: {value} > maximum {schema['maximum']}")
            if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
                raise SchemaViolationError(f"{where}: {value} <= {schema['exclusiveMinimum']}")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                raise SchemaViolationError(f"{where}: shorter than {schema['minLength']}")
            if "pattern" in schema and re.search(schema["pattern"], value) is None:
                raise SchemaViolationError(f"{where}: {value[:40]!r} !~ {schema['pattern']!r}")
        if isinstance(value, list):
            self._check_array(schema, value, where)
        if isinstance(value, Mapping):
            self._check_object(schema, value, where)

    def _check_array(self, schema: Mapping[str, Any], value: list[Any], where: str) -> None:
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaViolationError(f"{where}: {len(value)} items < {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaViolationError(f"{where}: {len(value)} items > {schema['maxItems']}")
        prefix = schema.get("prefixItems", [])
        for i, (item_schema, item) in enumerate(zip(prefix, value, strict=False)):
            self._check(item_schema, item, f"{where}[{i}]")
        if "items" in schema:
            for i in range(len(prefix), len(value)):
                self._check(schema["items"], value[i], f"{where}[{i}]")

    def _check_object(
        self, schema: Mapping[str, Any], value: Mapping[str, Any], where: str
    ) -> None:
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise SchemaViolationError(f"{where}: missing required {missing}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if "propertyNames" in schema:
                self._check(schema["propertyNames"], key, f"{where}<key {key!r}>")
            if key in properties:
                self._check(properties[key], item, f"{where}.{key}")
            elif "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    raise SchemaViolationError(f"{where}: unexpected property {key!r}")
                if extra is not True:
                    self._check(extra, item, f"{where}.{key}")
