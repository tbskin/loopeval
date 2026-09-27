from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .base import ProviderError

_CONSTRAINTS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
}


def _encoded(node: dict[str, Any]) -> bool:
    """Whether this value needs JSON-string transport in a closed provider schema."""
    if "$ref" in node:
        return False
    if "anyOf" in node:
        return any(_encoded(branch) for branch in node["anyOf"])
    if node.get("type") == "object":
        return node.get("additionalProperties") is not False and not node.get("properties")
    return not any(key in node for key in ("type", "enum", "const"))


def provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Build the common strict-output subset without changing public model schemas.

    OpenAI and Anthropic require closed objects. Arbitrary JSON values are sent
    as JSON strings, then restored by ``decode_response`` before local validation.
    Bounds unsupported by Anthropic remain instructions and are enforced locally.
    """

    def transform(node: dict[str, Any]) -> dict[str, Any]:
        if _encoded(node):
            return {
                "type": "string",
                "description": (
                    "Return a JSON-encoded value as a string, with no markdown. "
                    "The decoded value must match this schema: " + json.dumps(node)
                ),
            }
        result = deepcopy(node)
        result.pop("default", None)
        constraints = {key: result.pop(key) for key in sorted(_CONSTRAINTS) if key in result}
        if constraints:
            result["description"] = (
                result.get("description", "") + " Constraints: " + json.dumps(constraints)
            ).strip()
        for key in ("$defs", "definitions", "properties"):
            if key in result:
                result[key] = {name: transform(value) for name, value in result[key].items()}
        if result.get("type") == "object":
            result["properties"] = result.get("properties", {})
            result["additionalProperties"] = False
            result["required"] = list(result["properties"])
        if isinstance(result.get("items"), dict):
            result["items"] = transform(result["items"])
        for key in ("anyOf", "allOf", "oneOf"):
            if key in result:
                result[key] = [transform(branch) for branch in result[key]]
        return result

    return transform(schema)


def decode_response(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError("non-finite JSON number")

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        reference = node.get("$ref")
        if reference is None:
            return node
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ProviderError("structured response schema uses an unsupported reference")
        target: Any = schema
        for component in reference[2:].split("/"):
            target = target[component.replace("~1", "/").replace("~0", "~")]
        if not isinstance(target, dict):
            raise ProviderError("structured response schema contains an invalid reference")
        return target

    def matches(value: Any, node: dict[str, Any]) -> bool:
        node = resolve(node)
        if "anyOf" in node:
            return any(matches(value, branch) for branch in node["anyOf"])
        types = {
            "null": value is None,
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, int | float) and not isinstance(value, bool),
        }
        kind = node.get("type")
        valid_type = (
            any(types.get(item, False) for item in kind)
            if isinstance(kind, list)
            else types.get(kind, True)
            if isinstance(kind, str)
            else True
        )
        return bool(valid_type and ("enum" not in node or value in node["enum"]))

    def decode(value: Any, node: dict[str, Any]) -> Any:
        node = resolve(node)
        if _encoded(node):
            if not isinstance(value, str):
                raise ProviderError("structured response contained a non-string JSON value")
            try:
                decoded = json.loads(value, parse_constant=reject_constant)
            except (ValueError, TypeError):
                raise ProviderError(
                    "structured response contained an invalid JSON-encoded value"
                ) from None
            if not matches(decoded, node):
                raise ProviderError("structured response JSON value has an incorrect type")
            return decoded
        if "anyOf" in node:
            for branch in node["anyOf"]:
                if matches(value, branch):
                    return decode(value, branch)
        if isinstance(value, dict) and "properties" in node:
            return {
                key: decode(item, node["properties"][key]) if key in node["properties"] else item
                for key, item in value.items()
            }
        if isinstance(value, list) and isinstance(node.get("items"), dict):
            return [decode(item, node["items"]) for item in value]
        return value

    result = decode(payload, schema)
    if not isinstance(result, dict):
        raise ProviderError("structured response must decode to a JSON object")
    return result
