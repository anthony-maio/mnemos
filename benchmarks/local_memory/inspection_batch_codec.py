"""Experimental lossless JSON string sharing; not an enabled MCP format.

References only share identical strings. They never summarize, select facts,
or establish that a caller inspected a result. Reserved object shapes are
escaped so user-authored metadata cannot masquerade as transport references.
"""

from collections import Counter
import json


def serialized(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def encode(results):
    counts = Counter()

    def collect(value):
        if isinstance(value, str):
            counts[value] += 1
        elif isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError("JSON object keys must be strings")
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif value is not None and type(value) not in (int, float, bool):
            raise ValueError("Expected JSON values")

    collect(results)
    serialized(results)  # Reject NaN and infinities before any encoding.
    strings, references = [], {}
    for value, count in counts.items():
        index = len(strings)
        ref_bytes = len(serialized({"$ref": index}))
        literal_bytes = len(serialized(value))
        if count > 1 and count * literal_bytes > literal_bytes + 1 + count * ref_bytes:
            references[value] = index
            strings.append(value)

    def visit(value):
        if isinstance(value, str) and value in references:
            return {"$ref": references[value]}
        if isinstance(value, dict):
            pairs = [(key, visit(item)) for key, item in value.items()]
            if "$ref" in value or "$object" in value:
                return {"$object": [[key, item] for key, item in pairs]}
            return dict(pairs)
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return {"format": "inspection_strings_v1", "strings": strings, "results": visit(results)}


def decode(bundle):
    if (
        set(bundle) != {"format", "strings", "results"}
        or bundle["format"] != "inspection_strings_v1"
    ):
        raise ValueError("Unknown inspection format")
    strings = bundle["strings"]
    if not isinstance(strings, list) or any(not isinstance(s, str) for s in strings):
        raise ValueError("Invalid string dictionary")

    def visit(value):
        if isinstance(value, dict):
            if "$ref" in value:
                index = value["$ref"]
                if (
                    set(value) != {"$ref"}
                    or type(index) is not int
                    or not 0 <= index < len(strings)
                ):
                    raise ValueError("Invalid string reference")
                return strings[index]
            if "$object" in value:
                pairs = value["$object"]
                if set(value) != {"$object"} or not isinstance(pairs, list):
                    raise ValueError("Invalid escaped object")
                result = {}
                for pair in pairs:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or not isinstance(pair[0], str)
                        or pair[0] in result
                    ):
                        raise ValueError("Invalid escaped object entry")
                    result[pair[0]] = visit(pair[1])
                return result
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    result = visit(bundle["results"])
    serialized(result)
    return result
