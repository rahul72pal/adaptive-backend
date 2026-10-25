"""Versioned, explicit contracts for running coding questions."""

from __future__ import annotations

import inspect
import json
import math
import re
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


ExecutionMode = Literal["function", "stdin_stdout"]
ProblemVersion = Literal[1, 2]

SUPPORTED_LANGUAGES = {"python", "javascript"}
INPUT_ADAPTERS = {
    "legacy_json_v1",
    "json_args_v1",
    "python_linked_list_args_v1",
    "python_tree_level_order_args_v1",
    "python_graph_adjacency_args_v1",
    "stdin_text_v1",
}
OUTPUT_ADAPTERS = {"legacy_python_v1", "json_value_v1", "node_value_v1", "stdout_text_v1"}
TEST_CASE_SCHEMAS = {"legacy_json_v1", "args_kwargs_json_v1", "stdin_text_v1"}
COMPARISON_MODES = {
    "legacy_json_v1",
    "exact_json_v1",
    "float_tolerance_v1",
    "normalized_text_v1",
    "json_stdout_v1",
    "any_order_array_v1",
}


class ExecutionContract(BaseModel):
    """Question execution metadata validated before any user code is run."""

    model_config = ConfigDict(extra="forbid")

    execution_mode: ExecutionMode
    function_signature: Optional[str] = None
    input_adapter: str
    output_adapter: str
    supported_languages: List[str] = Field(min_length=1)
    test_case_schema: str
    comparison_mode: str
    problem_version: ProblemVersion
    absolute_tolerance: float = Field(default=1e-9, ge=0)
    relative_tolerance: float = Field(default=1e-9, ge=0)

    @model_validator(mode="after")
    def validate_contract(self) -> "ExecutionContract":
        if self.input_adapter not in INPUT_ADAPTERS:
            raise ValueError(f"Unsupported input adapter '{self.input_adapter}'.")
        if self.output_adapter not in OUTPUT_ADAPTERS:
            raise ValueError(f"Unsupported output adapter '{self.output_adapter}'.")
        if self.test_case_schema not in TEST_CASE_SCHEMAS:
            raise ValueError(f"Unsupported test-case schema '{self.test_case_schema}'.")
        if self.comparison_mode not in COMPARISON_MODES:
            raise ValueError(f"Unsupported comparison mode '{self.comparison_mode}'.")
        normalized_languages = {normalize_language(language) for language in self.supported_languages}
        unsupported = normalized_languages - SUPPORTED_LANGUAGES
        if unsupported:
            raise ValueError(f"Unsupported contract language(s): {', '.join(sorted(unsupported))}.")
        adapter_languages = {
            "python_linked_list_args_v1": {"python"},
            "python_tree_level_order_args_v1": {"python"},
            "python_graph_adjacency_args_v1": {"python"},
        }
        allowed = adapter_languages.get(self.input_adapter, SUPPORTED_LANGUAGES)
        if not normalized_languages.issubset(allowed):
            raise ValueError(f"Input adapter '{self.input_adapter}' is not available for every declared language.")

        if self.execution_mode == "function":
            if not (self.function_signature or "").strip():
                raise ValueError("Function mode requires function_signature.")
            if self.test_case_schema == "stdin_text_v1":
                raise ValueError("Function mode cannot use stdin_text_v1 test cases.")
            if self.input_adapter == "stdin_text_v1" or self.output_adapter == "stdout_text_v1":
                raise ValueError("Function mode cannot use stdin/stdout adapters.")
        else:
            if self.test_case_schema != "stdin_text_v1":
                raise ValueError("stdin_stdout mode requires stdin_text_v1 test cases.")
            if self.input_adapter != "stdin_text_v1" or self.output_adapter != "stdout_text_v1":
                raise ValueError("stdin_stdout mode requires stdin_text_v1 and stdout_text_v1 adapters.")
            if self.comparison_mode not in {"normalized_text_v1", "json_stdout_v1", "float_tolerance_v1"}:
                raise ValueError("stdin_stdout mode requires an explicit stdout comparison mode.")

        if self.problem_version == 2 and (
            self.input_adapter == "legacy_json_v1"
            or self.output_adapter == "legacy_python_v1"
            or self.test_case_schema == "legacy_json_v1"
            or self.comparison_mode == "legacy_json_v1"
        ):
            raise ValueError("Version 2 contracts cannot use legacy adapters or schemas.")
        if self.problem_version == 1 and self.execution_mode != "function":
            raise ValueError("Version 1 compatibility is limited to function-mode questions.")
        return self


def normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    if value in {"py", "python3"}:
        return "python"
    if value in {"js", "node"}:
        return "javascript"
    return value


def legacy_contract(language: str, function_signature: Optional[str] = "def solution():") -> ExecutionContract:
    """Named compatibility contract for pre-v2 API callers and stored questions."""
    return ExecutionContract.model_validate({
        "execution_mode": "function",
        "function_signature": function_signature or "def solution():",
        "input_adapter": "legacy_json_v1",
        "output_adapter": "legacy_python_v1",
        "supported_languages": [normalize_language(language)],
        "test_case_schema": "legacy_json_v1",
        "comparison_mode": "legacy_json_v1",
        "problem_version": 1,
    })


def _parse_legacy_input(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else raw
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


def _parameter_names(function: Any) -> List[str]:
    try:
        return list(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return []


def _build_legacy_node(value: Any, annotation: Any, namespace: Dict[str, Any]) -> Any:
    """Preserve pre-v2 node decoding only inside the explicitly named v1 adapter."""
    if not isinstance(value, (list, tuple)):
        return value
    type_name = getattr(annotation, "__name__", "")
    if type_name not in {"ListNode", "TreeNode"}:
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if origin is not None and args:
            type_name = getattr(args[0], "__name__", "")
    cls = namespace.get(type_name)
    if not isinstance(cls, type):
        return value
    if type_name == "ListNode":
        head = tail = None
        for item in value:
            node = item if isinstance(item, cls) else cls(item)
            if head is None:
                head = node
            else:
                tail.next = node
            tail = node
        return head
    if type_name == "TreeNode":
        if not value or value[0] is None:
            return None
        root = cls(value[0])
        queue = deque([root])
        idx = 1
        while queue and idx < len(value):
            parent = queue.popleft()
            left = value[idx]
            idx += 1
            if left is not None:
                parent.left = cls(left)
                queue.append(parent.left)
            if idx < len(value):
                right = value[idx]
                idx += 1
                if right is not None:
                    parent.right = cls(right)
                    queue.append(parent.right)
        return root
    return value


def _decode_typed_value(value: Any, adapter: str, namespace: Dict[str, Any]) -> Any:
    if not isinstance(value, dict) or "$type" not in value:
        return value
    kind = value.get("$type")
    if adapter == "python_linked_list_args_v1" and kind == "linked_list":
        values = value.get("values")
        cycle_index = value.get("cycle_index")
        if not isinstance(values, list):
            raise ValueError("linked_list values must be an array.")
        if cycle_index is not None and (isinstance(cycle_index, bool) or not isinstance(cycle_index, int)):
            raise ValueError("linked_list cycle_index must be an integer or null.")
        if cycle_index is not None and cycle_index != -1 and not 0 <= cycle_index < len(values):
            raise ValueError("linked_list cycle_index must be -1, null, or a valid zero-based node index.")
        if not values:
            return None
        cls = namespace.get("ListNode")
        if not isinstance(cls, type):
            raise ValueError("The submitted source must define ListNode for this input adapter.")
        nodes = [cls(item) for item in values]
        for left, right in zip(nodes, nodes[1:]):
            left.next = right
        if isinstance(cycle_index, int) and cycle_index >= 0:
            nodes[-1].next = nodes[cycle_index]
        return nodes[0]

    if adapter == "python_tree_level_order_args_v1" and kind == "binary_tree_level_order":
        values = value.get("values")
        if not isinstance(values, list):
            raise ValueError("binary_tree_level_order values must be an array.")
        if not values or values[0] is None:
            return None
        cls = namespace.get("TreeNode")
        if not isinstance(cls, type):
            raise ValueError("The submitted source must define TreeNode for this input adapter.")
        root = cls(values[0])
        queue = deque([root])
        idx = 1
        while queue and idx < len(values):
            parent = queue.popleft()
            left = values[idx]
            idx += 1
            if left is not None:
                parent.left = cls(left)
                queue.append(parent.left)
            if idx < len(values):
                right = values[idx]
                idx += 1
                if right is not None:
                    parent.right = cls(right)
                    queue.append(parent.right)
        return root

    if adapter == "python_graph_adjacency_args_v1" and kind == "graph_adjacency":
        node_count = value.get("node_count")
        edges = value.get("edges")
        directed = value.get("directed", False)
        if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 0:
            raise ValueError("graph_adjacency node_count must be a nonnegative integer.")
        if not isinstance(edges, list) or not isinstance(directed, bool):
            raise ValueError("graph_adjacency requires an edges array and boolean directed flag.")
        adjacency = [[] for _ in range(node_count)]
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("Each graph_adjacency edge must contain two node indices.")
            source, target = edge
            if any(isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < node_count for node in edge):
                raise ValueError("graph_adjacency edge index is outside the declared node range.")
            adjacency[source].append(target)
            if not directed:
                adjacency[target].append(source)
        return adjacency

    raise ValueError(f"Input value type '{kind}' is not supported by adapter '{adapter}'.")


def decode_function_case(
    test_case: Dict[str, Any],
    contract: ExecutionContract,
    namespace: Dict[str, Any],
    function: Any = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    raw = test_case.get("input")
    if contract.input_adapter == "legacy_json_v1":
        parsed = _parse_legacy_input(raw)
        names = _parameter_names(function) if function is not None else []
        if isinstance(parsed, dict):
            kwargs = parsed if not names or all(name in parsed for name in names) else parsed
            annotations = getattr(function, "__annotations__", {}) if function is not None else {}
            return [], {name: _build_legacy_node(value, annotations.get(name), namespace) for name, value in kwargs.items()}
        if isinstance(parsed, list) and function is not None:
            try:
                arity = len(inspect.signature(function).parameters)
            except (TypeError, ValueError):
                arity = 1
            if arity > 1 and len(parsed) == arity:
                values = parsed
            else:
                values = [parsed]
            annotations = getattr(function, "__annotations__", {})
            return [
                _build_legacy_node(value, annotations.get(names[idx]) if idx < len(names) else None, namespace)
                for idx, value in enumerate(values)
            ], {}
        return ([] if parsed is None else [parsed]), {}

    if contract.test_case_schema != "args_kwargs_json_v1":
        raise ValueError(f"Input adapter '{contract.input_adapter}' cannot read schema '{contract.test_case_schema}'.")
    parsed = raw
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError("Function-mode input must be valid JSON matching args_kwargs_json_v1.") from exc
    if not isinstance(parsed, dict) or set(parsed) - {"args", "kwargs"}:
        raise ValueError("Function-mode input must be an object with only 'args' and 'kwargs'.")
    args = parsed.get("args", [])
    kwargs = parsed.get("kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise ValueError("args must be an array and kwargs must be an object.")
    if contract.input_adapter == "python_linked_list_args_v1" or contract.input_adapter == "python_tree_level_order_args_v1" or contract.input_adapter == "python_graph_adjacency_args_v1":
        if normalize_language(test_case.get("language") or contract.supported_languages[0]) != "python":
            raise ValueError(f"Adapter '{contract.input_adapter}' only supports Python.")
        args = [_decode_typed_value(item, contract.input_adapter, namespace) for item in args]
        kwargs = {name: _decode_typed_value(value, contract.input_adapter, namespace) for name, value in kwargs.items()}
    return args, kwargs


def validate_test_cases(test_cases: List[Dict[str, Any]], contract: ExecutionContract) -> None:
    """Validate serialized cases against the declared schema before sandbox execution."""
    if not test_cases:
        raise ValueError("At least one test case is required.")
    for idx, case in enumerate(test_cases, start=1):
        if contract.execution_mode == "stdin_stdout":
            stdin = case.get("stdin")
            if stdin is None:
                raise ValueError(f"Test case {idx} requires a string 'stdin' value.")
            if not isinstance(stdin, str):
                raise ValueError(f"Test case {idx} stdin must be a string.")
            if "expected_output" not in case:
                raise ValueError(f"Test case {idx} requires expected_output.")
            if not isinstance(case["expected_output"], str):
                raise ValueError(f"Test case {idx} expected_output must be a string in stdin/stdout mode.")
            continue

        if contract.test_case_schema == "legacy_json_v1":
            if "input" not in case or "expected_output" not in case:
                raise ValueError(f"Legacy test case {idx} requires input and expected_output.")
            continue
        raw = case.get("input")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Test case {idx} input is not valid args_kwargs_json_v1 JSON.") from exc
        if not isinstance(raw, dict) or set(raw) - {"args", "kwargs"}:
            raise ValueError(f"Test case {idx} input must contain only args and kwargs.")
        args, kwargs = raw.get("args", []), raw.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError(f"Test case {idx} args must be an array and kwargs an object.")
        _validate_tagged_values([args, kwargs], contract.input_adapter, idx)
        if "expected_output" not in case:
            raise ValueError(f"Test case {idx} requires expected_output.")


def _validate_tagged_values(value: Any, adapter: str, case_index: int) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_tagged_values(item, adapter, case_index)
        return
    if not isinstance(value, dict):
        return
    kind = value.get("$type")
    if kind is not None:
        if adapter == "python_linked_list_args_v1" and kind == "linked_list":
            values = value.get("values")
            cycle_index = value.get("cycle_index")
            if not isinstance(values, list):
                raise ValueError(f"Test case {case_index} linked_list values must be an array.")
            if cycle_index is not None and (isinstance(cycle_index, bool) or not isinstance(cycle_index, int)):
                raise ValueError(f"Test case {case_index} linked_list cycle_index must be an integer or null.")
            if cycle_index is not None and cycle_index != -1 and not 0 <= cycle_index < len(values):
                raise ValueError(f"Test case {case_index} linked_list cycle_index is out of range.")
            return
        if adapter == "python_tree_level_order_args_v1" and kind == "binary_tree_level_order":
            if not isinstance(value.get("values"), list):
                raise ValueError(f"Test case {case_index} binary_tree_level_order values must be an array.")
            if value.get("serialization") != "level_order_nulls_v1":
                raise ValueError(f"Test case {case_index} binary tree must declare level_order_nulls_v1 serialization.")
            return
        if adapter == "python_graph_adjacency_args_v1" and kind == "graph_adjacency":
            nodes, edges = value.get("node_count"), value.get("edges")
            if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 0 or not isinstance(edges, list):
                raise ValueError(f"Test case {case_index} graph_adjacency metadata is invalid.")
            if not isinstance(value.get("directed", False), bool):
                raise ValueError(f"Test case {case_index} graph directed must be boolean.")
            for edge in edges:
                if not isinstance(edge, list) or len(edge) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < nodes for v in edge):
                    raise ValueError(f"Test case {case_index} graph contains an invalid edge.")
            return
        raise ValueError(f"Test case {case_index} type '{kind}' is not declared by adapter '{adapter}'.")
    for child in value.values():
        _validate_tagged_values(child, adapter, case_index)


def serialize_value(value: Any, adapter: str) -> Any:
    if adapter == "legacy_python_v1":
        if value is None or isinstance(value, (int, float, bool, str)):
            return value
        if isinstance(value, (list, tuple, set)):
            return [serialize_value(item, adapter) for item in value]
        if isinstance(value, dict):
            return value
        if hasattr(value, "val"):
            return value.val
        return str(value)
    if adapter == "node_value_v1":
        if value is None:
            return None
        if not hasattr(value, "val"):
            raise ValueError("node_value_v1 requires a node-like result with a val field.")
        return value.val
    if adapter != "json_value_v1":
        raise ValueError(f"Unsupported output adapter '{adapter}'.")
    return _json_value(value)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (int, float, bool, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite numbers are not valid JSON results.")
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError(f"Result type '{type(value).__name__}' is not JSON serializable by json_value_v1.")


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_strict_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_strict_equal(actual[key], expected[key]) for key in actual)
    return actual == expected


def compare_values(
    actual: Any,
    expected: Any,
    mode: str,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
) -> bool:
    if mode == "legacy_json_v1":
        actual = _legacy_parse(actual)
        expected = _legacy_parse(expected)
        if isinstance(actual, (dict, list)) or isinstance(expected, (dict, list)):
            try:
                return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
            except (TypeError, ValueError):
                return str(actual).strip() == str(expected).strip()
        if isinstance(actual, bool) or isinstance(expected, bool):
            return bool(actual) == bool(expected)
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return str(actual).strip() == str(expected).strip()
    if mode == "exact_json_v1":
        return _strict_equal(actual, expected)
    if mode == "float_tolerance_v1":
        if isinstance(actual, bool) or isinstance(expected, bool) or not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
    if mode == "normalized_text_v1":
        return _normalize_text(str(actual)) == _normalize_text(str(expected))
    if mode == "json_stdout_v1":
        try:
            left = json.loads(str(actual))
            right = json.loads(str(expected))
        except (json.JSONDecodeError, TypeError):
            return False
        return _strict_equal(left, right)
    if mode == "any_order_array_v1":
        if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
            return False
        return sorted(json.dumps(item, sort_keys=True) for item in actual) == sorted(json.dumps(item, sort_keys=True) for item in expected)
    raise ValueError(f"Unsupported comparison mode '{mode}'.")


def _legacy_parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_text(value: str) -> str:
    return "\n".join(line.strip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip())


def canonicalize_stdio(value: str, mode: str) -> Any:
    if mode == "normalized_text_v1":
        return _normalize_text(value)
    if mode == "json_stdout_v1":
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Program stdout is not valid JSON for json_stdout_v1.") from exc
    if mode == "float_tolerance_v1":
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError("Program stdout is not a single number for float_tolerance_v1.") from exc
    raise ValueError(f"Unsupported stdin/stdout comparison mode '{mode}'.")
