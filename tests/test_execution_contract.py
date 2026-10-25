from typing import Optional

import pytest
from pydantic import ValidationError

from app.services.execution_contract import (
    ExecutionContract,
    canonicalize_stdio,
    compare_values,
    decode_function_case,
    serialize_value,
    validate_test_cases,
)
from app.services.code_execution_service import _JS_HARNESS, _PY_HARNESS


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def function_contract(adapter="json_args_v1", serializer="json_value_v1", **overrides):
    data = {
        "execution_mode": "function",
        "function_signature": "def solve(a):",
        "input_adapter": adapter,
        "output_adapter": serializer,
        "supported_languages": ["python"],
        "test_case_schema": "args_kwargs_json_v1",
        "comparison_mode": "exact_json_v1",
        "problem_version": 2,
    }
    data.update(overrides)
    return ExecutionContract.model_validate(data)


def test_contract_rejects_unknown_adapters_and_ambiguous_new_test_schema():
    with pytest.raises(ValidationError):
        function_contract(adapter="guess_from_json")

    with pytest.raises(ValidationError):
        function_contract(test_case_schema="raw_json_v2")


def test_language_harnesses_format_with_explicit_contract_fields():
    assert "__input_adapter__ = 'json_args_v1'" in _PY_HARNESS.format(
        user_code="def solve(): pass", tests_json="[]", fn_name="solve",
        input_adapter="json_args_v1", output_adapter="json_value_v1",
    )
    assert "new ${declared[0]}().${declared[1]}" in _JS_HARNESS.format(
        user_code='class Solution { solve() {} }', tests_json="[]", fn_name='"Solution.solve"',
        input_adapter='"json_args_v1"', output_adapter='"json_value_v1"',
    )


@pytest.mark.parametrize(
    "args",
    [
        ["text"],
        [1, "x"],
        [[[1, 2], [3, 4]]],
        [[], -1],
    ],
)
def test_json_args_adapter_preserves_explicit_argument_boundaries(args):
    contract = function_contract()
    decoded_args, decoded_kwargs = decode_function_case(
        {"input": {"args": args, "kwargs": {}}}, contract, {"ListNode": ListNode}
    )
    assert decoded_args == args
    assert decoded_kwargs == {}


def test_linked_list_adapter_maps_empty_sentinel_to_none():
    contract = function_contract(adapter="python_linked_list_args_v1")
    args, kwargs = decode_function_case(
        {
            "input": {
                "args": [{"$type": "linked_list", "values": [], "cycle_index": -1}],
                "kwargs": {},
            }
        },
        contract,
        {"ListNode": ListNode},
    )
    assert args == [None]
    assert kwargs == {}


def test_linked_list_adapter_builds_declared_cycle_at_zero_based_index():
    contract = function_contract(adapter="python_linked_list_args_v1")
    args, _ = decode_function_case(
        {
            "input": {
                "args": [{"$type": "linked_list", "values": [3, 2, 0, -4], "cycle_index": 1}],
                "kwargs": {},
            }
        },
        contract,
        {"ListNode": ListNode},
    )
    nodes = []
    node = args[0]
    for _ in range(4):
        nodes.append(node)
        node = node.next
    assert [node.val for node in nodes] == [3, 2, 0, -4]
    assert nodes[-1].next is nodes[1]


def test_legacy_empty_case_keeps_the_explicit_legacy_compatibility_behavior():
    contract = ExecutionContract.model_validate(
        {
            "execution_mode": "function",
            "function_signature": "def solve(head):",
            "input_adapter": "legacy_json_v1",
            "output_adapter": "legacy_python_v1",
            "supported_languages": ["python"],
            "test_case_schema": "legacy_json_v1",
            "comparison_mode": "legacy_json_v1",
            "problem_version": 1,
        }
    )

    def solve(head: Optional[ListNode]):
        return head

    args, _ = decode_function_case(
        {"input": "[[], -1]"}, contract, {"ListNode": ListNode}, function=solve
    )
    assert len(args) == 1
    assert args[0].val == []
    assert args[0].next.val == -1


def test_tree_adapter_uses_declared_level_order_encoding():
    contract = function_contract(adapter="python_tree_level_order_args_v1")
    args, _ = decode_function_case(
        {
            "input": {
            "args": [{"$type": "binary_tree_level_order", "serialization": "level_order_nulls_v1", "values": [1, 2, 3, None, 4]}],
                "kwargs": {},
            }
        },
        contract,
        {"TreeNode": TreeNode},
    )
    root = args[0]
    assert (root.val, root.left.val, root.right.val, root.left.right.val) == (1, 2, 3, 4)


def test_output_serializer_and_comparators_are_explicit_and_recursive():
    assert serialize_value(None, "json_value_v1") is None
    assert serialize_value({"items": [1, 2]}, "json_value_v1") == {"items": [1, 2]}
    assert serialize_value(ListNode(7), "node_value_v1") == 7
    assert compare_values([1, {"x": True}], [1, {"x": True}], "exact_json_v1")
    assert not compare_values([1, 2], [2, 1], "exact_json_v1")
    assert compare_values(1.0, 1.00001, "float_tolerance_v1", abs_tol=0.0001)
    assert not compare_values(1.0, 1.01, "float_tolerance_v1", abs_tol=0.0001)
    assert not compare_values(True, 1, "exact_json_v1")


def test_stdio_normalization_is_declared_and_preserves_meaningful_order():
    assert canonicalize_stdio("  hello  \r\n world \n", "normalized_text_v1") == "hello\nworld"
    assert canonicalize_stdio("[1,2]", "json_stdout_v1") == [1, 2]




def test_contract_validation_never_guesses_special_types_from_json_shape():
    contract = function_contract()
    with pytest.raises(ValueError, match="must contain only args and kwargs"):
        validate_test_cases([{"input": "[[], -1]", "expected_output": None}], contract)
    with pytest.raises(ValueError, match="not declared"):
        validate_test_cases(
            [{"input": {"args": [{"$type": "linked_list", "values": [], "cycle_index": -1}]}, "expected_output": None}],
            contract,
        )


def test_declared_linked_list_schema_accepts_empty_sentinel_and_rejects_bad_cycle_index():
    contract = function_contract(adapter="python_linked_list_args_v1")
    validate_test_cases(
        [{"input": {"args": [{"$type": "linked_list", "values": [], "cycle_index": -1}]}, "expected_output": None}],
        contract,
    )
    with pytest.raises(ValueError, match="out of range"):
        validate_test_cases(
            [{"input": {"args": [{"$type": "linked_list", "values": [1, 2], "cycle_index": 2}]}, "expected_output": 1}],
            contract,
        )
