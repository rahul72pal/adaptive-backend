"""Server-side code execution for practice questions.

Strategy
--------
A submission is wrapped in a per-language harness that runs every test case and
prints a JSON result payload. The harness is executed on the OneCompiler hosted
runner (``ONE_COMPILER_API``); no untrusted code runs on this host.

The caller receives a normalised result::

    {
      "success": bool,
      "supported": bool,
      "all_passed": bool,
      "passed": int,
      "total": int,
      "results": [{"case", "passed", "input", "expected", "actual", "error"}],
      "provider": "onecompiler",
      "stderr": str,
    }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.services.execution_contract import (
    ExecutionContract,
    canonicalize_stdio,
    compare_values,
    legacy_contract,
    normalize_language,
    validate_test_cases,
)

logger = logging.getLogger(__name__)

# canonical language -> (OneCompiler language id, source file name)
_ONECOMPILER_LANGS = {
    "python": ("python", "index.py"),
    "python3": ("python", "index.py"),
    "py": ("python", "index.py"),
    "javascript": ("javascript", "index.js"),
    "js": ("javascript", "index.js"),
    "node": ("javascript", "index.js"),
    "typescript": ("typescript", "index.ts"),
    "c++": ("cpp", "index.cpp"),
    "cpp": ("cpp", "index.cpp"),
    "c": ("c", "index.c"),
}

_PY_HARNESS = r'''
import json, sys, inspect, io
import typing
from typing import *
import collections
from collections import *
import math, heapq, bisect, random

__user_code__ = {user_code!r}
__tests__ = json.loads({tests_json!r})
__fn_name__ = {fn_name!r}
__input_adapter__ = {input_adapter!r}
__output_adapter__ = {output_adapter!r}

ns = {{
    "__name__": "__main__",
    "math": math,
    "heapq": heapq,
    "bisect": bisect,
    "random": random,
    "collections": collections,
    "typing": typing,
}}
for mod in (typing, collections):
    for k in dir(mod):
        if not k.startswith("_"):
            ns[k] = getattr(mod, k)

try:
    exec(__user_code__, ns)
except SyntaxError:
    print(json.dumps({{"fatal": {{"category": "compile_error", "message": "The submitted source has a syntax error."}}}}))
    sys.exit(0)
except Exception:
    print(json.dumps({{"fatal": {{"category": "runtime_error", "message": "The program failed during initialization."}}}}))
    sys.exit(0)

if "." in __fn_name__:
    owner_name, method_name = __fn_name__.split(".", 1)
    owner = ns.get(owner_name)
    if isinstance(owner, type) and method_name.isidentifier():
        try:
            fn = getattr(owner(), method_name)
        except Exception:
            fn = None
    else:
        fn = None
else:
    fn = ns.get(__fn_name__)
if not callable(fn) and __input_adapter__ == "legacy_json_v1":
    sol_cls = ns.get("Solution")
    if isinstance(sol_cls, type):
        try:
            sol_inst = sol_cls()
            if hasattr(sol_inst, __fn_name__):
                fn = getattr(sol_inst, __fn_name__)
            else:
                methods = [getattr(sol_inst, m) for m in dir(sol_inst) if callable(getattr(sol_inst, m)) and not m.startswith("_")]
                fn = methods[-1] if methods else None
        except Exception:
            pass

if not callable(fn) and __input_adapter__ == "legacy_json_v1":
    candidates = [
        v for k, v in ns.items()
        if callable(v) and not k.startswith("_")
        and getattr(v, "__module__", None) in (None, "__main__")
    ]
    fn = candidates[-1] if candidates else None

if fn is None:
    print(json.dumps({{"fatal": {{"category": "invalid_contract", "message": "The declared function was not found."}}}}))
    sys.exit(0)

try:
    __hints__ = typing.get_type_hints(fn)
except Exception:
    __hints__ = {{}}

def _legacy_build(value, name):
    h = __hints__.get(name)
    if typing.get_origin(h) is typing.Union:
        args = [a for a in typing.get_args(h) if a is not type(None)]
        h = args[0] if args else None
    type_name = getattr(h, "__name__", "")
    cls = ns.get(type_name) if type_name in ("ListNode", "TreeNode") else None
    if not isinstance(value, (list, tuple)) or not isinstance(cls, type):
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
    if not value or value[0] is None:
        return None
    root = cls(value[0])
    queue = collections.deque([root])
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

def _typed_value(value):
    if isinstance(value, list):
        return [_typed_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("$type")
    if __input_adapter__ == "python_linked_list_args_v1" and kind == "linked_list":
        values = value.get("values")
        cycle_index = value.get("cycle_index")
        cls = ns.get("ListNode")
        if not isinstance(values, list) or not isinstance(cls, type):
            raise ValueError("Invalid linked-list contract or ListNode definition.")
        if not values:
            return None
        if cycle_index is not None and (isinstance(cycle_index, bool) or not isinstance(cycle_index, int)):
            raise ValueError("cycle_index must be an integer or null.")
        if cycle_index is not None and cycle_index != -1 and not 0 <= cycle_index < len(values):
            raise ValueError("cycle_index is outside the linked-list node range.")
        nodes = [cls(item) for item in values]
        for left, right in zip(nodes, nodes[1:]):
            left.next = right
        if isinstance(cycle_index, int) and cycle_index >= 0:
            nodes[-1].next = nodes[cycle_index]
        return nodes[0]
    if __input_adapter__ == "python_tree_level_order_args_v1" and kind == "binary_tree_level_order":
        values = value.get("values")
        cls = ns.get("TreeNode")
        if not isinstance(values, list) or not isinstance(cls, type):
            raise ValueError("Invalid level-order tree contract or TreeNode definition.")
        if not values or values[0] is None:
            return None
        root = cls(values[0])
        queue = collections.deque([root])
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
    if __input_adapter__ == "python_graph_adjacency_args_v1" and kind == "graph_adjacency":
        count = value.get("node_count")
        edges = value.get("edges")
        directed = value.get("directed", False)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or not isinstance(edges, list) or not isinstance(directed, bool):
            raise ValueError("Invalid graph adjacency contract.")
        graph = [[] for _ in range(count)]
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < count for v in edge):
                raise ValueError("Invalid graph edge in adjacency contract.")
            source, target = edge
            graph[source].append(target)
            if not directed:
                graph[target].append(source)
        return graph
    return {{key: _typed_value(item) for key, item in value.items()}}

def _decode_input(raw):
    if __input_adapter__ == "legacy_json_v1":
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
        except Exception:
            parsed = raw
        try:
            signature = inspect.signature(fn)
            names = list(signature.parameters)
            arity = len(names)
        except Exception:
            names, arity = [], 1
        if isinstance(parsed, dict):
            if names:
                return [], {{name: _legacy_build(parsed.get(name), name) for name in names}}
            return [], parsed
        if isinstance(parsed, list) and arity > 1 and len(parsed) == arity:
            values = parsed
        else:
            values = [parsed]
        return [_legacy_build(value, names[idx] if idx < len(names) else "") for idx, value in enumerate(values)], {{}}

    parsed = raw
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception as exc:
            raise ValueError("Function input must be valid args_kwargs_json_v1 JSON.") from exc
    if not isinstance(parsed, dict) or set(parsed) - {{"args", "kwargs"}}:
        raise ValueError("Function input must contain only args and kwargs.")
    args, kwargs = parsed.get("args", []), parsed.get("kwargs", {{}})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise ValueError("Function input args must be an array and kwargs an object.")
    if __input_adapter__ != "json_args_v1":
        args = [_typed_value(value) for value in args]
        kwargs = {{name: _typed_value(value) for name, value in kwargs.items()}}
    return args, kwargs

def _legacy_serialize(value):
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_legacy_serialize(item) for item in value]
    if isinstance(value, dict):
        return value
    if hasattr(value, "val"):
        return value.val
    return str(value)

def _serialize(value):
    if __output_adapter__ == "legacy_python_v1":
        return _legacy_serialize(value)
    if __output_adapter__ == "node_value_v1":
        if value is None:
            return None
        if not hasattr(value, "val"):
            raise ValueError("node_value_v1 requires a node result.")
        return value.val
    if value is None or type(value) in (int, float, bool, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {{key: _serialize(item) for key, item in value.items()}}
    raise ValueError("Result is not JSON serializable by json_value_v1.")

results = []
for idx, tc in enumerate(__tests__):
    raw = tc.get("input", "")
    expected = tc.get("expected_output", "")
    try:
        call_args, call_kwargs = _decode_input(tc.get("input", ""))
        actual = fn(*call_args, **call_kwargs)
        results.append({{
            "case": idx + 1,
            "input": raw,
            "expected": expected,
            "actual": _serialize(actual),
            "error": None,
            "status": "executed",
        }})
    except Exception as exc:
        results.append({{
            "case": idx + 1,
            "input": raw,
            "expected": expected,
            "actual": None,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:400]),
            "status": "runtime_error",
        }})

print(json.dumps({{"results": results}}))
'''

_JS_HARNESS = r'''
const __userCode__ = {user_code};
const __tests__ = {tests_json};
const __fnName__ = {fn_name};
const __inputAdapter__ = {input_adapter};
const __outputAdapter__ = {output_adapter};

let fn;
try {{
  const declared = __fnName__.split(".");
  const expression = declared.length === 2 ? `new ${{declared[0]}}().${{declared[1]}}` : __fnName__;
  fn = eval(__userCode__ + "\n;" + expression);
}} catch (e) {{
  console.log(JSON.stringify({{ fatal: String(e) }}));
  process.exit(0);
}}
if (typeof fn !== "function" && __inputAdapter__ === "legacy_json_v1") {{
  const names = Object.keys(globalThis).filter((k) => typeof globalThis[k] === "function");
  fn = names.length ? globalThis[names[names.length - 1]] : null;
}}
if (typeof fn !== "function") {{
  console.log(JSON.stringify({{ fatal: {{ category: "invalid_contract", message: "The declared function was not found." }} }}));
  process.exit(0);
}}

const results = [];
for (let i = 0; i < __tests__.length; i++) {{
  const tc = __tests__[i];
  let parsed;
  try {{
    parsed = typeof tc.input === "string" && tc.input.trim() ? JSON.parse(tc.input) : tc.input;
  }} catch (e) {{
    parsed = tc.input;
  }}
  try {{
    let args;
    let kwargs;
    if (__inputAdapter__ === "legacy_json_v1") {{
      if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {{
        args = [];
        kwargs = parsed;
      }} else if (Array.isArray(parsed) && fn.length > 1 && parsed.length === fn.length) {{
        args = parsed;
        kwargs = {{}};
      }} else {{
        args = [parsed];
        kwargs = {{}};
      }}
    }} else {{
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object" || Object.keys(parsed).some((key) => !["args", "kwargs"].includes(key))) {{
        throw new Error("Function input must contain only args and kwargs.");
      }}
      args = parsed.args ?? [];
      kwargs = parsed.kwargs ?? {{}};
      if (!Array.isArray(args) || !kwargs || Array.isArray(kwargs) || typeof kwargs !== "object") {{
        throw new Error("Function input args must be an array and kwargs an object.");
      }}
      if (Object.keys(kwargs).length) throw new Error("JavaScript function mode supports positional args only.");
    }}
    const actualValue = __inputAdapter__ === "legacy_json_v1" ? fn(...args, ...Object.values(kwargs)) : fn(...args);
    let actual = actualValue;
    if (__outputAdapter__ === "node_value_v1") actual = actualValue == null ? null : actualValue.val;
    if (__outputAdapter__ === "json_value_v1" && actualValue === undefined) throw new Error("Function returned undefined.");
    results.push({{ case: i + 1, input: tc.input, expected: tc.expected_output, actual, error: null }});
  }} catch (e) {{
    results.push({{ case: i + 1, input: tc.input, expected: tc.expected_output, actual: null, error: String(e).slice(0, 400), status: "runtime_error" }});
  }}
}}
console.log(JSON.stringify({{ results }}));
'''


def _normalise_language(language: Optional[str]) -> Optional[str]:
    lang = normalize_language(language or "")
    return lang if lang in {"python", "javascript"} else None


def _normalise_results(raw_results: List[Dict[str, Any]], contract: ExecutionContract) -> Dict[str, Any]:
    results = []
    for idx, item in enumerate(raw_results or []):
        actual = item.get("actual")
        expected = item.get("expected", item.get("expected_output", ""))
        if contract.problem_version == 2 and isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                pass
        error = item.get("error")
        passed = item.get("passed")
        if passed is None:
            passed = bool(error is None and compare_values(
                actual,
                expected,
                contract.comparison_mode,
                abs_tol=contract.absolute_tolerance,
                rel_tol=contract.relative_tolerance,
            ))
        results.append({
            "case": item.get("case", idx + 1),
            "passed": bool(passed),
            "input": item.get("input", ""),
            "expected": expected,
            "actual": actual,
            "error": error,
            "execution_status": item.get("status", "executed" if error is None else "runtime_error"),
        })
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "success": True,
        "supported": True,
        "all_passed": total > 0 and passed_count == total,
        "passed": passed_count,
        "total": total,
        "results": results,
        "execution_status": "runtime_error" if any(r["execution_status"] == "runtime_error" for r in results) else "completed",
    }


async def _run_onecompiler(language: str, code: str) -> Dict[str, Any]:
    """Compatibility wrapper retained for callers; provider details live below."""
    return await _onecompiler_request(language, code)


class OneCompilerFailure(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.public_message = message


def _exception_category(value: Any) -> str:
    message = str(value or "").lower()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if any(term in message for term in ("syntaxerror", "syntax error", "compilation error", "compile error", "compiler error")):
        return "compile_error"
    return "runtime_error"


def _onecompiler_api_version() -> str:
    configured = (settings.ONE_COMPILER_API_VERSION or "auto").lower()
    if configured in {"v1", "legacy_exec"}:
        return configured
    if configured == "auto":
        return "legacy_exec" if "/api/code/exec" in settings.ONE_COMPILER_API_URL else "v1"
    return configured


async def _onecompiler_request(language: str, code: str, stdin: Optional[List[str]] = None) -> Dict[str, Any]:
    """Translate platform execution requests to the configured OneCompiler API."""
    lang_id, filename = _ONECOMPILER_LANGS[language]
    version = _onecompiler_api_version()
    if version not in {"v1", "legacy_exec"}:
        raise OneCompilerFailure("configuration_error", "The configured OneCompiler API version is unsupported.")
    if version == "legacy_exec":
        if stdin is not None:
            raise OneCompilerFailure("invalid_contract", "The configured OneCompiler endpoint does not support stdin/stdout batches.")
        payload = {"properties": {"language": lang_id, "files": [{"name": filename, "content": code}]}}
    else:
        payload = {"language": lang_id, "files": [{"name": filename, "content": code}]}
        if stdin is not None:
            payload["stdin"] = stdin
    headers = {"Content-Type": "application/json"}
    if settings.ONE_COMPILER_API:
        headers["X-API-Key"] = settings.ONE_COMPILER_API

    try:
        async with httpx.AsyncClient(timeout=settings.ONE_COMPILER_TIMEOUT_SECONDS) as client:
            resp = await client.post(settings.ONE_COMPILER_API_URL, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise OneCompilerFailure("timeout", "Execution timed out while contacting the code runner.") from exc
    except httpx.HTTPError as exc:
        raise OneCompilerFailure("infrastructure_error", "The code runner could not be reached.") from exc
    if resp.status_code == 429:
        raise OneCompilerFailure("rate_limited", "The code runner is rate limited. Try again shortly.")
    if resp.status_code >= 500:
        raise OneCompilerFailure("infrastructure_error", "The code runner is temporarily unavailable.")
    if not resp.is_success:
        raise OneCompilerFailure("provider_error", f"The code runner rejected the request (HTTP {resp.status_code}).")
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise OneCompilerFailure("malformed_response", "The code runner returned an invalid response.") from exc
    if isinstance(data, list):
        if stdin is None or len(data) != len(stdin):
            raise OneCompilerFailure("malformed_response", "The code runner returned an unexpected batch response.")
        entries = data
    elif isinstance(data, dict):
        entries = [data]
    else:
        raise OneCompilerFailure("malformed_response", "The code runner returned an invalid response shape.")
    if stdin is not None and len(entries) != len(stdin):
        raise OneCompilerFailure("malformed_response", "The code runner returned a different number of results than submitted inputs.")
    normalized = []
    output_limit = settings.ONE_COMPILER_MAX_OUTPUT_BYTES
    for entry in entries:
        if not isinstance(entry, dict) or not all(key in entry for key in ("stdout", "stderr")):
            raise OneCompilerFailure("malformed_response", "The code runner response is missing execution output fields.")
        stdout, stderr = entry.get("stdout"), entry.get("stderr")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise OneCompilerFailure("malformed_response", "The code runner returned invalid output fields.")
        if len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")) > output_limit:
            raise OneCompilerFailure("output_limit", "Program output exceeded the configured size limit.")
        provider_status = entry.get("status")
        if provider_status is not None and str(provider_status).lower() not in {"success", "successful", "completed"}:
            raise OneCompilerFailure("provider_error", "The code runner did not complete execution successfully.")
        normalized.append({"stdout": stdout, "stderr": stderr, "exception": entry.get("exception"),
                           "exit_status": entry.get("exitCode", entry.get("exit_code")),
                           "execution_time": entry.get("executionTime"), "memory_used": entry.get("memoryUsed")})
    return normalized[0] if stdin is None else {"batch": normalized}


def _extract_json_payload(stdout: str) -> Optional[Dict[str, Any]]:
    text = (stdout or "").strip()
    if not text:
        return None
    # Tolerate log lines preceding the JSON payload.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    start = text.find("{")
    if start != -1:
        try:
            return json.loads(text[start:])
        except Exception:
            return None
    return None


async def execute_code(
    language: Optional[str],
    code: str,
    test_cases: List[Dict[str, Any]],
    function_name: str = "solution",
    contract_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute ``code`` against ``test_cases`` in an isolated sandbox."""
    base = {"success": False, "supported": True, "results": [], "total": len(test_cases or []), "passed": 0, "all_passed": False}
    if not settings.CODE_EXECUTION_ENABLED:
        return {**base, "supported": False, "error": "Code execution is disabled.", "execution_status": "disabled"}

    lang = _normalise_language(language)
    if lang is None:
        return {**base, "supported": False, "error": f"Unsupported language '{language}'.", "execution_status": "unsupported_language"}

    if not (code or "").strip():
        return {**base, "error": "No code submitted.", "execution_status": "invalid_submission"}

    tests = test_cases or []
    fn_name = (function_name or "solution").strip() or "solution"
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", fn_name):
        return {**base, "error": "function_name must be a declared identifier or Class.method path.", "execution_status": "invalid_contract", "runner_error": True}
    api_version = _onecompiler_api_version()
    if api_version not in {"v1", "legacy_exec"}:
        return {**base, "error": "The configured OneCompiler API version is unsupported.", "execution_status": "configuration_error", "runner_error": True}

    try:
        if contract_data:
            explicit_contract_data = dict(contract_data)
            # Older v1 records sometimes stored the callable separately from
            # its signature. Preserve that existing declared name in the
            # compatibility path; v2 contracts must provide the full field.
            if explicit_contract_data.get("problem_version") == 1 and not explicit_contract_data.get("function_signature"):
                explicit_contract_data["function_signature"] = f"def {fn_name}():"
            contract = ExecutionContract.model_validate(explicit_contract_data)
        else:
            contract = legacy_contract(lang, f"def {fn_name}():")
        if lang not in [normalize_language(item) for item in contract.supported_languages]:
            raise ValueError(f"Language '{lang}' is not supported by this problem contract.")
        validate_test_cases(tests, contract)
        if contract.execution_mode == "stdin_stdout" and api_version == "legacy_exec":
            raise ValueError("The configured OneCompiler API version cannot run stdin/stdout problems.")
    except Exception as exc:
        return {**base, "error": str(exc), "execution_status": "invalid_contract", "runner_error": True}

    if contract.execution_mode == "stdin_stdout":
        try:
            run = await _onecompiler_request(lang, code, [tc["stdin"] for tc in tests])
            results = []
            for idx, (tc, entry) in enumerate(zip(tests, run["batch"]), start=1):
                if entry["exception"] or entry["stderr"] or entry["exit_status"] not in (None, 0):
                    status = _exception_category(entry["exception"]) if entry["exception"] else "runtime_error"
                    results.append({"case": idx, "input": tc["stdin"], "expected": tc["expected_output"], "actual": entry["stdout"],
                                    "error": "Program execution failed.", "execution_status": status, "passed": False})
                    continue
                actual = canonicalize_stdio(entry["stdout"], contract.comparison_mode)
                expected = canonicalize_stdio(tc["expected_output"], contract.comparison_mode)
                results.append({"case": idx, "input": tc["stdin"], "expected": expected, "actual": actual, "error": None,
                                "execution_status": "executed", "passed": compare_values(actual, expected, contract.comparison_mode,
                                    abs_tol=contract.absolute_tolerance, rel_tol=contract.relative_tolerance)})
            result = {"success": True, "supported": True, "total": len(results), "passed": sum(row["passed"] for row in results),
                      "all_passed": bool(results) and all(row["passed"] for row in results), "results": results,
                      "execution_status": "completed", "provider": "onecompiler", "stderr": ""}
            return result
        except OneCompilerFailure as exc:
            return {**base, "error": exc.public_message, "execution_status": exc.category, "provider": "onecompiler", "runner_error": True}
        except Exception:
            logger.warning("[code_execution] stdin/stdout evaluation failed")
            return {**base, "error": "Unable to evaluate the runner response.", "execution_status": "malformed_response", "provider": "onecompiler", "runner_error": True}

    if lang == "python":
        harness = _PY_HARNESS.format(
            user_code=code,
            tests_json=json.dumps(tests),
            fn_name=fn_name,
            input_adapter=contract.input_adapter,
            output_adapter=contract.output_adapter,
        )
    elif lang == "javascript":
        harness = _JS_HARNESS.format(
            user_code=json.dumps(code),
            tests_json=json.dumps(tests),
            fn_name=json.dumps(fn_name),
            input_adapter=json.dumps(contract.input_adapter),
            output_adapter=json.dumps(contract.output_adapter),
        )
    else:
        return {
            "success": False,
            "supported": False,
            "error": (
                f"Server-side execution for '{language}' is not available yet. "
                "Python and JavaScript are supported."
            ),
            "results": [],
            "total": len(tests),
            "passed": 0,
            "all_passed": False,
        }

    try:
        run = await _run_onecompiler(lang, harness)
    except OneCompilerFailure as exc:
        logger.warning("[code_execution] OneCompiler request failed: %s", exc.category)
        return {
            "success": False,
            "supported": True,
            "error": exc.public_message,
            "results": [],
            "total": len(tests),
            "passed": 0,
            "all_passed": False,
            "provider": "onecompiler",
            "runner_error": True,
            "execution_status": exc.category,
        }
    except Exception:
        logger.warning("[code_execution] OneCompiler request failed: unexpected provider error")
        return {**base, "error": "The code runner could not complete the request.", "execution_status": "infrastructure_error", "provider": "onecompiler", "runner_error": True}

    if run.get("exception") or run.get("exit_status") not in (None, 0):
        category = _exception_category(run.get("exception") or "runtime error")
        return {**base, "error": "Program compilation or execution failed.", "execution_status": category, "provider": "onecompiler", "runner_error": True}
    if run.get("stderr"):
        # Function harnesses report compilation/runtime exceptions on stdout as JSON,
        # while raw provider diagnostics indicate an actual runner-level failure.
        if not (run.get("stdout") or "").strip():
            return {**base, "error": "Program compilation or execution failed.", "execution_status": "compile_or_runtime_error", "provider": "onecompiler", "runner_error": True}

    payload = _extract_json_payload(run.get("stdout") or "")
    if isinstance(payload, dict) and isinstance(payload.get("fatal"), dict):
        fatal = payload["fatal"]
        category = fatal.get("category") or "runtime_error"
        return {**base, "error": str(fatal.get("message") or "The submitted program could not be evaluated.")[:300],
                "execution_status": category, "provider": "onecompiler", "runner_error": category in {"invalid_contract", "compile_error"}}
    if not payload or "results" not in payload:
        fatal = payload.get("fatal") if isinstance(payload, dict) else None
        detail = fatal.get("message") if isinstance(fatal, dict) else None
        category = fatal.get("category") if isinstance(fatal, dict) else "malformed_response"
        public = detail or "The code runner returned no valid results."
        logger.warning("[code_execution] OneCompiler returned no results: %s", category)
        return {
            "success": False,
            "supported": True,
            "error": public[:300],
            "results": [],
            "total": len(tests),
            "passed": 0,
            "all_passed": False,
            "provider": "onecompiler",
            "stderr": run.get("stderr") or "",
            "runner_error": True,
            "execution_status": category,
        }

    normalised = _normalise_results(payload.get("results") or [], contract)
    normalised["provider"] = "onecompiler"
    normalised["stderr"] = run.get("stderr") or ""
    return normalised
