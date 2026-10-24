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
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

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
import json, sys, inspect, io, traceback
import typing
from typing import *
import collections
from collections import *
import math, heapq, bisect, random

__user_code__ = {user_code!r}
__tests__ = json.loads({tests_json!r})
__fn_name__ = {fn_name!r}

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
except Exception:
    print(json.dumps({{"fatal": traceback.format_exc()}}))
    sys.exit(0)

fn = ns.get(__fn_name__)
if not callable(fn):
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

if not callable(fn):
    candidates = [
        v for k, v in ns.items()
        if callable(v) and not k.startswith("_")
        and getattr(v, "__module__", None) in (None, "__main__")
    ]
    fn = candidates[-1] if candidates else None

if fn is None:
    print(json.dumps({{"fatal": "function '%s' not found" % __fn_name__}}))
    sys.exit(0)

try:
    sig = inspect.signature(fn)
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    arity = len(positional)
    param_names = [p.name for p in positional]
except Exception:
    arity = 1
    param_names = []

try:
    __hints__ = typing.get_type_hints(fn)
except Exception:
    __hints__ = {{}}

def _hint_name(name):
    h = __hints__.get(name)
    if h is None:
        return ""
    if typing.get_origin(h) is typing.Union:
        args = [a for a in typing.get_args(h) if a is not type(None)]
        h = args[0] if args else None
    return getattr(h, "__name__", "") or ""

def _link(vals, cls):
    head = None
    tail = None
    for v in vals:
        node = v if isinstance(v, cls) else cls(v)
        if head is None:
            head = node
        else:
            tail.next = node
        tail = node
    return head

def _tree(vals, cls):
    if not vals or vals[0] is None:
        return None
    root = cls(vals[0])
    queue = collections.deque([root])
    i = 1
    while queue and i < len(vals):
        node = queue.popleft()
        lv = vals[i]
        i += 1
        if lv is not None:
            node.left = cls(lv)
            queue.append(node.left)
        if i < len(vals):
            rv = vals[i]
            i += 1
            if rv is not None:
                node.right = cls(rv)
                queue.append(node.right)
    return root

def _build(value, name):
    """Decode a JSON test input using the learner's OWN node classes.

    Nothing is defined by the runner: if the submitted code does not define
    ListNode / TreeNode, the raw value is passed through untouched.
    """
    if not isinstance(value, (list, tuple)):
        return value
    tn = _hint_name(name)
    cls = ns.get(tn) if tn in ("ListNode", "TreeNode") else None
    if not isinstance(cls, type):
        return value
    try:
        return _link(list(value), cls) if tn == "ListNode" else _tree(list(value), cls)
    except Exception:
        return value

def _ser(value):
    """JSON-safe serialisation of the return value."""
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        try:
            return [_ser(v) for v in value]
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "val"):
        return value.val
    return str(value)

results = []
for idx, tc in enumerate(__tests__):
    raw = tc.get("input", "")
    expected = tc.get("expected_output", "")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    except Exception:
        parsed = raw
    try:
        if isinstance(parsed, dict):
            kwargs = {{ n: _build(parsed.get(n), n) for n in param_names }} if param_names else parsed
            actual = fn(**kwargs)
        else:
            if isinstance(parsed, list) and arity > 1 and len(parsed) == arity:
                values = parsed
            else:
                values = [parsed]
            call_args = [
                _build(v, param_names[i] if i < len(param_names) else "")
                for i, v in enumerate(values)
            ]
            actual = fn(*call_args)
        results.append({{
            "case": idx + 1,
            "input": raw,
            "expected": expected,
            "actual": _ser(actual),
            "error": None,
        }})
    except Exception as exc:
        results.append({{
            "case": idx + 1,
            "input": raw,
            "expected": expected,
            "actual": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }})

print(json.dumps({{"results": results}}))
'''

_JS_HARNESS = r'''
const __userCode__ = {user_code};
const __tests__ = {tests_json};
const __fnName__ = {fn_name};

let fn;
try {{
  fn = eval(__userCode__ + "\n;" + __fnName__);
}} catch (e) {{
  console.log(JSON.stringify({{ fatal: String(e) }}));
  process.exit(0);
}}
if (typeof fn !== "function") {{
  const names = Object.keys(globalThis).filter((k) => typeof globalThis[k] === "function");
  fn = names.length ? globalThis[names[names.length - 1]] : null;
}}
if (typeof fn !== "function") {{
  console.log(JSON.stringify({{ fatal: "function not found" }}));
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
    let actual;
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {{
      actual = fn(...Object.values(parsed));
    }} else if (Array.isArray(parsed) && fn.length > 1 && parsed.length === fn.length) {{
      actual = fn(...parsed);
    }} else {{
      actual = fn(parsed);
    }}
    results.push({{ case: i + 1, input: tc.input, expected: tc.expected_output, actual, error: null }});
  }} catch (e) {{
    results.push({{ case: i + 1, input: tc.input, expected: tc.expected_output, actual: null, error: String(e) }});
  }}
}}
console.log(JSON.stringify({{ results }}));
'''


def _normalise_language(language: Optional[str]) -> Optional[str]:
    lang = (language or "").strip().lower()
    return lang if lang in _ONECOMPILER_LANGS else None


def _normalise_results(raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for idx, item in enumerate(raw_results or []):
        actual = item.get("actual")
        expected = item.get("expected", item.get("expected_output", ""))
        error = item.get("error")
        passed = item.get("passed")
        if passed is None:
            passed = bool(error is None and _loose_equal(actual, expected))
        results.append({
            "case": item.get("case", idx + 1),
            "passed": bool(passed),
            "input": item.get("input", ""),
            "expected": expected,
            "actual": actual,
            "error": error,
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
    }


def _coerce_json(value: Any) -> Any:
    """Parse a JSON-looking string so '42' == 42 and '[0, 1]' == [0, 1]."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if s[0] in '[{"' or s in ("true", "false", "null"):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def _loose_equal(actual: Any, expected: Any) -> bool:
    a = _coerce_json(expected)
    b = _coerce_json(actual)
    if a is None and b is None:
        return True
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        try:
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        except Exception:
            return str(a).strip() == str(b).strip()
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


async def _run_onecompiler(language: str, code: str) -> Dict[str, Any]:
    """POST a single-file program to the OneCompiler code-execution API.

    Request:  {"properties": {"language": ..., "files": [{"name", "content"}]}}
    Response: {"stdout", "stderr", "exception", "compilationTime", ...}
    Unknown languages return a plain-text error body (not JSON).
    """
    lang_id, filename = _ONECOMPILER_LANGS[language]
    payload = {
        "properties": {
            "language": lang_id,
            "files": [{"name": filename, "content": code}],
        }
    }
    headers = {"Content-Type": "application/json"}
    if settings.ONE_COMPILER_API:
        headers["X-API-Key"] = settings.ONE_COMPILER_API

    async with httpx.AsyncClient(timeout=settings.ONE_COMPILER_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            settings.ONE_COMPILER_API_URL, json=payload, headers=headers
        )
        resp.raise_for_status()
        text = resp.text

    try:
        data = resp.json()
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {"raw": text}

    return {
        "stdout": data.get("stdout") or "",
        "stderr": data.get("stderr") or "",
        "exception": data.get("exception"),
        "raw": data.get("raw"),
        "execution_time": data.get("executionTime"),
        "memory_used": data.get("memoryUsed"),
    }


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
) -> Dict[str, Any]:
    """Execute ``code`` against ``test_cases`` in an isolated sandbox."""
    if not settings.CODE_EXECUTION_ENABLED:
        return {"success": False, "supported": False, "error": "Code execution is disabled.", "results": [], "total": 0, "passed": 0, "all_passed": False}

    lang = _normalise_language(language)
    if lang is None:
        return {"success": False, "supported": False, "error": f"Unsupported language '{language}'.", "results": [], "total": 0, "passed": 0, "all_passed": False}

    if not (code or "").strip():
        return {"success": False, "supported": True, "error": "No code submitted.", "results": [], "total": 0, "passed": 0, "all_passed": False}

    tests = test_cases or []
    fn_name = (function_name or "solution").strip() or "solution"

    if lang == "python":
        harness = _PY_HARNESS.format(
            user_code=code,
            tests_json=json.dumps(tests),
            fn_name=fn_name,
        )
    elif lang == "javascript":
        harness = _JS_HARNESS.format(
            user_code=json.dumps(code),
            tests_json=json.dumps(tests),
            fn_name=json.dumps(fn_name),
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
    except Exception as exc:
        logger.warning("[code_execution] OneCompiler request failed: %s", exc)
        return {
            "success": False,
            "supported": True,
            "error": f"Execution service error: {exc}",
            "results": [],
            "total": len(tests),
            "passed": 0,
            "all_passed": False,
            "provider": "onecompiler",
            "runner_error": True,
        }

    payload = _extract_json_payload(run.get("stdout") or "")
    if not payload or "results" not in payload:
        detail = run.get("stderr") or run.get("exception")
        if not detail and isinstance(payload, dict):
            detail = payload.get("fatal")
        if not detail:
            detail = run.get("raw")
        detail = str(detail or "Execution returned no results.").strip()
        logger.warning("[code_execution] OneCompiler returned no results: %s", detail[:500])
        return {
            "success": False,
            "supported": True,
            "error": detail[:1500],
            "results": [],
            "total": len(tests),
            "passed": 0,
            "all_passed": False,
            "provider": "onecompiler",
            "stderr": run.get("stderr") or "",
            "runner_error": True,
        }

    normalised = _normalise_results(payload.get("results") or [])
    normalised["provider"] = "onecompiler"
    normalised["stderr"] = run.get("stderr") or ""
    return normalised
