"""Server-side code execution endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.core.dependencies import get_current_user
from app.db.models import User
from app.services.code_execution_service import execute_code

router = APIRouter(prefix="/api/code", tags=["Code Execution"])


class TestCaseItem(BaseModel):
    input: Any = ""
    expected_output: Any = ""
    stdin: Optional[str] = None


class ExecuteCodeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Optional[str] = "python"
    # Empty code is allowed here so the service can return a friendly
    # "No code submitted." result instead of a 422 from validation.
    code: str = ""
    function_name: Optional[str] = "solution"
    test_cases: List[TestCaseItem] = Field(default_factory=list)
    execution_mode: Optional[str] = None
    function_signature: Optional[str] = None
    input_adapter: Optional[str] = None
    output_adapter: Optional[str] = None
    supported_languages: Optional[List[str]] = None
    test_case_schema: Optional[str] = None
    comparison_mode: Optional[str] = None
    problem_version: Optional[int] = None
    absolute_tolerance: Optional[float] = None
    relative_tolerance: Optional[float] = None


@router.post("/execute")
async def execute(
    payload: ExecuteCodeSchema,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run submitted code against generated test cases in a sandbox."""
    return await execute_code(
        language=payload.language,
        code=payload.code,
        test_cases=[tc.model_dump() for tc in payload.test_cases],
        function_name=payload.function_name or "solution",
        contract_data={key: value for key, value in payload.model_dump(exclude_unset=True).items()
                       if key in {"execution_mode", "function_signature", "input_adapter", "output_adapter",
                                  "supported_languages", "test_case_schema", "comparison_mode", "problem_version",
                                  "absolute_tolerance", "relative_tolerance"}}
        or None,
    )
