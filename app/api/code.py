"""Server-side code execution endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import get_current_user
from app.db.models import User
from app.services.code_execution_service import execute_code

router = APIRouter(prefix="/api/code", tags=["Code Execution"])


class TestCaseItem(BaseModel):
    input: str = ""
    expected_output: str = ""


class ExecuteCodeSchema(BaseModel):
    language: Optional[str] = "python"
    # Empty code is allowed here so the service can return a friendly
    # "No code submitted." result instead of a 422 from validation.
    code: str = ""
    function_name: Optional[str] = "solution"
    test_cases: List[TestCaseItem] = Field(default_factory=list)


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
    )
