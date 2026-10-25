"""Tag existing question documents with explicit legacy execution contracts.

Run once against the target Mongo database after backing it up:
    python -m scripts.migrate_question_contracts_v1

The migration never rewrites test inputs or outputs. It only identifies the
existing interpretation as compatibility version 1 so future code will not
guess what those values mean.
"""

from __future__ import annotations

import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings


LEGACY_FIELDS = {
    "execution_mode": "function",
    "input_adapter": "legacy_json_v1",
    "output_adapter": "legacy_python_v1",
    "supported_languages": ["python", "javascript"],
    "test_case_schema": "legacy_json_v1",
    "comparison_mode": "legacy_json_v1",
    "problem_version": 1,
}


async def migrate() -> int:
    if not settings.MONGO_URL:
        raise RuntimeError("MONGO_URL is required")
    client = AsyncIOMotorClient(settings.MONGO_URL)
    try:
        collection = client[settings.MONGO_DB_NAME]["questions"]
        result = await collection.update_many(
            {"type": "practice", "problem_version": {"$exists": False}},
            {"$set": LEGACY_FIELDS},
        )
        return result.modified_count
    finally:
        client.close()


if __name__ == "__main__":
    print(f"Tagged {asyncio.run(migrate())} legacy practice questions.")
