import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from pydantic import BaseModel, Field

Base = declarative_base()

class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as string.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            else:
                return value

# SQLite & Postgres cross-compatibility types
JSONType = JSONB().with_variant(JSON(), 'sqlite')
UUIDType = GUID()

# ==========================================
# 1. POSTGRESQL SCHEMA (Relational / Core)
# ==========================================

class User(Base):
    __tablename__ = "users"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    diagnostic_sessions = relationship("DiagnosticSession", back_populates="user", cascade="all, delete-orphan")
    concept_mastery = relationship("ConceptMastery", back_populates="user", cascade="all, delete-orphan")
    attempts = relationship("Attempt", back_populates="user", cascade="all, delete-orphan")
    user_topics = relationship("UserTopic", back_populates="user", cascade="all, delete-orphan")

class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDType, ForeignKey("users.id"), nullable=False)
    topic = Column(String, nullable=False, index=True)
    knowledge_profile = Column(JSONType, nullable=False, default=dict)
    turns_used = Column(Integer, nullable=False, default=0)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="diagnostic_sessions")

class ConceptMastery(Base):
    __tablename__ = "concept_mastery"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDType, ForeignKey("users.id"), nullable=False)
    topic = Column(String, nullable=False, index=True)
    concept_id = Column(String, nullable=False, index=True)
    mastery = Column(Float, nullable=False, default=0.0)
    attempts = Column(Integer, nullable=False, default=0)
    failures = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="learning")  # known, partial, learning, mastered
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="concept_mastery")

class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDType, ForeignKey("users.id"), nullable=False)
    question_id = Column(String, nullable=False, index=True)
    concept_id = Column(String, nullable=False, index=True)
    answer = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    evaluation_breakdown = Column(JSONType, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="attempts")

class UserTopic(Base):
    __tablename__ = "user_topics"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDType, ForeignKey("users.id"), nullable=False, index=True)
    topic_slug = Column(String, nullable=False, index=True)
    canonical_topic = Column(String, nullable=False)
    current_level = Column(String, nullable=False, default="beginner")
    status = Column(String, nullable=False, default="active")
    completed_nodes = Column(JSONType, nullable=False, default=list)
    completed_node_ids = Column(JSONType, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="user_topics")

# ==========================================
# 2. MONGODB DOCUMENT SCHEMAS (Pydantic)
# ==========================================

class ConceptNodeSchema(BaseModel):
    id: str
    depends_on: List[str] = Field(default_factory=list)
    level: str  # beginner, intermediate, advanced, expert
    concept_type: str  # code, math, conceptual, other

class ConceptGraphDocument(BaseModel):
    id: str = Field(alias="_id")  # topic slug, e.g. "react"
    topic: str
    aliases: List[str] = Field(default_factory=list)
    concepts: List[ConceptNodeSchema]

    class Config:
        populate_by_name = True

class PathNodeSchema(BaseModel):
    concept_id: str
    level: str
    status: str  # known, locked, learning, mastered
    action: str  # skip_teach_verify_only, full_teach, refresher
    estimated_minutes: int = 15
    depends_on: List[str] = Field(default_factory=list)

class LearningPathDocument(BaseModel):
    id: str = Field(alias="_id")  # e.g., lp_001
    user_id: str
    topic: str
    target_level: str  # beginner, intermediate, advanced, expert
    generated_from_diagnostic_session: Optional[str] = None
    nodes: List[PathNodeSchema]
    current_node: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True

class TestCaseSchema(BaseModel):
    input: str
    expected_output: str

class QuestionDocument(BaseModel):
    id: str = Field(alias="_id")
    topic: str
    concept_id: str
    concept_type: str  # code, math, conceptual, other
    difficulty: int = Field(ge=1, le=5)
    type: str  # mcq, true_false, short_answer, open_ended, practice
    question: str
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    rubric: Optional[str] = None
    test_cases: Optional[List[TestCaseSchema]] = None
    expected_answer: Optional[str] = None
    execution_method: str  # browser_pyodide, browser_js, piston_api, sympy, numeric_tolerance, llm_rubric

    class Config:
        populate_by_name = True
