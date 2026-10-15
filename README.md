# Adaptive Learning Platform — Backend API

A production-ready, modular **FastAPI** backend for an AI-driven Adaptive Learning SaaS.

The platform diagnoses a student's prior knowledge via a stateful 10-turn diagnostic chat, generates a personalized concept DAG learning path (skipping known concepts), adaptively re-teaches failed concepts using different pedagogical strategies, and tracks mastery through an auditable Exponential Moving Average (EMA) engine with Item Response Theory (IRT) difficulty scaling.

---

## 🏗️ 3-Path Architecture Overview

```
                            ┌───────────────────────────────────┐
                            │     FastAPI REST Controllers      │
                            └─────────────────┬─────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
   [Stateful Cycles]                [Stateless Single-Shot]         [Deterministic Path (No AI)]
┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────────┐
│   LangGraph Engine    │         │   LangChain Engine    │         │ Mastery & Eval Engine     │
│ (Diagnostic & Reteach)│         │ (Path / Question Gen) │         │ (EMA Math, Sandboxing)    │
└──────────┬────────────┘         └───────────┬───────────┘         └─────────────┬─────────────┘
           │                                  │                                   │
           └──────────────────┬───────────────┘                                   │
                              ▼                                                   │
               ┌─────────────────────────────┐                                    │
               │         LLM Gateway         │                                    │
               │   (Gemini ➔ Groq Fallback)  │                                    │
               └──────────────┬──────────────┘                                    │
                              │                                                   │
                              ▼ (Structured JSON / Pydantic)                      │
               ┌─────────────────────────────┐                                    │
               │  Application Service Layer  │◀───────────────────────────────────┘
               └──────────────┬──────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  PostgreSQL  │    │   MongoDB    │    │    Redis     │
   │ (Mastery,    │    │ (Concept DAG,│    │ (Checkpoints,│
   │  Attempts,   │    │  Questions,  │    │  In-flight   │
   │  Auth)       │    │  Paths)      │    │  Rate Limits)│
   └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Stateful Cycles Path (LangGraph)**:
   - **Diagnostic Probe**: 10-turn adaptive Q&A cycle (`DiagnosticState`) with early completion criteria ($\ge 0.8$ or $\le 0.2$ confidence across core concepts) and complete beginner opt-out.
   - **Adaptive Re-teaching Loop**: Stateful remediation graph (`ReteachState`) trying unused pedagogical strategies (`analogy` $\rightarrow$ `visual_diagram` $\rightarrow$ `worked_example` $\rightarrow$ `interactive_exercise`) or triggering prerequisite regression.
2. **Stateless Single-Shot Path (LangChain)**:
   - **Curriculum Generation**: One-shot concept DAG generation with topological cycle validation.
   - **Question Bank Generation**: On-demand batch question generation when bank pool drops below 3.
   - **Rubric Evaluation**: Multi-criteria code quality and conceptual breakdown scoring.
3. **Deterministic Path (No AI)**:
   - **Mastery Calculation**: Exponential Moving Average (`EMA = 0.6 * score + 0.4 * prev_mastery`) scaled by IRT difficulty rating (1 to 5).
   - **Code & Math Sandboxing**: Exact string matching, SymPy symbolic expression equivalence, numeric tolerance, and Pyodide/Piston API execution checks.
   - **Level Unlocks**: Deterministic thresholds for Beginner, Intermediate, Advanced, and Expert status badges.

---

## 🗄️ Database Design & Storage Split

### 1. PostgreSQL (Relational Core)
- `users`: User identity, email, name, bcrypt hashed passwords.
- `diagnostic_sessions`: Diagnostic chat results, final per-concept `knowledge_profile` map.
- `concept_mastery`: Single source of truth for user concept mastery, status (`known`, `partial`, `learning`, `mastered`), attempt counts, and failures.
- `attempts`: Historical log of user question submissions, scores, and evaluation breakdowns.

### 2. MongoDB (Document Store)
- `concept_graphs`: Generate-once, reuse-forever topic concept DAGs with `aliases` for rapidfuzz matching.
- `learning_paths`: Personalized learning paths with node actions (`skip_teach_verify_only`, `refresher`, `full_teach`).
- `questions`: Shared question pool tagged by `concept_id`, `difficulty` (1 to 5), `type`, and `execution_method`.

### 3. Redis (Cache & Checkpoints)
- LangGraph thread state checkpoints, rate-limiting, and ephemeral diagnostic session cache.

---

## ⚙️ Environment Configuration

Create or update your `.env` file in the root directory:

```env
# LLM Providers
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Databases
POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/adaptive_learn
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=adaptive_learn
REDIS_URL=redis://localhost:6379/0

# App Settings
JWT_SECRET=super-secret-jwt-key-change-in-production-123456789
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ENV=development
```

> **Note**: If PostgreSQL, MongoDB, or Redis instances are not running locally, the backend automatically activates fallback stores (SQLite database fallback, In-Memory Mongo fallback, In-Memory Redis fallback, and LLM Gateway mock provider) so the backend remains fully operational for development and testing.

---

## 🚀 Running the Project

### Option A: Running locally with Python

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI dev server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. Open [http://localhost:8000/docs](http://localhost:8000/docs) to access the interactive Swagger OpenAPI documentation.

---

### Option B: Running with Docker Compose

To launch PostgreSQL, MongoDB, Redis, and the FastAPI Backend together:

```bash
docker compose up -d
```
docker compose up -d; docker compose logs -f backend


To stop containers:
```bash
docker compose down
```

---

## 🧪 Running Automated Tests

Run the comprehensive end-to-end verification test suite:

```bash
python test_backend.py
```

This verifies:
- Health check endpoint `/health`
- User Registration & JWT Authentication (`/api/auth/register`, `/api/auth/login`, `/api/auth/me`)
- 10-turn Diagnostic Chat Loop & Complete Beginner Opt-Out
- Concept Graph DAG generation & Personalized Learning Path creation
- Shared Question Pool generation & caching
- Multi-signal answer evaluation (Deterministic Sandbox/SymPy + LLM Rubric)
- Exponential Moving Average (EMA) mastery updates & level unlock thresholds
- Adaptive Re-teaching strategy node execution

---

## 🔌 API Reference Matrix

| Endpoint | Method | Description | Auth Required? |
| --- | --- | --- | --- |
| `/health` | GET | Server health check status | No |
| `/api/auth/register` | POST | Register new user & return JWT token | No |
| `/api/auth/login` | POST | Authenticate user credentials | No |
| `/api/auth/me` | GET | Retrieve authenticated user profile | Yes |
| `/api/diagnostic/start` | POST | Start 10-turn stateful diagnostic probe | Yes |
| `/api/diagnostic/answer` | POST | Submit turn answer & receive next question/profile | Yes |
| `/api/diagnostic/opt-out` | POST | Opt-out as complete beginner | Yes |
| `/api/diagnostic/{session_id}` | GET | Get diagnostic session state | Yes |
| `/api/learning/generate-path` | POST | Generate personalized concept learning path | Yes |
| `/api/learning/{user_id}/{topic}` | GET | Fetch user's current learning path | Yes |
| `/api/learning/reteach` | POST | Trigger adaptive re-teaching strategy loop | Yes |
| `/api/questions/generate` | POST | Fetch or generate concept question | Yes |
| `/api/attempts/submit` | POST | Submit attempt & compute multi-signal mastery | Yes |
| `/api/progress/{user_id}/{topic}` | GET | Get overall mastery & level unlock breakdown | Yes |


docker compose up -d --force-recreate

docker exec -it adaptive_mongo mongosh