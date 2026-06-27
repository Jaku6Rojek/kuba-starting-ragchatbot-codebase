# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses **uv** (not pip) for dependency management and Python 3.13.

```bash
uv sync                                              # Install dependencies
./run.sh                                             # Start the app (runs from backend/, port 8000)
cd backend && uv run uvicorn app:app --reload --port 8000   # Manual start
```

- Web UI: `http://localhost:8000` — API docs: `http://localhost:8000/docs`
- Requires a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`).
- There is currently no test suite, linter config, or build step.

## Architecture

A Retrieval-Augmented Generation (RAG) system for answering questions about course materials. FastAPI backend (`backend/`) + vanilla JS frontend (`frontend/`) served as static files by the same app.

**Key design point — tool-based retrieval, not classic RAG.** Rather than always retrieving context and stuffing it into the prompt, the system gives Claude a `search_course_content` tool and lets the model *decide* whether to search. General-knowledge questions are answered directly; course-specific questions trigger a search. This logic lives in the system prompt in `ai_generator.py` and the agentic loop in `AIGenerator._handle_tool_execution`.

### Query flow diagram

```
  BROWSER  ── POST /api/query { query, session_id } ──►          frontend/script.js
     │
     ▼
  FastAPI ENDPOINT  ── create session if null ── rag_system.query()      backend/app.py
     │
     ▼
  ORCHESTRATOR                                              backend/rag_system.py
     • history ◄── SessionManager      • tool schema ◄── ToolManager
     • AIGenerator.generate_response(query, history, tools)
     │
   ╔═▼══════════════════ AGENTIC LOOP ════════════════════ backend/ai_generator.py ═╗
   ║  [M] Claude call #1 (WITH tools): "do I need to search?"                        ║
   ║        │                                                                        ║
   ║        ├─ stop_reason ≠ tool_use ──────────────► answer directly (general)      ║
   ║        │                                                                        ║
   ║        └─ stop_reason = tool_use (course-specific)                              ║
   ║                │                                                                ║
   ║                ▼  [R] CourseSearchTool.execute()      search_tools → vector_store║
   ║                     • semantic search  • stash labels → self.last_sources       ║
   ║                │ results appended to conversation                               ║
   ║                ▼  [M] Claude call #2 (WITHOUT tools) → final answer             ║
   ╚════════════════════════════════════╪═══════════════════════════════════════════╝
                                         │
                                         ▼
  ORCHESTRATOR (return path)                                backend/rag_system.py
     • sources = tool_manager.get_last_sources(); then reset()
     • SessionManager.add_exchange(session_id, query, answer)
     │
     ▼
  RESPONSE → BROWSER   { answer, sources, session_id }

  [M] = Claude / model      [R] = retrieval / vector search

  VECTOR STORE — two-step search                            backend/vector_store.py
     course_name "MCP"
        │ (1) resolve name against catalog        (2) filter content by exact title
        ▼                                              ▼
   ┌─────────────────────┐                      ┌──────────────────────────┐
   │ course_catalog       │ ───── exact title ──►│ course_content            │
   │ 1 doc/course         │                      │ chunked lesson text       │
   │ title = unique ID    │                      │ → relevant passages       │
   │ + lessons_json meta  │                      └──────────────────────────┘
   └─────────────────────┘
```

A rendered HTML version of this diagram lives at `docs/architecture.html`.

### Request flow (`/api/query`)
1. `app.py` receives the query, creates a session if needed.
2. `RAGSystem.query` (`rag_system.py`) is the orchestrator — assembles conversation history, tool definitions, and calls the AI generator.
3. `AIGenerator.generate_response` calls Claude with tools. If Claude returns `stop_reason == "tool_use"`, it executes the tool(s) via `ToolManager`, appends results, and makes a **second** Claude call **without tools** to produce the final answer. Note: only one search round-trip is supported (the follow-up call has no tools), and the system prompt enforces "one search per query maximum".
4. `CourseSearchTool.execute` (`search_tools.py`) searches the vector store and stashes UI source labels in `self.last_sources`.
5. Back in `RAGSystem.query`, sources are pulled via `tool_manager.get_last_sources()` then **reset** — sources flow out-of-band through the tool, not through the model's text response.

### Vector store (`vector_store.py`, ChromaDB)
Two separate collections:
- `course_catalog` — one entry per course; document is the course title, metadata holds instructor, links, and lessons (serialized as a `lessons_json` string since ChromaDB metadata must be scalar).
- `course_content` — the chunked lesson text used for semantic search.

**The course title is the unique ID** throughout the system (catalog doc ID, chunk metadata, filters). Search is two-step: a fuzzy `course_name` is first resolved to an exact title via semantic search against `course_catalog`, then used to filter `course_content`.

### Document ingestion (`document_processor.py`)
On startup, `app.py` loads everything from `../docs`, skipping courses whose title already exists. Expected document format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: Introduction
Lesson Link: [url]
<lesson content...>
Lesson 1: ...
```
Content is split into sentence-based chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`), and the first chunk of each lesson is prefixed with lesson context before embedding.

### Config & data models
- All tunables live in `backend/config.py` (`Config` dataclass): model name, embedding model (`all-MiniLM-L6-v2`), chunk size/overlap, `MAX_RESULTS`, `MAX_HISTORY`, and `CHROMA_PATH` (`./chroma_db`, created relative to wherever the server is launched).
- `models.py` defines the Pydantic domain models: `Course` → `Lesson[]`, and `CourseChunk` (the unit stored in the vector DB).
- `session_manager.py` keeps conversation history in-memory only (lost on restart); `MAX_HISTORY` exchanges are retained.

## Important gotchas

- **Backend modules use flat imports** (`from config import config`, `from vector_store import ...`) with no package prefix. The app must be run from within `backend/` — which is why `run.sh` does `cd backend` first.
- **The repo root `main.py` is a placeholder** ("Hello from starting-codebase") and is not the entrypoint. The real entrypoint is `backend/app.py`.
- Adding a new retrieval/tool capability means implementing the `Tool` abstract base class in `search_tools.py` and registering it on the `ToolManager` in `RAGSystem.__init__`.
