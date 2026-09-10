# Query Sequence Diagram

Sequence of a user query through the RAG system, from the browser to the rendered
answer. Renders anywhere Mermaid is supported (GitHub, VS Code, most Markdown viewers).

See also `architecture.html` for the component/flow diagram, and the project `CLAUDE.md`
for the narrative walkthrough.

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser<br/>(script.js)
    participant F as FastAPI<br/>(app.py)
    participant R as RAGSystem<br/>(rag_system.py)
    participant S as SessionManager
    participant A as AIGenerator<br/>(ai_generator.py)
    participant T as ToolManager /<br/>CourseSearchTool
    participant V as VectorStore<br/>(ChromaDB)

    B->>F: POST /api/query { query, session_id }
    F->>F: create session if session_id is null
    F->>R: query(query, session_id)
    R->>S: get_conversation_history(session_id)
    S-->>R: history

    R->>A: generate_response(query, history, tools)
    Note over A: Claude call #1 (WITH tools)<br/>"do I need to search?"

    alt stop_reason == tool_use (course-specific)
        A->>T: execute_tool("search_course_content", ...)
        T->>V: search(query, course_name, lesson_number)
        Note over V: 1. resolve name → course_catalog<br/>2. filter content → course_content
        V-->>T: relevant chunks
        T-->>A: formatted results (+ stash last_sources)
        Note over A: Claude call #2 (WITHOUT tools)<br/>synthesize final answer
    else stop_reason != tool_use (general knowledge)
        Note over A: answer directly, no search
    end

    A-->>R: answer text
    R->>T: get_last_sources() then reset_sources()
    T-->>R: sources [{ text, link }]
    R->>S: add_exchange(session_id, query, answer)
    R-->>F: (answer, sources)
    F-->>B: 200 { answer, sources, session_id }
    B->>B: render markdown (marked) + collapsible "Sources"
```

## Notes

- **`alt / else`** is the defining branch: the `tool_use` path makes two Claude calls
  with a search between them; the general-knowledge path is a single call, no search.
- **Sources flow out-of-band** — stashed on `CourseSearchTool.last_sources` during the
  search, then pulled via `get_last_sources()` and cleared with `reset_sources()` *after*
  the answer returns, never through Claude's text response.
- **History** is read before generation and written after, so the current turn is not in
  its own context.
