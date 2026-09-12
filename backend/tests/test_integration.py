"""End-to-end diagnostic tests against the REAL system.

These are marked ``integration`` and skipped by default (see pyproject.toml
addopts = "-m 'not integration'"). Run them explicitly to check whether the
live dependencies — the vector store and the Anthropic API — actually work:

    uv run pytest -m integration

They exist to answer the operational question "which real component is
failing?" separately from the hermetic unit tests, which prove the code logic.
"""

import os

import pytest

from config import config

# The app runs from backend/ (run.sh does `cd backend`), so CHROMA_PATH
# ("./chroma_db") is relative to backend/. pytest runs from the repo root, so
# resolve the real store path against the backend dir to stay cwd-independent.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CHROMA_PATH = os.path.join(BACKEND_DIR, "chroma_db")


@pytest.mark.integration
def test_vector_store_search_returns_content():
    """The vector store + search tool return real course content (no API)."""
    from vector_store import VectorStore
    from search_tools import CourseSearchTool

    store = VectorStore(REAL_CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
    tool = CourseSearchTool(store)

    out = tool.execute(query="What is MCP?", course_name="MCP")

    assert "No relevant content found" not in out
    assert out.strip(), "expected non-empty search results"
    assert tool.last_sources, "expected sources to be recorded"


@pytest.mark.integration
def test_real_anthropic_api_reachable():
    """A minimal real Anthropic call via the RAW client, so it still surfaces
    a real API failure (billing 400, outage). Note: AIGenerator.generate_response
    now catches API errors and returns a friendly message (Fix 2), so this
    health-check deliberately bypasses that wrapper."""
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )

    assert resp.content[0].text.strip()


@pytest.mark.integration
def test_full_content_query_end_to_end():
    """The complete path a content question takes through the app. Because
    Fix 2 degrades gracefully, we assert we got a REAL answer, not the friendly
    fallback — so this test still fails when the API is actually down."""
    from ai_generator import AIGenerator
    from rag_system import RAGSystem

    if not config.ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")

    rag = RAGSystem(config)
    answer, sources = rag.query("What is covered in lesson 1 of the MCP course?")

    assert isinstance(answer, str) and answer.strip()
    assert answer != AIGenerator.API_ERROR_MESSAGE, "API is failing (fallback returned)"
    assert "credit balance" not in answer.lower()
