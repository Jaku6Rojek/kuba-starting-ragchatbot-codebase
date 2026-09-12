"""Tests for RAGSystem.query handling of content-related questions
(backend/rag_system.py).

Goal: evaluate the orchestration — RAGSystem must hand the tool schema and the
ToolManager to the AI generator, return the generator's answer, surface the
sources recorded by the search tool, and reset those sources afterwards.

VectorStore and AIGenerator are patched at import time so no embeddings load
and no real API call is made.
"""

from unittest.mock import MagicMock, patch

from config import config


def _build_rag():
    """Construct a RAGSystem with VectorStore and AIGenerator mocked out.

    Returns (rag, mock_generator) where mock_generator is the AIGenerator
    instance used inside the system.
    """
    with patch("rag_system.VectorStore") as MockVS, patch(
        "rag_system.AIGenerator"
    ) as MockGen:
        MockVS.return_value = MagicMock()
        mock_generator = MagicMock()
        MockGen.return_value = mock_generator
        from rag_system import RAGSystem

        rag = RAGSystem(config)
    return rag, mock_generator


def test_query_returns_generator_answer():
    rag, gen = _build_rag()
    gen.generate_response.return_value = "MCP is a protocol for context."

    answer, sources = rag.query("What is MCP?")

    assert answer == "MCP is a protocol for context."


def test_query_passes_tools_and_tool_manager():
    rag, gen = _build_rag()
    gen.generate_response.return_value = "answer"

    rag.query("What is MCP?")

    kwargs = gen.generate_response.call_args.kwargs
    assert kwargs["tools"] == rag.tool_manager.get_tool_definitions()
    assert kwargs["tool_manager"] is rag.tool_manager


def test_query_registers_search_tool():
    rag, _ = _build_rag()
    names = {d["name"] for d in rag.tool_manager.get_tool_definitions()}
    assert "search_course_content" in names


def test_query_surfaces_sources_from_search_tool():
    rag, gen = _build_rag()
    gen.generate_response.return_value = "answer"
    # Simulate the search tool having recorded sources during the call.
    injected = [{"text": "MCP - Lesson 1", "link": "https://example.com/l1"}]
    rag.search_tool.last_sources = injected

    _, sources = rag.query("What is MCP?")

    assert sources == injected


def test_query_resets_sources_after_returning():
    rag, gen = _build_rag()
    gen.generate_response.return_value = "answer"
    rag.search_tool.last_sources = [{"text": "MCP - Lesson 1", "link": None}]

    rag.query("What is MCP?")

    # After a query, sources must be cleared so they don't leak into the next.
    assert rag.tool_manager.get_last_sources() == []


def test_query_wraps_prompt_for_ai():
    rag, gen = _build_rag()
    gen.generate_response.return_value = "answer"

    rag.query("What is MCP?")

    passed_query = gen.generate_response.call_args.kwargs["query"]
    assert "What is MCP?" in passed_query


def test_query_handles_api_outage_gracefully():
    """After Fix 2, an Anthropic outage/billing failure no longer crashes the
    request: AIGenerator returns a friendly message, so RAGSystem.query returns
    it with empty sources instead of raising a 500 / 'Query failed'."""
    from ai_generator import AIGenerator

    rag, gen = _build_rag()
    gen.generate_response.return_value = AIGenerator.API_ERROR_MESSAGE

    answer, sources = rag.query("What is MCP?")

    assert answer == AIGenerator.API_ERROR_MESSAGE
    assert sources == []
