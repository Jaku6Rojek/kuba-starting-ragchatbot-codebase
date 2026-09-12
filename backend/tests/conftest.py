"""Shared fixtures for the backend test suite.

These tests are hermetic by default: the Anthropic API and ChromaDB are
replaced with mocks so the *code* under test is exercised in isolation from
external services. The only tests that touch the real API are marked
``integration`` (see pyproject.toml) and are skipped unless explicitly run.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Backend uses flat imports (`from config import config`). Make sure the
# backend/ directory is importable even if pytest is invoked oddly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults  # noqa: E402


# --------------------------------------------------------------------------- #
# Vector-store / search-result fixtures
# --------------------------------------------------------------------------- #

COURSE_TITLE = "MCP: Build Rich-Context AI Apps with Anthropic"


@pytest.fixture
def sample_search_results():
    """Two content chunks from the same course, different lessons."""
    return SearchResults(
        documents=[
            "MCP is an open protocol for connecting AI apps to context.",
            "Servers expose tools, resources, and prompts to clients.",
        ],
        metadata=[
            {"course_title": COURSE_TITLE, "lesson_number": 1, "chunk_index": 0},
            {"course_title": COURSE_TITLE, "lesson_number": 2, "chunk_index": 1},
        ],
        distances=[0.12, 0.23],
    )


@pytest.fixture
def content_only_results():
    """A chunk with no lesson number (course-level citation)."""
    return SearchResults(
        documents=["Course-level overview text."],
        metadata=[{"course_title": COURSE_TITLE, "lesson_number": None, "chunk_index": 0}],
        distances=[0.15],
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults.empty("No course found matching 'ghost course'")


@pytest.fixture
def mock_vector_store():
    """A MagicMock standing in for VectorStore, with link lookups stubbed."""
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson-link"
    store.get_course_link.return_value = "https://example.com/course-link"
    return store


# --------------------------------------------------------------------------- #
# Anthropic response builders (mimic the shape of anthropic SDK objects)
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_text_response():
    """Build a fake 'end_turn' response whose content[0].text is `text`."""

    def _make(text: str):
        block = SimpleNamespace(type="text", text=text)
        return SimpleNamespace(stop_reason="end_turn", content=[block])

    return _make


@pytest.fixture
def make_tool_use_response():
    """Build a fake 'tool_use' response requesting one tool call."""

    def _make(tool_name: str, tool_input: dict, tool_id: str = "toolu_123"):
        block = SimpleNamespace(
            type="tool_use", name=tool_name, input=tool_input, id=tool_id
        )
        return SimpleNamespace(stop_reason="tool_use", content=[block])

    return _make
