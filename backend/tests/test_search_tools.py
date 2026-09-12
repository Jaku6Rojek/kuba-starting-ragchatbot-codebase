"""Tests for CourseSearchTool.execute (backend/search_tools.py).

Goal: evaluate the *outputs* of execute() across the paths it can take —
successful results, empty results, upstream errors — and confirm it passes
filters through to the vector store and records UI sources correctly.

The vector store is mocked, so these tests isolate the tool's own logic from
ChromaDB and embeddings.
"""

from search_tools import CourseSearchTool, ToolManager


# --------------------------------------------------------------------------- #
# get_tool_definition
# --------------------------------------------------------------------------- #


def test_tool_definition_shape(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    defn = tool.get_tool_definition()

    assert defn["name"] == "search_course_content"
    assert "query" in defn["input_schema"]["properties"]
    assert defn["input_schema"]["required"] == ["query"]


# --------------------------------------------------------------------------- #
# Successful search
# --------------------------------------------------------------------------- #


def test_execute_returns_formatted_results(mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="What is MCP?")

    # Each chunk is rendered with a bracketed course/lesson header.
    assert "[MCP: Build Rich-Context AI Apps with Anthropic - Lesson 1]" in out
    assert "[MCP: Build Rich-Context AI Apps with Anthropic - Lesson 2]" in out
    assert "MCP is an open protocol" in out
    assert "Servers expose tools" in out


def test_execute_records_sources_with_links(mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="What is MCP?")

    assert len(tool.last_sources) == 2
    assert tool.last_sources[0]["text"].endswith("Lesson 1")
    assert tool.last_sources[0]["link"] == "https://example.com/lesson-link"
    # A lesson-scoped source looks up the lesson link, not the course link.
    mock_vector_store.get_lesson_link.assert_called()


def test_execute_course_only_source_falls_back_to_course_link(
    mock_vector_store, content_only_results
):
    mock_vector_store.search.return_value = content_only_results
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="overview")

    assert tool.last_sources[0]["link"] == "https://example.com/course-link"
    mock_vector_store.get_course_link.assert_called_once()


# --------------------------------------------------------------------------- #
# Filters are forwarded to the vector store
# --------------------------------------------------------------------------- #


def test_execute_forwards_course_and_lesson_filters(
    mock_vector_store, sample_search_results
):
    mock_vector_store.search.return_value = sample_search_results
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="architecture", course_name="MCP", lesson_number=2)

    mock_vector_store.search.assert_called_once_with(
        query="architecture", course_name="MCP", lesson_number=2
    )


# --------------------------------------------------------------------------- #
# Empty results
# --------------------------------------------------------------------------- #


def test_execute_empty_results_message(mock_vector_store, empty_search_results):
    mock_vector_store.search.return_value = empty_search_results
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="nothing here")

    assert out == "No relevant content found."


def test_execute_empty_results_includes_filter_context(
    mock_vector_store, empty_search_results
):
    mock_vector_store.search.return_value = empty_search_results
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="x", course_name="MCP", lesson_number=9)

    assert "in course 'MCP'" in out
    assert "in lesson 9" in out


# --------------------------------------------------------------------------- #
# Error propagation
# --------------------------------------------------------------------------- #


def test_execute_propagates_store_error(mock_vector_store, error_search_results):
    mock_vector_store.search.return_value = error_search_results
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="x", course_name="ghost course")

    assert out == "No course found matching 'ghost course'"


# --------------------------------------------------------------------------- #
# ToolManager wiring
# --------------------------------------------------------------------------- #


def test_tool_manager_executes_registered_tool(
    mock_vector_store, sample_search_results
):
    mock_vector_store.search.return_value = sample_search_results
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    out = manager.execute_tool("search_course_content", query="What is MCP?")

    assert "MCP is an open protocol" in out


def test_tool_manager_unknown_tool(mock_vector_store):
    manager = ToolManager()
    out = manager.execute_tool("does_not_exist", query="x")
    assert out == "Tool 'does_not_exist' not found"


def test_tool_manager_get_and_reset_sources(
    mock_vector_store, sample_search_results
):
    mock_vector_store.search.return_value = sample_search_results
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    manager.execute_tool("search_course_content", query="What is MCP?")
    assert manager.get_last_sources()  # populated

    manager.reset_sources()
    assert manager.get_last_sources() == []
