"""Tests for AIGenerator (backend/ai_generator.py).

Goal: evaluate whether AIGenerator correctly calls for tools — i.e. it passes
the tool schema to Claude on the first turn, dispatches tool_use requests to
the ToolManager, and makes a tools-free follow-up call to produce the answer.

The Anthropic client is patched, so no network/API key is required.
"""

from unittest.mock import MagicMock, patch

import anthropic
import httpx

from ai_generator import AIGenerator


@patch("ai_generator.anthropic.Anthropic")
def test_direct_answer_when_no_tool_use(mock_anthropic, make_text_response):
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.return_value = make_text_response("Paris")

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response("What is the capital of France?")

    assert out == "Paris"
    assert client.messages.create.call_count == 1


@patch("ai_generator.anthropic.Anthropic")
def test_tools_and_choice_passed_on_first_call(mock_anthropic, make_text_response):
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.return_value = make_text_response("answer")

    gen = AIGenerator("fake-key", "fake-model")
    tools = [{"name": "search_course_content"}]
    gen.generate_response("hi", tools=tools)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == {"type": "auto"}


@patch("ai_generator.anthropic.Anthropic")
def test_tool_use_dispatches_to_tool_manager(
    mock_anthropic, make_tool_use_response, make_text_response
):
    client = MagicMock()
    mock_anthropic.return_value = client
    # 1st call: Claude asks to search. 2nd call: Claude answers.
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "MCP"}),
        make_text_response("MCP is a protocol."),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results text"

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "Tell me about MCP",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert out == "MCP is a protocol."
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="MCP"
    )
    assert client.messages.create.call_count == 2


@patch("ai_generator.anthropic.Anthropic")
def test_followup_call_has_no_tools(
    mock_anthropic, make_tool_use_response, make_text_response
):
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "MCP"}),
        make_text_response("final"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator("fake-key", "fake-model")
    gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    second_kwargs = client.messages.create.call_args_list[1].kwargs
    assert "tools" not in second_kwargs
    assert "tool_choice" not in second_kwargs


@patch("ai_generator.anthropic.Anthropic")
def test_tool_result_message_structure(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """The follow-up call must include the assistant tool_use turn and a
    user turn carrying the tool_result with the matching tool_use_id."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "MCP"}, tool_id="toolu_abc"),
        make_text_response("final"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "the results"

    gen = AIGenerator("fake-key", "fake-model")
    gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    messages = client.messages.create.call_args_list[1].kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user"]

    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_abc"
    assert tool_result["content"] == "the results"


@patch("ai_generator.anthropic.Anthropic")
def test_conversation_history_included_in_system(mock_anthropic, make_text_response):
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.return_value = make_text_response("ok")

    gen = AIGenerator("fake-key", "fake-model")
    gen.generate_response("q", conversation_history="User: hi\nAssistant: hello")

    system = client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:" in system
    assert "User: hi" in system


def _api_error(message: str) -> anthropic.APIError:
    """Build a real anthropic.APIError (base class of billing/rate-limit errors)."""
    return anthropic.APIConnectionError(
        message=message, request=httpx.Request("POST", "https://api.anthropic.com")
    )


@patch("ai_generator.anthropic.Anthropic")
def test_api_error_returns_friendly_message(mock_anthropic):
    """Fix 2: when the Anthropic call fails (e.g. billing 400, outage), the
    error is caught and a friendly message is returned instead of the raw
    error propagating as a 500 / 'Query failed'."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = _api_error("credit balance is too low")

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response("q")

    assert out == AIGenerator.API_ERROR_MESSAGE


@patch("ai_generator.anthropic.Anthropic")
def test_api_error_during_tool_execution_returns_friendly_message(
    mock_anthropic, make_tool_use_response
):
    """Fix 2: a failure on the SECOND (post-tool) call is also caught."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "MCP"}),
        _api_error("service unavailable"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    assert out == AIGenerator.API_ERROR_MESSAGE


@patch("ai_generator.anthropic.Anthropic")
def test_tool_use_without_tool_manager_does_not_crash(
    mock_anthropic, make_tool_use_response
):
    """Fix 3: if the model returns tool_use but no tool_manager is wired up,
    calling .text on the tool_use block would raise AttributeError. Instead we
    return a friendly fallback."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.return_value = make_tool_use_response(
        "search_course_content", {"query": "MCP"}
    )

    gen = AIGenerator("fake-key", "fake-model")
    # tools provided, but tool_manager omitted
    out = gen.generate_response("q", tools=[{"name": "search_course_content"}])

    assert out == AIGenerator.API_ERROR_MESSAGE
