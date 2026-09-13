"""Tests for AIGenerator (backend/ai_generator.py).

Goal: evaluate AIGenerator's sequential tool-calling behavior — it passes tools
to Claude, dispatches tool_use requests to the ToolManager, and may run up to
MAX_TOOL_ROUNDS (2) tool rounds in separate API requests before a final answer.
After the round limit or a tool failure it makes a tools-free "synthesis" call.

All assertions check EXTERNAL behavior: the number of API calls, whether tools
are attached per call, which tools were executed with what args, the message
structure sent back, and the returned string. The Anthropic client is patched,
so no network/API key is required.
"""

from unittest.mock import MagicMock, patch

import anthropic
import httpx

from ai_generator import AIGenerator


# --------------------------------------------------------------------------- #
# No-tool / single-round basics
# --------------------------------------------------------------------------- #


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
def test_single_round_dispatches_to_tool_manager(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """One tool round then a direct answer: 2 API calls, one tool executed."""
    client = MagicMock()
    mock_anthropic.return_value = client
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
def test_second_call_still_offers_tools(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """After the first tool round, the follow-up call MUST still offer tools so
    Claude can start a second round. (Replaces the old no-tools-on-2nd-call rule.)"""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "MCP"}),
        make_text_response("done"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator("fake-key", "fake-model")
    gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    second_kwargs = client.messages.create.call_args_list[1].kwargs
    assert second_kwargs["tools"] == [{"name": "search_course_content"}]
    assert second_kwargs["tool_choice"] == {"type": "auto"}


@patch("ai_generator.anthropic.Anthropic")
def test_single_round_message_structure(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """The follow-up call includes the assistant tool_use turn and a user turn
    carrying the tool_result with the matching tool_use_id."""
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


# --------------------------------------------------------------------------- #
# Sequential (two-round) behavior
# --------------------------------------------------------------------------- #


@patch("ai_generator.anthropic.Anthropic")
def test_two_sequential_tool_rounds(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """The headline flow: outline lookup -> search using its result -> answer.
    3 API calls; 2 tools executed in order; calls 1&2 offer tools, call 3 doesn't."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="a"),
        make_tool_use_response("search_course_content", {"query": "lesson 4 title"}, tool_id="b"),
        make_text_response("final answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["outline text", "search text"]

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "find a course like lesson 4 of X",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert out == "final answer"
    assert client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
    calls = tool_manager.execute_tool.call_args_list
    assert calls[0].args[0] == "get_course_outline" and calls[0].kwargs == {"course_name": "X"}
    assert calls[1].args[0] == "search_course_content" and calls[1].kwargs == {"query": "lesson 4 title"}
    # calls 1 & 2 offer tools; the final synthesis call (3) does not.
    assert "tools" in client.messages.create.call_args_list[0].kwargs
    assert "tools" in client.messages.create.call_args_list[1].kwargs
    assert "tools" not in client.messages.create.call_args_list[2].kwargs
    assert "tool_choice" not in client.messages.create.call_args_list[2].kwargs


@patch("ai_generator.anthropic.Anthropic")
def test_second_round_sees_first_round_results(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """The second round's request carries the first round's tool_result."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="a"),
        make_tool_use_response("search_course_content", {"query": "t"}, tool_id="b"),
        make_text_response("answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["outline text", "search text"]

    gen = AIGenerator("fake-key", "fake-model")
    gen.generate_response(
        "q",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    assert [m["role"] for m in second_messages] == ["user", "assistant", "user"]
    assert second_messages[2]["content"][0]["tool_use_id"] == "a"
    assert second_messages[2]["content"][0]["content"] == "outline text"


@patch("ai_generator.anthropic.Anthropic")
def test_stops_after_two_rounds_third_call_is_tools_free(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """Two tool rounds, then a tools-free synthesis call — the cap is enforced."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "1"}),
        make_tool_use_response("search_course_content", {"query": "2"}),
        make_text_response("synth"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    assert out == "synth"
    assert client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
    third_kwargs = client.messages.create.call_args_list[2].kwargs
    assert "tools" not in third_kwargs
    assert "tool_choice" not in third_kwargs


@patch("ai_generator.anthropic.Anthropic")
def test_no_third_tool_round_even_if_model_keeps_requesting(
    mock_anthropic, make_tool_use_response
):
    """If Claude requests a tool on every response, we still execute at most 2
    and never a 3rd; the tools-free synthesis call yields no text -> fallback."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "1"}),
        make_tool_use_response("search_course_content", {"query": "2"}),
        make_tool_use_response("search_course_content", {"query": "3"}),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    assert tool_manager.execute_tool.call_count == 2
    assert client.messages.create.call_count == 3
    # 3rd response is tool_use-only (no text) on a tools-free call -> fallback.
    assert out == AIGenerator.API_ERROR_MESSAGE


@patch("ai_generator.anthropic.Anthropic")
def test_parallel_tool_calls_count_as_one_round(
    mock_anthropic, make_multi_tool_use_response, make_text_response
):
    """Multiple tool_use blocks in one response are all executed and each gets a
    matching tool_result, counting as a single round (2 API calls total)."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_multi_tool_use_response([
            ("search_course_content", {"query": "a"}, "id_a"),
            ("get_course_outline", {"course_name": "b"}, "id_b"),
        ]),
        make_text_response("combined answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["res a", "res b"]

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "q",
        tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
        tool_manager=tool_manager,
    )

    assert out == "combined answer"
    assert tool_manager.execute_tool.call_count == 2
    assert client.messages.create.call_count == 2
    tool_results = client.messages.create.call_args_list[1].kwargs["messages"][2]["content"]
    assert [tr["tool_use_id"] for tr in tool_results] == ["id_a", "id_b"]


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@patch("ai_generator.anthropic.Anthropic")
def test_tool_error_terminates_and_synthesizes(
    mock_anthropic, make_tool_use_response, make_text_response
):
    """A raising tool: the error is fed back as a tool_result, no further tool
    round is granted, and a tools-free synthesis call produces the answer."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "x"}, tool_id="err"),
        make_text_response("sorry, that lookup failed"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("boom")

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    assert out == "sorry, that lookup failed"
    assert tool_manager.execute_tool.call_count == 1
    assert client.messages.create.call_count == 2
    # synthesis call is tools-free
    assert "tools" not in client.messages.create.call_args_list[1].kwargs
    # error was fed back with the matching id
    tool_result = client.messages.create.call_args_list[1].kwargs["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "err"
    assert tool_result.get("is_error") is True
    assert "failed" in tool_result["content"].lower()


@patch("ai_generator.anthropic.Anthropic")
def test_empty_final_text_returns_friendly_message(
    mock_anthropic, make_tool_use_response, make_text_response
):
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", {"query": "x"}),
        make_text_response(""),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response(
        "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    assert out == AIGenerator.API_ERROR_MESSAGE


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
    """When the Anthropic call fails (e.g. billing 400, outage), the error is
    caught and a friendly message is returned instead of propagating as a 500."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.side_effect = _api_error("credit balance is too low")

    gen = AIGenerator("fake-key", "fake-model")
    out = gen.generate_response("q")

    assert out == AIGenerator.API_ERROR_MESSAGE


@patch("ai_generator.anthropic.Anthropic")
def test_api_error_during_tool_round_returns_friendly_message(
    mock_anthropic, make_tool_use_response
):
    """A failure on a later (post-tool) round is also caught by the outer handler."""
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
    """If the model returns tool_use but no tool_manager is wired up, calling
    .text on the tool_use block would raise AttributeError. Instead we return a
    friendly fallback."""
    client = MagicMock()
    mock_anthropic.return_value = client
    client.messages.create.return_value = make_tool_use_response(
        "search_course_content", {"query": "MCP"}
    )

    gen = AIGenerator("fake-key", "fake-model")
    # tools provided, but tool_manager omitted
    out = gen.generate_response("q", tools=[{"name": "search_course_content"}])

    assert out == AIGenerator.API_ERROR_MESSAGE
