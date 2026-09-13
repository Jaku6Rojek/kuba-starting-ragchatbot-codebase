import logging
import anthropic
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses.

    Supports sequential tool calling: Claude may make up to ``MAX_TOOL_ROUNDS``
    tool-using rounds per query, each a separate API request, so it can reason
    about a round's results before deciding whether to call another tool. After
    the round limit (or a tool failure) a final tools-free "synthesis" call is
    made so Claude produces a text answer grounded in the accumulated results.
    """

    # User-facing message returned when the Anthropic API call fails (e.g.
    # out of credits, rate limited, or an outage). The raw error is logged
    # server-side rather than leaked to the browser.
    API_ERROR_MESSAGE = "The assistant is temporarily unavailable. Please try again later."

    # Maximum number of tool-using rounds (separate API requests that returned
    # tool_use and whose tools we executed) allowed per user query. The worst
    # case is MAX_TOOL_ROUNDS + 1 API calls (the extra one being the tools-free
    # synthesis call).
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
- **search_course_content**: Search within course materials for specific content or detailed educational information.
- **get_course_outline**: Retrieve a course's outline — its title, course link, and complete lesson list (each lesson's number and title).

Tool Usage:
- Use **search_course_content** for questions about specific course content or detailed educational materials.
- Use **get_course_outline** for questions about a course's structure, syllabus, or which lessons it contains.
- **You may use tools across up to 2 sequential rounds per query.** After seeing a tool's results you may either call another tool to follow up, or give your final answer.
- Make a follow-up tool call when answering requires information you can only obtain from a previous tool result — for example, first get a course's outline to find a specific lesson's title, then search other courses for that title's topic.
- Once you have enough information to answer, respond directly and do not call more tools.
- Do not repeat a tool call with the same arguments.
- Synthesize tool results into accurate, fact-based responses.
- If a tool yields no results or fails, state this clearly without offering alternatives.

Course Outline Responses:
- When answering an outline query, return the **course title**, the **course link**, and for **every lesson** its **number and title**.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional (sequential) tool usage.

        Claude may make up to ``MAX_TOOL_ROUNDS`` tool-using rounds; each round
        is a separate API request and the conversation is preserved between
        rounds so Claude can reason about prior tool results.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

        # A single outer handler catches an API failure on ANY round (first,
        # second, or the final synthesis call) and returns the friendly message.
        try:
            return self._run_tool_loop(messages, system_content, tools, tool_manager)
        except anthropic.APIError:
            logger.exception("Anthropic API call failed")
            return self.API_ERROR_MESSAGE

    def _run_tool_loop(self, messages: List[Dict[str, Any]], system_content: str,
                       tools: Optional[List], tool_manager) -> str:
        """Drive up to MAX_TOOL_ROUNDS tool rounds, then return a text answer.

        Termination: (a) after MAX_TOOL_ROUNDS executed rounds, tools stop being
        offered so the next call is a tools-free synthesis; (b) a response with
        no tool_use returns its text immediately; (c) a tool failure stops
        further rounds and triggers the synthesis call.
        """
        offering_tools = bool(tools)
        rounds_completed = 0

        while True:
            offer_tools = offering_tools and rounds_completed < self.MAX_TOOL_ROUNDS
            response = self._call_api(
                messages, system_content, tools if offer_tools else None
            )

            # (b) Claude answered directly, or this was a tools-free synthesis
            # call (no tools offered, so it cannot request one).
            if not offer_tools or response.stop_reason != "tool_use":
                return self._extract_text(response) or self.API_ERROR_MESSAGE

            # Tool_use requested but no manager wired up: can't run tools.
            if tool_manager is None:
                logger.warning(
                    "Model requested tool use but no tool_manager was provided"
                )
                return self._extract_text(response) or self.API_ERROR_MESSAGE

            # Record the assistant's tool_use turn, run the tools, feed results.
            messages.append({"role": "assistant", "content": response.content})
            tool_results, had_error = self._execute_tools(response.content, tool_manager)

            # Defensive: tool_use stop_reason with no tool_use blocks would make
            # an empty (invalid) user turn. Fall back to whatever text exists.
            if not tool_results:
                return self._extract_text(response) or self.API_ERROR_MESSAGE

            messages.append({"role": "user", "content": tool_results})
            rounds_completed += 1

            # (c) A tool raised — stop offering tools so the next pass is a
            # tools-free synthesis call that lets Claude respond gracefully.
            if had_error:
                offering_tools = False
            # (a) is automatic: once rounds_completed == MAX_TOOL_ROUNDS,
            # offer_tools becomes False on the next pass -> synthesis -> return.

    def _call_api(self, messages: List[Dict[str, Any]], system_content: str,
                  tools: Optional[List]):
        """Make a single Anthropic API call.

        Tools (and ``tool_choice``) are attached only when ``tools`` is truthy,
        so the same helper serves tool-bearing rounds and the tools-free
        synthesis call (sending ``tool_choice`` without ``tools`` is rejected).
        API errors are not caught here — the caller's outer handler owns them.
        """
        params = {
            **self.base_params,
            # Snapshot the growing conversation so each call reflects exactly
            # what was sent (the loop mutates `messages` in place afterwards).
            "messages": list(messages),
            "system": system_content,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**params)

    def _execute_tools(self, content_blocks, tool_manager) -> Tuple[List[Dict[str, Any]], bool]:
        """Execute every tool_use block in a response.

        Returns ``(tool_results, had_error)``. Always produces one tool_result
        per tool_use block (the API requires a matching result for every
        tool_use id, even on failure). A raising tool is caught, its error is
        fed back to Claude as the tool_result content, and ``had_error`` is set.
        """
        results: List[Dict[str, Any]] = []
        had_error = False
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            result: Dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id}
            try:
                result["content"] = tool_manager.execute_tool(block.name, **block.input)
            except Exception:  # noqa: BLE001 - surface tool failure to the model
                logger.exception(
                    "Tool '%s' raised during execution", getattr(block, "name", "?")
                )
                result["content"] = "Tool execution failed; unable to complete this step."
                result["is_error"] = True
                had_error = True
            results.append(result)
        return results, had_error

    @staticmethod
    def _extract_text(response) -> str:
        """Return the text of the first text block in a response, or ''.

        Robust to responses whose first content block is not text (e.g. a
        tool_use block), which would otherwise raise AttributeError.
        """
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""
