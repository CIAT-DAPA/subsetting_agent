"""SubsettingAgent: LLM tool-calling loop that orchestrates the skills.

The agent is stateless with respect to the chat history (Gradio provides it on
every call) but stateful with respect to the session data: the Original and
Candidate lists live in a :class:`~core.state.SessionStore` shared by every
turn of the same session.
"""

import json
from pathlib import Path
from typing import Any

from litellm import acompletion

from agent.prompts import build_system_prompt
from agent.skill_registry import SkillRegistry
from core.config import Settings
from core.formatter import ResponseFormatter
from core.logger import get_logger
from core.session import SessionManager
from core.state import SessionState, SessionStore

logger = get_logger(__name__)


class _RescuedFunction:
    """Mimics the ``.function`` attribute of a native tool call."""

    def __init__(self, name: str, arguments: str) -> None:
        """Store the tool name and its JSON-encoded arguments.

        Args:
            name: Tool name emitted by the model.
            arguments: Arguments serialised as a JSON string.
        """
        self.name = name
        self.arguments = arguments


class _RescuedToolCall:
    """Tool call reconstructed from JSON the model emitted as plain text.

    Some open models (e.g. llama3.1) occasionally write the tool call as text
    instead of using the structured channel. This class gives such calls the
    same interface as native ones so the loop can treat them uniformly.
    """

    _counter = 0

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        """Create a synthetic tool call with a unique id.

        Args:
            name: Tool name.
            arguments: Parsed arguments dictionary.
        """
        _RescuedToolCall._counter += 1
        self.id = f"rescued_call_{_RescuedToolCall._counter}"
        self.function = _RescuedFunction(
            name=name, arguments=json.dumps(arguments, ensure_ascii=False)
        )

    def model_dump(self) -> dict[str, Any]:
        """Serialise the call like a native LiteLLM tool call."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class SubsettingAgent:
    """LLM agent that builds accession subsets through registered skills."""

    def __init__(
        self,
        settings: Settings,
        registry: SkillRegistry,
        session_store: SessionStore,
        session_manager: SessionManager,
        formatter: ResponseFormatter | None = None,
    ) -> None:
        """Wire the agent with its collaborators.

        Args:
            settings: Application settings (model, limits...).
            registry: Skills available to the model.
            session_store: In-memory store of per-session states.
            session_manager: File-system manager of per-session folders.
            formatter: Response formatter; a default one is built when omitted.
        """
        self.settings = settings
        self.registry = registry
        self.session_store = session_store
        self.session_manager = session_manager
        self.formatter = formatter or ResponseFormatter(preview_rows=settings.agent_preview_rows)

    # ------------------------------------------------------------ public API
    async def chat(
        self,
        session_id: str,
        user_message: str,
        history: list[dict[str, str]],
        uploaded_files: list[Path] | None = None,
    ) -> str:
        """Process one user turn and return the formatted answer.

        Args:
            session_id: Identifier of the session (Gradio ``session_hash``).
            user_message: Text typed by the user.
            history: Previous ``{"role", "content"}`` messages of the chat.
            uploaded_files: Files attached to this message, already stored in
                the session ``inputs`` folder.

        Returns:
            Markdown answer including the activity summary and the preview.
        """
        state = self.session_store.get_or_create(session_id)
        paths = self.session_manager.get_paths(session_id)
        uploaded_files = uploaded_files or []

        # Remember every attached file so later turns can refer to it.
        for file_path in uploaded_files:
            state.register_uploaded_file(file_path)

        message_text = self._compose_user_message(user_message, uploaded_files)

        # An empty message with no files gives the model nothing to work with.
        if not message_text.strip():
            return "Please write a message or attach a file so I can help you."

        memory: list[dict[str, Any]] = [
            *history[-self.settings.agent_max_history_messages :],
            {"role": "user", "content": message_text},
        ]

        system_prompt = build_system_prompt(
            tools_description=self.registry.describe(),
            session_context=state.summary_context(),
        )

        agent_text = await self._run_agent_loop(
            state=state, paths=paths, memory=memory, system_prompt=system_prompt
        )

        return self.formatter.build_response(agent_text, state)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _compose_user_message(user_message: str, uploaded_files: list[Path]) -> str:
        """Append the attached file paths to the user text.

        Args:
            user_message: Text typed by the user.
            uploaded_files: Files attached to this turn.

        Returns:
            The message the model receives.
        """
        # Nothing to add when the turn has no attachments.
        if not uploaded_files:
            return user_message

        files_block = "\n".join(f"- {path}" for path in uploaded_files)
        return f"{user_message}\n\nAttached files:\n{files_block}"

    def _completion_kwargs(self) -> dict[str, Any]:
        """Build the provider-specific keyword arguments for LiteLLM.

        Returns:
            Keyword arguments merged into every ``acompletion`` call.
        """
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "max_tokens": self.settings.llm_max_tokens,
            "temperature": self.settings.llm_temperature,
        }

        # Local servers (Ollama) need a base URL; hosted providers need a key.
        if self.settings.llm_api_base:
            kwargs["api_base"] = self.settings.llm_api_base
        if self.settings.llm_api_key:
            kwargs["api_key"] = self.settings.llm_api_key

        # ``num_ctx`` is an Ollama-only option; other providers would reject it.
        if self.settings.llm_model.startswith("ollama"):
            kwargs["num_ctx"] = self.settings.llm_num_ctx

        return kwargs

    # ------------------------------------------------------------- main loop
    async def _run_agent_loop(
        self,
        state: SessionState,
        paths,
        memory: list[dict[str, Any]],
        system_prompt: dict[str, str],
    ) -> str:
        """Alternate model calls and skill executions until a final answer.

        Args:
            state: State of the current session.
            paths: Folders of the current session.
            memory: Conversation messages (history + current user message).
            system_prompt: System message built for this turn.

        Returns:
            The final natural-language answer of the model.
        """
        tools = self.registry.tools()
        executed_calls: dict[str, dict[str, Any]] = {}  # call key -> cached result
        stalled_iterations = 0  # consecutive iterations without a new tool call
        completion_kwargs = self._completion_kwargs()

        # Each iteration is one model call optionally followed by tool executions.
        for iteration in range(1, self.settings.agent_max_iterations + 1):
            logger.debug("Session %s iteration %s", state.session_id, iteration)

            response = await acompletion(
                messages=[system_prompt, *memory],
                tools=tools or None,
                **completion_kwargs,
            )

            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])

            # Rescue tool calls that the model wrote as plain-text JSON.
            if not tool_calls and message.content:
                rescued = self._extract_text_tool_calls(message.content)

                if rescued:
                    logger.warning("Rescued %s tool call(s) emitted as text", len(rescued))
                    tool_calls = rescued
                    message.content = None

            # No tool calls means the model produced its final answer.
            if not tool_calls:
                return message.content or "I was not able to generate an answer for this request."

            memory.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in tool_calls],
                }
            )

            made_progress = False

            # Execute every requested tool and feed the results back to the model.
            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_arguments = self._parse_tool_arguments(tool_call.function.arguments)
                except ValueError as exc:
                    # Malformed arguments: tell the model instead of crashing the turn.
                    memory.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps({"status": "error", "message": str(exc)}),
                        }
                    )
                    continue

                call_key = self._build_call_key(tool_name, tool_arguments)

                # Identical repeated calls are served from cache to avoid loops.
                if call_key in executed_calls:
                    logger.warning("Repeated call to %s; serving cached result", tool_name)
                    result = {
                        "repeated_call": True,
                        "note": (
                            f"You already called '{tool_name}' with these arguments in this "
                            "conversation. The previous result is included below; the tool was "
                            "not executed again. Use it to move forward or change strategy."
                        ),
                        "previous_result": executed_calls[call_key],
                    }
                else:
                    result = self._execute_skill(state, paths, tool_name, tool_arguments)
                    executed_calls[call_key] = result
                    made_progress = True

                memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            # Track consecutive iterations that only repeated cached calls.
            if made_progress:
                stalled_iterations = 0
            else:
                stalled_iterations += 1
                logger.warning(
                    "Iteration %s produced no new tool calls (stalled=%s)",
                    iteration,
                    stalled_iterations,
                )

                # Two stalled iterations in a row: stop and ask the user to rephrase.
                if stalled_iterations >= 2:
                    return (
                        "I could not make progress: the same operations were repeated without "
                        "obtaining new information. Please rephrase your request or give more "
                        "details (file, accession names, column, criteria)."
                    )

        return (
            "It was not possible to complete the request within the maximum number of "
            "allowed steps. Please try a simpler or more specific request."
        )

    # ------------------------------------------------------- skill execution
    def _execute_skill(
        self,
        state: SessionState,
        paths,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a registered skill and shield the loop from its exceptions.

        Args:
            state: State of the current session.
            paths: Folders of the current session.
            tool_name: Name of the skill requested by the model.
            tool_arguments: Parsed arguments for the skill.

        Returns:
            The skill result, or an error dictionary the model can explain.
        """
        skill = self.registry.get(tool_name)

        # The model may hallucinate a tool name; answer with the real options.
        if skill is None:
            return {
                "status": "error",
                "message": f"Unknown tool '{tool_name}'. Available tools: {self.registry.names()}",
            }

        logger.info("Executing skill %s with %s", tool_name, tool_arguments)

        try:
            return skill.run(state, paths, **tool_arguments)
        except TypeError as exc:
            # Usually a missing or unexpected argument coming from the model.
            logger.exception("Invalid arguments for skill %s", tool_name)
            return {"status": "error", "message": f"Invalid arguments for '{tool_name}': {exc}"}
        except Exception as exc:  # noqa: BLE001 - any skill failure must reach the model
            logger.exception("Skill %s failed", tool_name)
            return {"status": "error", "message": f"Error executing '{tool_name}': {exc}"}

    # ---------------------------------------------------------- static utils
    @staticmethod
    def _extract_text_tool_calls(content: str) -> list[_RescuedToolCall]:
        """Detect tool calls that the model emitted as plain-text JSON.

        Recognised shapes (one object, or several separated by newlines/';'):
        ``{"type": "function", "name": ..., "parameters": {...}}``,
        ``{"name": ..., "parameters": {...}}`` and ``{"name": ..., "arguments": {...}}``.

        Args:
            content: Text content returned by the model.

        Returns:
            The rescued calls, or ``[]`` if the content is not exclusively tool JSON.
        """
        text = content.strip()

        # Prose answers never start with "{"; reject them fast.
        if not text.startswith("{"):
            return []

        try:
            json.loads(text)
            candidates = [text]
        except json.JSONDecodeError:
            # Maybe several objects separated by newlines or semicolons.
            parts = [part.strip().rstrip(";") for part in text.replace("};", "}\n").splitlines()]
            candidates = [part for part in parts if part.startswith("{")]

        rescued: list[_RescuedToolCall] = []

        # Every candidate must be a valid call; otherwise treat the text as prose.
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                return []

            if not isinstance(obj, dict):
                return []

            name = obj.get("name")
            arguments = obj.get("parameters", obj.get("arguments"))

            if not isinstance(name, str) or not isinstance(arguments, dict):
                return []

            rescued.append(_RescuedToolCall(name=name, arguments=arguments))

        return rescued

    @staticmethod
    def _build_call_key(tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Identity of a call: tool name plus normalised arguments.

        Args:
            tool_name: Name of the tool.
            tool_arguments: Parsed arguments.

        Returns:
            A deterministic string key.
        """
        return f"{tool_name}:" + json.dumps(tool_arguments, sort_keys=True, default=str)

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Decode the arguments of a tool call into a dictionary.

        Args:
            arguments: Either a dict, a JSON string or ``None``.

        Returns:
            The arguments as a dictionary (empty when none were given).

        Raises:
            ValueError: If the JSON is invalid or does not decode to an object.
        """
        # Native calls may already deliver a dictionary.
        if isinstance(arguments, dict):
            return arguments

        # No arguments at all is valid for parameter-less tools.
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON arguments received from the model: {arguments}") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")

        return parsed
