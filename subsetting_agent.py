"""LLM agent that builds subsets of genebank accessions.

The agent follows the tool-calling loop of the AClimate ``AClimateAgent``:
conversation memory, a cache of executed calls, rescue of tool calls that the
model emits as plain-text JSON, and stall detection. Tools are dispatched
locally through :class:`tools.registry.ToolRegistry` instead of an MCP session.

The agent is stateless between chat turns: the caller provides the previous
memory and the serialized :class:`~tools.accession_context.AccessionContext`
and receives the updated ones back.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litellm import acompletion

from document_processing.document_store import DocumentStore
from genesys_sdk.client import GenesysClient
from prompts.system_prompt import build_system_prompt
from subsetting_sdk.client import SubsettingClient
from tools.accession_context import AccessionContext
from tools.registry import ToolRegistry, build_registry
from tools.services import SOURCE_FILE, SOURCE_GENESYS, ToolServices

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ollama_chat/llama3.1:8b"
DEFAULT_API_BASE = "http://localhost:11434"


class _RescuedFunction:
    """Mimics the ``.function`` attribute of a native tool call."""

    def __init__(self, name: str, arguments: str) -> None:
        """Store the function name and its JSON-encoded arguments.

        Args:
            name: Tool name.
            arguments: JSON string with the arguments.
        """
        self.name = name
        self.arguments = arguments


class _RescuedToolCall:
    """Tool call reconstructed from JSON the model emitted as plain text."""

    _counter = 0

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        """Build a tool call with a synthetic id.

        Args:
            name: Tool name.
            arguments: Arguments as a dictionary.
        """
        _RescuedToolCall._counter += 1
        self.id = f"rescued_call_{_RescuedToolCall._counter}"
        self.function = _RescuedFunction(
            name=name, arguments=json.dumps(arguments, ensure_ascii=False)
        )

    def model_dump(self) -> dict[str, Any]:
        """Serialize like a native litellm tool call."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


@dataclass
class AgentTurn:
    """Result of one chat turn.

    Attributes:
        answer: Final assistant text.
        memory: Updated conversation memory (OpenAI-style messages).
        context_json: Serialized accession selection after the turn.
        document_errors: Messages about PDFs that could not be processed.
    """

    answer: str
    memory: list[dict[str, Any]]
    context_json: str
    document_errors: list[str] = field(default_factory=list)


class SubsettingAgent:
    """LLM agent that builds accession subsets with local tools."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_base: str | None = None,
        max_iterations: int = 15,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        registry: ToolRegistry | None = None,
        services: ToolServices | None = None,
        document_cache_dir: str | Path | None = None,
    ) -> None:
        """Configure the agent.

        Args:
            model: litellm model name (``SUBSETTING_AGENT_MODEL``).
            api_base: LLM endpoint (``SUBSETTING_AGENT_API_BASE``).
            max_iterations: Maximum LLM calls per turn.
            max_tokens: Maximum tokens per LLM answer.
            temperature: Sampling temperature.
            num_ctx: Context window requested from Ollama.
            registry: Tool registry; when ``None`` the registry matching the
                accession source of each turn is built automatically.
            services: Pre-built services (tests); when ``None`` they are created
                per turn from the environment and the uploaded files.
            document_cache_dir: Cache directory for PDF conversions.
        """
        self.model = model or os.getenv("SUBSETTING_AGENT_MODEL", DEFAULT_MODEL)
        self.api_base = api_base or os.getenv("SUBSETTING_AGENT_API_BASE", DEFAULT_API_BASE)
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._registry = registry
        self._services = services
        self.document_cache_dir = document_cache_dir

        self.memory: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset_memory(self) -> None:
        """Forget the conversation memory."""
        self.memory.clear()

    async def chat(
        self,
        user_message: str,
        *,
        document_paths: list[str | Path] | None = None,
        accession_file_paths: list[str | Path] | None = None,
        context_json: str | None = None,
    ) -> AgentTurn:
        """Process one user message through the LLM and the tools.

        The accession source is decided here, deterministically: when the user
        uploaded an accession spreadsheet the turn runs in file mode (no Genesys
        client, file tools registered); otherwise it runs in Genesys mode.

        Args:
            user_message: Text written by the user.
            document_paths: PDFs attached in this conversation (all turns).
            accession_file_paths: Excel/CSV accession lists attached in this
                conversation (all turns).
            context_json: Serialized selection from the previous turn.

        Returns:
            The answer, the updated memory and the updated selection state.
        """
        # An empty message cannot be processed; keep the state untouched.
        if not user_message.strip():
            return AgentTurn(
                answer="Please write a message describing the accessions you need.",
                memory=self.memory,
                context_json=context_json or AccessionContext().to_json(),
            )

        accession_files = [Path(path) for path in accession_file_paths or []]
        services = self._services or self._build_services(accession_files)
        services.context = AccessionContext.from_json(context_json)
        document_errors = self._load_documents(services, document_paths or [])

        # The registry and prompt follow the source; an injected registry wins (tests).
        registry = self._registry or build_registry(services.source)
        system_prompt = build_system_prompt(registry.describe(), services.source)

        self.memory.append(
            {"role": "user", "content": self._annotate_message(user_message, services)}
        )

        try:
            answer = await self._run_agent_loop(services, registry, system_prompt)

        finally:
            # Services built for this turn own their HTTP clients; injected ones don't.
            if self._services is None:
                await services.aclose()

        return AgentTurn(
            answer=answer,
            memory=self.memory,
            context_json=services.context.to_json(),
            document_errors=document_errors,
        )

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def _build_services(self, accession_files: list[Path]) -> ToolServices:
        """Create the clients and stores for one turn from the environment.

        Args:
            accession_files: Spreadsheets uploaded by the user; a non-empty list
                switches the turn to file mode and skips the Genesys client.
        """
        # In file mode no Genesys client is created: the API is not contacted at all.
        genesys = None if accession_files else GenesysClient()

        return ToolServices(
            genesys=genesys,
            subsetting=SubsettingClient(),
            documents=DocumentStore(self.document_cache_dir),
            accession_files=accession_files,
        )

    @staticmethod
    def _load_documents(services: ToolServices, paths: list[str | Path]) -> list[str]:
        """Add the uploaded PDFs to the document store.

        Args:
            services: Services of the turn.
            paths: PDF paths attached by the user.

        Returns:
            Error messages for files that could not be converted.
        """
        # Nothing attached means nothing to convert.
        if not paths:
            return []

        _, errors = services.documents.add_pdfs(paths)

        return errors

    @staticmethod
    def _annotate_message(user_message: str, services: ToolServices) -> str:
        """Append a short state note to the user message so the model knows the context.

        Args:
            user_message: Original text.
            services: Services holding the selection and documents.
        """
        notes = []
        context = services.context

        # Name the source so the model never tries the other one.
        if services.source == SOURCE_FILE:
            names = ", ".join(path.name for path in services.accession_files[:3])
            notes.append(f"[accession source: spreadsheet ({names}); Genesys is disabled]")

        elif services.source == SOURCE_GENESYS:
            notes.append("[accession source: Genesys API]")

        # Tell the model about an existing selection so it continues instead of restarting.
        if not context.is_empty:
            notes.append(
                f"[state: {context.count} accessions selected, stage={context.stage.value}]"
            )

        # Tell the model which documents exist so it uses the document tools.
        if not services.documents.is_empty:
            titles = ", ".join(d.title for d in services.documents.list_documents()[:5])
            notes.append(f"[documents uploaded: {titles}]")

        if not notes:
            return user_message

        return f"{user_message}\n\n" + "\n".join(notes)

    # ------------------------------------------------------------------ #
    # Agent loop
    # ------------------------------------------------------------------ #

    async def _run_agent_loop(
        self, services: ToolServices, registry: ToolRegistry, system_prompt: dict[str, str]
    ) -> str:
        """Alternate LLM calls and tool executions until a final answer is produced.

        Args:
            services: Services of the turn.
            registry: Tools available in this turn.
            system_prompt: System message.
        """
        # (tool + normalized arguments) -> result already obtained in this turn
        executed_calls: dict[str, dict[str, Any]] = {}

        # consecutive iterations without any new tool call
        stalled_iterations = 0
        tools = registry.openai_tools()

        print(f"Tools: {tools}")
        print(f"Memory: {self.memory}")
        # Each iteration is one LLM call followed by the tool calls it requested.
        for iteration in range(1, self.max_iterations + 1):
            logger.debug("Agent iteration %s", iteration)
            print("Agent iteration %s", iteration)
            

            response = await acompletion(
                model=self.model,
                api_base=self.api_base,
                messages=[system_prompt, *self.memory],
                tools=tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
            )

            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])

            # Small models sometimes emit the tool call as text instead of using
            # the structured channel; rescue it before treating it as an answer.
            if not tool_calls and message.content:
                rescued = self._extract_text_tool_calls(message.content)

                if rescued:
                    logger.warning("Rescued %s tool call(s) emitted as plain text", len(rescued))
                    tool_calls = rescued
                    message.content = None

            # No tool calls means the model produced its final answer.
            if not tool_calls:
                final_content = message.content or (
                    "It was not possible to generate a response for the query."
                )
                self.memory.append({"role": "assistant", "content": final_content})

                return final_content

            self.memory.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in tool_calls],
                }
            )

            made_progress = False

            # Execute each requested tool, serving cached results for repeats.
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = self._parse_tool_arguments(tool_call.function.arguments)
                call_key = self._build_call_key(tool_name, tool_arguments)
                print(f"Tool calling: {tool_name} with {tool_arguments}")
                if call_key in executed_calls:
                    logger.warning("Repeated call to %s with %s", tool_name, tool_arguments)
                    result = {
                        "repeated_call": True,
                        "note": (
                            f"You already called '{tool_name}' with these arguments in this turn. "
                            "The previous result is below; use it to move to the next step or "
                            "change strategy. Do not repeat this call."
                        ),
                        "previous_result": executed_calls[call_key],
                    }

                else:
                    logger.info("Executing tool %s with %s", tool_name, tool_arguments)
                    result = await registry.execute(services, tool_name, tool_arguments)
                    executed_calls[call_key] = result
                    made_progress = True

                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            # Stall detection: two iterations of only repeated calls end the turn.
            if made_progress:
                stalled_iterations = 0

            else:
                stalled_iterations += 1
                logger.warning(
                    "Iteration %s produced no new tool calls (stalled=%s)",
                    iteration,
                    stalled_iterations,
                )

                if stalled_iterations >= 2:
                    stalled = (
                        "I could not make progress: the same queries were repeated without new "
                        "information. Please rephrase the request or specify the crop, the "
                        "criteria and the climate condition you need."
                    )
                    self.memory.append({"role": "assistant", "content": stalled})

                    return stalled

        fallback = (
            "It was not possible to complete the request within the maximum number of "
            "allowed steps. Please split the request into smaller steps."
        )
        self.memory.append({"role": "assistant", "content": fallback})

        return fallback

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text_tool_calls(content: str) -> list[_RescuedToolCall]:
        """Detect tool calls that the model emitted as plain-text JSON.

        Recognized shapes (one object, or several separated by newlines or ';'):
        ``{"type": "function", "name": ..., "parameters": {...}}``,
        ``{"name": ..., "parameters": {...}}`` and ``{"name": ..., "arguments": {...}}``.

        Args:
            content: Assistant text.

        Returns:
            Rescued calls, or an empty list when the text is not only tool-call JSON.
        """
        text = content.strip()

        # Fast reject: prose answers never start with a JSON object.
        if not text.startswith("{"):
            return []

        try:
            json.loads(text)
            candidates = [text]

        except json.JSONDecodeError:
            # Maybe several objects separated by newlines or semicolons.
            parts = [p.strip().rstrip(";") for p in text.replace("};", "}\n").splitlines()]
            candidates = [p for p in parts if p.startswith("{")]

        rescued: list[_RescuedToolCall] = []

        # Every candidate must be a valid call object; otherwise it is prose.
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
        """Identity of a call: name plus normalized arguments.

        Args:
            tool_name: Tool name.
            tool_arguments: Parsed arguments.
        """
        return f"{tool_name}:" + json.dumps(tool_arguments, sort_keys=True, default=str)

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Decode the arguments of a tool call into a dictionary.

        Args:
            arguments: Dictionary or JSON string produced by the model.

        Returns:
            The arguments; invalid JSON yields an empty dictionary so the tool can
            report the missing arguments instead of crashing the loop.
        """
        if isinstance(arguments, dict):
            return arguments

        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)

        except (TypeError, json.JSONDecodeError):
            logger.warning("Invalid JSON arguments from the model: %s", arguments)
            return {}

        # Non-object JSON (a list, a string) is not a valid argument set.
        if not isinstance(parsed, dict):
            return {}

        return parsed
