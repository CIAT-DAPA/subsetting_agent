"""Unit tests for :class:`SubsettingAgent` with a scripted fake LLM."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

import subsetting_agent as agent_module
from document_processing import DocumentStore
from genesys_sdk import GenesysClient
from prompts import SYSTEM_PROMPT_TEMPLATE, build_system_prompt
from subsetting_agent import SubsettingAgent
from subsetting_sdk import SubsettingClient
from tests.test_genesys_sdk import page_payload
from tools import AccessionContext, ToolServices

GENESYS = "https://genesys.example"
SUBSETTING = "https://subsetting.example"


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> SimpleNamespace:
    """Build an object shaped like a litellm tool call.

    Args:
        name: Tool name.
        arguments: Arguments dictionary (encoded as JSON like the real API).
        call_id: Identifier of the call.
    """
    payload = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    call.model_dump = lambda: payload  # type: ignore[attr-defined]

    return call


def llm_response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """Build an object shaped like a litellm completion response.

    Args:
        content: Assistant text.
        tool_calls: Tool calls requested by the model.
    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])

    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    """Scripted replacement for ``litellm.acompletion``.

    Attributes:
        responses: Responses returned in order; the last one repeats.
        calls: Keyword arguments of every call, for assertions.
    """

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        """Store the scripted responses.

        Args:
            responses: Responses to return in order.
        """
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> SimpleNamespace:
        """Return the next scripted response and record the call."""
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.responses) - 1)

        return self.responses[index]


@pytest.fixture
def services(tmp_path: Path) -> ToolServices:
    """Services wired to the mocked hosts."""
    return ToolServices(
        genesys=GenesysClient(GENESYS, token="t", max_retries=0, backoff_seconds=0.0),
        subsetting=SubsettingClient(
            SUBSETTING, api_prefix="/api/v1", max_retries=0, backoff_seconds=0.0
        ),
        documents=DocumentStore(tmp_path / "cache"),
    )


def make_agent(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLLM, services: ToolServices
) -> SubsettingAgent:
    """Create an agent whose LLM is the fake and whose services are injected.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        fake: Scripted LLM.
        services: Injected services.
    """
    monkeypatch.setattr(agent_module, "acompletion", fake)

    return SubsettingAgent(
        model="fake", api_base="http://fake", services=services, max_iterations=6
    )


class TestSystemPrompt:
    """The prompt encodes the business order and the tool list."""

    def test_prompt_contains_stages_in_order(self) -> None:
        """Passport comes before traits, documents and climate."""
        text = SYSTEM_PROMPT_TEMPLATE

        assert text.index("Stage 1 - PASSPORT") < text.index("Stage 2 - TRAITS")
        assert text.index("Stage 2 - TRAITS") < text.index("Stage 3 - DOCUMENTS")
        assert text.index("Stage 3 - DOCUMENTS") < text.index("Stage 4 - CLIMATE")

    def test_build_system_prompt_injects_tools(self) -> None:
        """Tool descriptions are rendered inside the system message."""
        message = build_system_prompt("- select_accessions: load accessions")

        assert message["role"] == "system"
        assert "- select_accessions: load accessions" in message["content"]


class TestAgentLoop:
    """The loop executes tools, updates state and returns the final answer."""

    async def test_empty_message(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices
    ) -> None:
        """Blank input is answered without calling the LLM."""
        fake = FakeLLM([llm_response("never")])
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("   ")

        assert "Please write a message" in turn.answer
        assert fake.calls == []

    async def test_tool_call_then_answer(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """A tool call is executed, its result stored in memory, then the answer returned."""
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/list?p=0&l=50",
            json=page_payload(["u-1", "u-2"], number=0, total=2, last=True),
        )
        fake = FakeLLM(
            [
                llm_response(tool_calls=[tool_call("select_accessions", {"crop_codes": ["bean"]})]),
                llm_response("I selected 2 bean accessions."),
            ]
        )
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("Find bean accessions")

        assert turn.answer == "I selected 2 bean accessions."
        assert len(fake.calls) == 2
        # System prompt first, then the user message with tools passed along.
        assert fake.calls[0]["messages"][0]["role"] == "system"
        assert any(t["function"]["name"] == "select_accessions" for t in fake.calls[0]["tools"])
        # Memory holds user, assistant(tool_calls), tool, assistant(answer).
        assert [m["role"] for m in turn.memory] == ["user", "assistant", "tool", "assistant"]
        tool_result = json.loads(turn.memory[2]["content"])
        assert tool_result["accession_count"] == 2
        # The selection state is returned serialized for the next turn.
        assert AccessionContext.from_json(turn.context_json).count == 2

    async def test_state_is_restored_between_turns(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices
    ) -> None:
        """A previous selection is rebuilt and announced to the model."""
        previous = AccessionContext()
        previous.set_passport_selection([], passport_filter={}, total_matching=0, description="x")
        previous_json = previous.to_json()
        fake = FakeLLM([llm_response("ok")])
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("hello", context_json=previous_json)

        assert turn.answer == "ok"
        assert AccessionContext.from_json(turn.context_json).stage.value == "passport"

    async def test_rescues_text_tool_call(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices
    ) -> None:
        """A JSON tool call emitted as text is executed like a native one."""
        fake = FakeLLM(
            [
                llm_response('{"name": "describe_selection", "parameters": {"sample": 2}}'),
                llm_response("Nothing selected yet."),
            ]
        )
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("what do we have?")

        assert turn.answer == "Nothing selected yet."
        assert turn.memory[1]["tool_calls"][0]["function"]["name"] == "describe_selection"
        assert turn.memory[1]["content"] is None
        assert json.loads(turn.memory[2]["content"])["stage"] == "empty"

    async def test_repeated_call_is_served_from_cache_and_stalls(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices
    ) -> None:
        """Repeating the same call twice more ends the turn with a stall message."""
        call = tool_call("describe_selection", {"sample": 1})
        fake = FakeLLM([llm_response(tool_calls=[call])])
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("loop forever")

        assert "could not make progress" in turn.answer
        # 1 real execution + 2 cached repeats = 3 LLM calls.
        assert len(fake.calls) == 3
        cached = json.loads(turn.memory[4]["content"])
        assert cached["repeated_call"] is True

    async def test_unknown_tool_is_reported_to_model(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices
    ) -> None:
        """An invented tool name yields an error result instead of an exception."""
        fake = FakeLLM(
            [
                llm_response(tool_calls=[tool_call("teleport", {})]),
                llm_response("Sorry, I cannot do that."),
            ]
        )
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("teleport the seeds")

        assert "Unknown tool" in json.loads(turn.memory[2]["content"])["error"]
        assert turn.answer == "Sorry, I cannot do that."

    async def test_max_iterations_fallback(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices
    ) -> None:
        """Endless distinct tool calls hit the iteration cap."""
        responses = [
            llm_response(tool_calls=[tool_call("describe_selection", {"sample": i}, f"c{i}")])
            for i in range(1, 10)
        ]
        fake = FakeLLM(responses)
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("keep describing")

        assert "maximum number" in turn.answer
        assert len(fake.calls) == 6

    async def test_documents_are_loaded_and_announced(
        self, monkeypatch: pytest.MonkeyPatch, services: ToolServices, tmp_path: Path
    ) -> None:
        """Uploaded PDFs are converted; bad files are reported without failing."""
        from tests.test_document_processing import write_pdf

        pdf = write_pdf(
            tmp_path / "paper.pdf", [[("Bean paper", 20), ("Text about beans. " * 10, 10)]]
        )
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"garbage")
        fake = FakeLLM([llm_response("done")])
        agent = make_agent(monkeypatch, fake, services)

        turn = await agent.chat("summarize the paper", document_paths=[pdf, bad])

        assert len(turn.document_errors) == 1
        assert "documents uploaded" in fake.calls[0]["messages"][-1]["content"]
        assert not services.documents.is_empty

    def test_parse_tool_arguments_tolerates_bad_json(self) -> None:
        """Invalid JSON arguments become an empty dictionary instead of raising."""
        assert SubsettingAgent._parse_tool_arguments("{not json") == {}
        assert SubsettingAgent._parse_tool_arguments("[1, 2]") == {}
        assert SubsettingAgent._parse_tool_arguments({"a": 1}) == {"a": 1}
        assert SubsettingAgent._parse_tool_arguments("") == {}

    def test_extract_text_tool_calls_rejects_prose(self) -> None:
        """Ordinary text and partial JSON are never treated as tool calls."""
        assert SubsettingAgent._extract_text_tool_calls("Here are your results.") == []
        assert SubsettingAgent._extract_text_tool_calls('{"foo": 1}') == []
        rescued = SubsettingAgent._extract_text_tool_calls(
            '{"name": "a", "arguments": {}}\n{"name": "b", "parameters": {"x": 1}}'
        )
        assert [r.function.name for r in rescued] == ["a", "b"]
