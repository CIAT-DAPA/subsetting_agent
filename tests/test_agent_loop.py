"""End-to-end test of the agent loop with a scripted (fake) LLM.

The fake ``acompletion`` returns a pre-defined sequence of responses so the
whole chain (tool call -> skill -> tool result -> final answer -> formatter)
runs without a real model server.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.subsetting_agent as agent_module
from agent.skill_registry import SkillRegistry
from agent.subsetting_agent import SubsettingAgent
from core.config import Settings
from core.session import SessionManager
from core.state import SessionStore

FIXTURES = Path(__file__).parent / "fixtures"


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    """Build an object shaped like a LiteLLM tool call."""
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    call.model_dump = lambda: {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return call


def _response(content: str | None, tool_calls: list | None = None) -> SimpleNamespace:
    """Build an object shaped like a LiteLLM completion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def scripted_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Agent wired to a fake LLM that follows a script of responses."""
    script: list[SimpleNamespace] = []
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        """Record the request and pop the next scripted response."""
        calls.append(kwargs)
        return script.pop(0)

    monkeypatch.setattr(agent_module, "acompletion", fake_acompletion)

    settings = Settings(tmp_dir=tmp_path / "tmp", llm_model="fake/model")
    manager = SessionManager(settings.tmp_dir)
    agent = SubsettingAgent(
        settings=settings,
        registry=SkillRegistry().discover(),
        session_store=SessionStore(),
        session_manager=manager,
    )
    return agent, manager, script, calls


async def test_load_then_export_through_the_loop(scripted_agent) -> None:
    """Turn 1 loads a file; turn 2 exports the Candidate list and attaches it."""
    agent, manager, script, calls = scripted_agent
    uploaded = manager.store_input_file("s1", FIXTURES / "accessions_sample.xlsx")

    # Turn 1: the model calls list_accessions, then answers.
    script.extend(
        [
            _response(
                None,
                [_tool_call("c1", "list_accessions", {"source": "local", "file_path": str(uploaded)})],
            ),
            _response("Loaded 5 accessions in local mode."),
        ]
    )
    first = await agent.chat("s1", "Load my file", history=[], uploaded_files=[uploaded])

    assert "Loaded 5 accessions" in first.text
    assert "Activities performed so far" in first.text
    assert "Candidate list preview" in first.text
    assert "G50001" in first.text
    assert first.files == []

    # The tool result was fed back to the model as a "tool" message.
    tool_messages = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["status"] == "ok"

    # Turn 2: the model exports the Candidate list.
    script.extend(
        [
            _response(None, [_tool_call("c2", "export_list", {"which": "candidate"})]),
            _response("Here is your final list."),
        ]
    )
    second = await agent.chat("s1", "Give me the final list", history=[])

    assert len(second.files) == 1
    assert second.files[0].suffix == ".csv"
    assert second.files[0].parent == manager.get_paths("s1").outputs


async def test_unknown_tool_and_stall_are_reported(scripted_agent) -> None:
    """Hallucinated tools get an error result; repeated calls end the turn."""
    agent, _, script, _ = scripted_agent
    repeated = _tool_call("c1", "does_not_exist", {})

    # Three identical calls: the first executes, the next two hit the cache
    # twice in a row and trigger the stall guard.
    script.extend([_response(None, [repeated]), _response(None, [repeated]), _response(None, [repeated])])

    result = await agent.chat("s2", "do something", history=[])

    assert "could not make progress" in result.text
