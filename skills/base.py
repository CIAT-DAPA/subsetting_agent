"""Base contract every skill of the SubsettingAgent must implement.

A *skill* is a self-contained capability exposed to the LLM as a function tool.
Each skill lives in its own sub package of ``skills/`` with two files:

* ``SKILL.md`` - human readable documentation and usage guidance for the model.
* ``skill.py`` - a subclass of :class:`Skill`.

Skills receive the ``SessionState`` and the ``SessionPaths`` of the current
session, so they can read the Candidate list, modify it and write files to the
session ``outputs`` folder.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.session import SessionPaths
from core.state import SessionState


class Skill(ABC):
    """Abstract skill exposed to the LLM as an OpenAI-style function tool."""

    #: Unique tool name used by the model to call the skill (snake_case).
    name: str = ""
    #: One-paragraph description injected into the tool schema and system prompt.
    description: str = ""
    #: JSON Schema of the arguments accepted by :meth:`run`.
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    # -------------------------------------------------------------- execution
    @abstractmethod
    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Execute the skill.

        Args:
            state: Mutable state of the current session.
            paths: Folders of the current session.
            **arguments: Arguments produced by the model, matching ``parameters``.

        Returns:
            A JSON-serialisable dictionary that is sent back to the model as the
            tool result. Implementations should include a ``"status"`` key with
            ``"ok"`` or ``"error"`` and a ``"message"`` the model can relay.
        """

    # -------------------------------------------------------------- utilities
    def to_openai_tool(self) -> dict[str, Any]:
        """Serialise the skill as an OpenAI/LiteLLM function tool definition.

        Returns:
            The tool dictionary expected by ``litellm.acompletion(tools=...)``.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def load_skill_markdown(self) -> str:
        """Read the ``SKILL.md`` file that sits next to the skill module.

        Returns:
            The markdown content, or an empty string when the file is missing.
        """
        module_file = Path(__import__(self.__class__.__module__, fromlist=["__file__"]).__file__)
        skill_md = module_file.parent / "SKILL.md"

        # A skill without documentation is still usable; just return nothing.
        if not skill_md.is_file():
            return ""

        return skill_md.read_text(encoding="utf-8")

    @staticmethod
    def ok(message: str, **payload: Any) -> dict[str, Any]:
        """Build a successful tool result.

        Args:
            message: Plain-language summary for the model.
            **payload: Extra JSON-serialisable data.

        Returns:
            Result dictionary with ``status="ok"``.
        """
        return {"status": "ok", "message": message, **payload}

    @staticmethod
    def error(message: str, **payload: Any) -> dict[str, Any]:
        """Build a failed tool result.

        Args:
            message: Plain-language explanation of the failure.
            **payload: Extra JSON-serialisable data.

        Returns:
            Result dictionary with ``status="error"``.
        """
        return {"status": "error", "message": message, **payload}
