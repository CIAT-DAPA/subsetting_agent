"""Gradio chat interface of the SubsettingAgent.

Every browser session gets its own ``session_hash`` from Gradio. That hash is
used as the session id for both the in-memory state (Original / Candidate
lists) and the temporary folders (``tmp/<session>/inputs|outputs``).
"""

from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from agent.skill_registry import SkillRegistry
from agent.subsetting_agent import SubsettingAgent
from core.config import get_settings
from core.formatter import ResponseFormatter
from core.logger import get_logger, setup_logging
from core.session import SessionManager
from core.state import SessionStore

load_dotenv()
settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

# --- Long-lived collaborators shared by every request ----------------------------
session_manager = SessionManager(settings.tmp_dir)
session_store = SessionStore()
skill_registry = SkillRegistry().discover()
formatter = ResponseFormatter(preview_rows=settings.agent_preview_rows)
agent = SubsettingAgent(
    settings=settings,
    registry=skill_registry,
    session_store=session_store,
    session_manager=session_manager,
    formatter=formatter,
)


def extract_text(content: Any) -> str:
    """Get the plain text out of a Gradio message content.

    Gradio 6 passes content as a list of blocks (``[{"text": ..., "type": "text"}]``);
    older versions pass a plain string. Non-text blocks (files) are ignored.

    Args:
        content: Message content as delivered by Gradio.

    Returns:
        The concatenated text of the message.
    """
    # Older Gradio versions deliver the text directly.
    if isinstance(content, str):
        return content

    # Gradio 6 delivers a list of typed blocks; keep the text ones only.
    if isinstance(content, list):
        return " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )

    return ""


def build_memory_from_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert Gradio's chat history into plain ``role``/``content`` messages.

    Args:
        history: Messages kept by Gradio for the current browser session.

    Returns:
        User and assistant messages with non-empty text.
    """
    memory: list[dict[str, str]] = []

    # Keep only textual user/assistant turns; tool blocks and files are dropped.
    for message in history:
        if not isinstance(message, dict):
            continue

        if message.get("role") not in ("user", "assistant"):
            continue

        text = extract_text(message.get("content")).strip()

        if text:
            memory.append({"role": message["role"], "content": text})

    return memory


def store_uploaded_files(session_id: str, files: list[Any]) -> list[Path]:
    """Copy the files attached to the current message into the session inputs.

    Args:
        session_id: Identifier of the session.
        files: File entries delivered by Gradio (paths or ``FileData`` dicts).

    Returns:
        Paths of the stored copies.
    """
    stored: list[Path] = []

    # Gradio may deliver plain paths or dictionaries with a ``path`` key.
    for entry in files:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry

        if not raw_path:
            continue

        try:
            stored.append(session_manager.store_input_file(session_id, Path(raw_path)))
        except FileNotFoundError as exc:
            logger.warning("Could not store uploaded file: %s", exc)

    return stored


async def chat(message: Any, history: list[dict[str, Any]], request: gr.Request) -> str:
    """Gradio callback executed for every user message.

    Args:
        message: ``{"text": str, "files": list}`` in multimodal mode, or a string.
        history: Chat history of this browser session.
        request: Gradio request, used to obtain the ``session_hash``.

    Returns:
        The agent answer in markdown.
    """
    # Multimodal ChatInterface sends a dict; be tolerant to the plain-string form.
    if isinstance(message, dict):
        text = message.get("text") or ""
        files = message.get("files") or []
    else:
        text = str(message)
        files = []

    session_id = request.session_hash if request and request.session_hash else "default"

    uploaded = store_uploaded_files(session_id, files)
    memory = build_memory_from_history(history)

    return await agent.chat(
        session_id=session_id,
        user_message=text,
        history=memory,
        uploaded_files=uploaded,
    )


app = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="SubsettingAgent",
    description=(
        "Assistant that builds subsets of germplasm accessions combining passport data, "
        "traits, climate indicators and research papers. "
        "Upload an Excel/CSV file or ask for accessions from Genesys PGR to start."
    ),
)


# Only launch the server when the module is executed directly (``uv run app.py``).
if __name__ == "__main__":
    app.launch(server_name=settings.app_host, server_port=settings.app_port)
