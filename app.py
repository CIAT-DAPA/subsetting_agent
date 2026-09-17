"""Gradio front-end of the subsetting agent.

Follows the AClimate ``app.py`` pattern: a new agent per call, no state shared
between users, and the conversation rebuilt from the per-browser history that
Gradio passes on every call. Two additions:

* the chat is multimodal so users can attach PDF papers and an accession
  spreadsheet (Excel/CSV). The paths of every file attached so far are collected
  from the history and handed to the agent, which decides the accession source:
  a spreadsheet means file mode, otherwise the Genesys API;
* the accession selection (``AccessionContext``) is carried between turns in a
  ``gr.State`` component through ``additional_inputs``/``additional_outputs``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from subsetting_agent import SubsettingAgent

load_dotenv()

SUBSETTING_AGENT_MODEL = os.getenv("SUBSETTING_AGENT_MODEL", "ollama_chat/llama3.1:8b")
SUBSETTING_AGENT_API_BASE = os.getenv("SUBSETTING_AGENT_API_BASE", "http://localhost:11434")
SUBSETTING_AGENT_HOST = os.getenv("SUBSETTING_AGENT_HOST", "localhost")
SUBSETTING_AGENT_PORT = int(os.getenv("SUBSETTING_AGENT_PORT", "7860"))
LOG_LEVEL = os.getenv("SUBSETTING_AGENT_LOG_LEVEL", "INFO")

# How many recent history messages are passed to the model. Bounds the context
# window use (num_ctx) in long conversations.
MAX_HISTORY_MESSAGES = 20

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_text(content: Any) -> str:
    """Get the plain text out of a Gradio message content.

    Gradio may pass content as a plain string, as a list of blocks
    (``[{"type": "text", "text": "..."}]``) or as a file descriptor. Non-text
    blocks are ignored here; files are handled by :func:`extract_file_paths`.

    Args:
        content: ``content`` field of a Gradio history message.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )

    return ""


def extract_file_paths(content: Any) -> list[str]:
    """Get the file paths referenced by a Gradio message content.

    Recognized shapes: ``{"path": ...}`` (a single file message), ``(path, alt)``
    tuples used by older Gradio versions, lists of blocks with a ``file`` or
    ``path`` entry, and Gradio ``FileData`` objects exposing ``.path``.

    Args:
        content: ``content`` field of a Gradio history message.
    """
    paths: list[str] = []

    # A dictionary with a path is one file; a dictionary with a nested file
    # descriptor is one block.
    if isinstance(content, dict):
        if isinstance(content.get("path"), str):
            paths.append(content["path"])

        elif isinstance(content.get("file"), dict) and isinstance(content["file"].get("path"), str):
            paths.append(content["file"]["path"])

    # Older Gradio versions pass files as (path, alt_text) tuples.
    elif isinstance(content, tuple) and content and isinstance(content[0], str):
        paths.append(content[0])

    # Block lists may mix text and file blocks; recurse into each block.
    elif isinstance(content, list):
        for block in content:
            paths.extend(extract_file_paths(block))

    # FileData-like objects expose the path as an attribute.
    elif hasattr(content, "path") and isinstance(content.path, str):
        paths.append(content.path)

    return paths


ACCESSION_FILE_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".csv", ".tsv"})


def is_pdf(path: str) -> bool:
    """Whether a path points to a PDF file (by extension).

    Args:
        path: File path.
    """
    return Path(path).suffix.lower() == ".pdf"


def is_accession_file(path: str) -> bool:
    """Whether a path points to a spreadsheet that may hold an accession list.

    Args:
        path: File path.
    """
    return Path(path).suffix.lower() in ACCESSION_FILE_SUFFIXES


def build_memory_from_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the agent memory from Gradio's per-session chat history.

    Gradio keeps one history per browser session and passes it on every call,
    so the agent stays stateless: no memory is shared between users and the
    conversation survives across turns. Only text is kept; tool traces are not
    part of the visible history and are therefore not replayed.

    Args:
        history: OpenAI-style messages kept by Gradio.
    """
    memory: list[dict[str, Any]] = []

    # Keep user/assistant text messages only, in order.
    for message in history:
        if not isinstance(message, dict):
            continue

        if message.get("role") not in ("user", "assistant"):
            continue

        text = extract_text(message.get("content")).strip()

        if text:
            memory.append({"role": message["role"], "content": text})

    return memory[-MAX_HISTORY_MESSAGES:]


def collect_attachments(history: list[dict[str, Any]], current_files: list[Any]) -> list[str]:
    """Collect every file attached in the conversation, oldest first, without duplicates.

    Args:
        history: OpenAI-style messages kept by Gradio.
        current_files: ``files`` entry of the current multimodal message.
    """
    paths: list[str] = []

    # Files from previous turns live inside user messages of the history.
    for message in history:
        if isinstance(message, dict) and message.get("role") == "user":
            paths.extend(extract_file_paths(message.get("content")))

    # Files of the current message come separately in the multimodal payload.
    for item in current_files or []:
        paths.extend(extract_file_paths(item) or ([item] if isinstance(item, str) else []))

    return list(dict.fromkeys(paths))


def collect_pdf_paths(history: list[dict[str, Any]], current_files: list[Any]) -> list[str]:
    """Collect the PDFs attached in the conversation.

    Args:
        history: OpenAI-style messages kept by Gradio.
        current_files: ``files`` entry of the current multimodal message.
    """
    return [p for p in collect_attachments(history, current_files) if is_pdf(p)]


def collect_accession_file_paths(
    history: list[dict[str, Any]], current_files: list[Any]
) -> list[str]:
    """Collect the accession spreadsheets attached in the conversation.

    Args:
        history: OpenAI-style messages kept by Gradio.
        current_files: ``files`` entry of the current multimodal message.
    """
    return [p for p in collect_attachments(history, current_files) if is_accession_file(p)]


def format_document_errors(errors: list[str]) -> str:
    """Render PDF conversion errors as a short note appended to the answer.

    Args:
        errors: Messages produced by the document store.
    """
    if not errors:
        return ""

    lines = "\n".join(f"- {error}" for error in errors)

    return f"\n\n_Some attached files could not be processed:_\n{lines}"


async def chat(
    message: dict[str, Any] | str,
    history: list[dict[str, Any]],
    context_json: str,
) -> tuple[str, str]:
    """Handle one chat turn.

    A new agent is created per call so nothing is shared between users; the
    session lives in the history Gradio keeps per browser and in the
    ``context_json`` state.

    Args:
        message: Multimodal message ``{"text": ..., "files": [...]}`` (or a
            plain string when multimodal input is disabled).
        history: OpenAI-style messages of the conversation so far.
        context_json: Serialized accession selection from the previous turn.

    Returns:
        The assistant answer and the updated serialized selection.
    """
    # Normalize both message shapes into text plus files.
    if isinstance(message, dict):
        text = str(message.get("text") or "")
        files = list(message.get("files") or [])

    else:
        text = str(message or "")
        files = []

    document_paths = collect_pdf_paths(history, files)
    accession_file_paths = collect_accession_file_paths(history, files)

    # A message with only attachments still deserves an answer.
    if not text.strip() and accession_file_paths:
        text = "I attached my accession list. Load it and tell me what it contains."

    elif not text.strip() and document_paths:
        text = "I attached a document. Tell me what you found in it."

    agent = SubsettingAgent(model=SUBSETTING_AGENT_MODEL, api_base=SUBSETTING_AGENT_API_BASE)
    agent.memory = build_memory_from_history(history)

    try:
        turn = await agent.chat(
            text,
            document_paths=document_paths,
            accession_file_paths=accession_file_paths,
            context_json=context_json,
        )

    except Exception:  # noqa: BLE001 - the UI must always answer something
        logger.exception("Agent turn failed")

        return (
            "Something went wrong while processing your request. Please try again or rephrase it.",
            context_json,
        )

    return turn.answer + format_document_errors(turn.document_errors), turn.context_json


def build_app() -> gr.Blocks:
    """Create the Gradio application."""
    with gr.Blocks(title="Genesys Subsetting Assistant") as demo:
        # Serialized AccessionContext carried between turns, per browser session.
        selection_state = gr.State("")

        gr.ChatInterface(
            fn=chat,
            multimodal=True,
            additional_inputs=[selection_state],
            additional_outputs=[selection_state],
            title="Genesys Subsetting Assistant",
            description=(
                "Build subsets of genebank accessions from passport data, traits, your own "
                "papers (attach PDFs) and the climate of the collecting sites. Attach an "
                "Excel/CSV with accession ids and coordinates to work from your own list "
                "instead of Genesys."
            ),
            textbox=gr.MultimodalTextbox(
                file_types=[".pdf", ".xlsx", ".xls", ".csv", ".tsv"],
                file_count="multiple",
                placeholder="e.g. Find bean landraces from Colombia collected in dry areas",
            ),
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name=SUBSETTING_AGENT_HOST, server_port=SUBSETTING_AGENT_PORT)
