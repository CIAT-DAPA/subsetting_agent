"""Tools backed by the document store: list, search and read user PDFs."""

from __future__ import annotations

from typing import Any

from tools.accession_context import Stage
from tools.services import ToolServices, empty_selection_error


async def list_documents(services: ToolServices) -> dict[str, Any]:
    """List the documents the user uploaded in this conversation.

    Args:
        services: Shared services.
    """
    documents = [
        {
            "document_id": d.document_id,
            "title": d.title,
            "file_name": d.file_name,
            "pages": d.page_count,
            "sections": len(services.documents.get_outline(d.document_id)),
        }
        for d in services.documents.list_documents()
    ]

    return {"count": len(documents), "documents": documents}


async def search_documents(
    services: ToolServices,
    *,
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Search the uploaded documents for passages relevant to a query.

    Args:
        services: Shared services.
        query: Keywords to look for (any language).
        top_k: Maximum number of passages.
        document_id: Restrict the search to one document.
    """
    # Searching is meaningless without documents; say so instead of returning nothing.
    if services.documents.is_empty:
        return {"error": "No documents uploaded. Ask the user to attach a PDF."}

    hits = services.documents.search(
        query, top_k=max(1, min(int(top_k), 10)), document_id=document_id
    )

    # Record that documents informed the workflow, without changing the selection.
    if hits and services.context.stage in (Stage.PASSPORT, Stage.TRAITS):
        services.context.stage = Stage.DOCUMENTS

    return {
        "query": query,
        "count": len(hits),
        "passages": [
            {
                "document_id": hit.section.document_id,
                "section_index": hit.section.index,
                "heading": hit.section.heading,
                "matched_terms": hit.matched_terms,
                "snippet": hit.snippet,
            }
            for hit in hits
        ],
    }


async def read_document_section(
    services: ToolServices, *, document_id: str, section_index: int
) -> dict[str, Any]:
    """Read the full text of one section of an uploaded document.

    Args:
        services: Shared services.
        document_id: Document id from ``list_documents`` or ``search_documents``.
        section_index: Section index from the outline or a search hit.
    """
    try:
        section = services.documents.read_section(document_id, int(section_index))

    except (KeyError, IndexError) as exc:
        return {"error": str(exc)}

    outline = services.documents.get_outline(document_id)

    return {
        "document_id": document_id,
        "section_index": section.index,
        "heading": section.heading,
        "text": section.text,
        "total_sections": len(outline),
    }


async def keep_accessions_from_documents(
    services: ToolServices, *, accession_numbers: list[str], reason: str
) -> dict[str, Any]:
    """Reduce the selection to accession numbers cited in the documents.

    The LLM reads passages, extracts the accession numbers the paper reports as
    relevant and passes them here; the tool applies the reduction and records
    the reason for traceability.

    Args:
        services: Shared services.
        accession_numbers: Accession numbers (or UUIDs) to keep.
        reason: Why these accessions were chosen (cite the document).
    """
    context = services.context

    if context.is_empty:
        return empty_selection_error(services)

    wanted = {value.strip().lower() for value in accession_numbers if value and value.strip()}

    if not wanted:
        return {"error": "Provide at least one accession number."}

    kept = [
        record.uuid
        for record in context.records()
        if (record.accession_number or "").lower() in wanted or record.uuid.lower() in wanted
    ]
    missing = sorted(
        wanted
        - {(r.accession_number or "").lower() for r in context.records()}
        - {r.uuid.lower() for r in context.records()}
    )

    # Refuse to empty the selection by mistake; the model must re-check numbers.
    if not kept:
        return {
            "error": "None of the accession numbers are in the current selection.",
            "not_found": missing[:20],
        }

    context.keep(kept, stage=Stage.DOCUMENTS, description=f"Documents: {reason}")

    # The reason (document title and passage) is the evidence of this stage.
    context.add_evidence(kept, **{"document evidence": reason})

    return {"kept": len(kept), "not_found": missing[:20], "summary": context.summary()}
