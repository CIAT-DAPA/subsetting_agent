"""Session state of the subset being built.

The agent is stateless between chat turns, so the current selection of
accessions lives in an :class:`AccessionContext` that is serialized to JSON,
stored in the Gradio history and rebuilt on every call. The context is also
what enforces the business order: climate tools only ever operate on the
cellids of the accessions that survived the passport, trait and document
stages.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from genesys_sdk.models import Accession
from subsetting_sdk.models import CropCellIds


class Stage(str, Enum):
    """Stages of the subset-building workflow, in business order.

    Attributes:
        EMPTY: No selection yet.
        PASSPORT: Accessions selected with passport filters.
        TRAITS: Selection reduced with trait observations.
        DOCUMENTS: Selection informed or reduced with user documents.
        CLIMATE: Selection filtered or clustered with climate indicators.
    """

    EMPTY = "empty"
    PASSPORT = "passport"
    TRAITS = "traits"
    DOCUMENTS = "documents"
    CLIMATE = "climate"


# Order used to validate that stages are reached in sequence.
STAGE_ORDER = [Stage.EMPTY, Stage.PASSPORT, Stage.TRAITS, Stage.DOCUMENTS, Stage.CLIMATE]


@dataclass
class AccessionRecord:
    """Compact view of an accession kept in the session state.

    Attributes:
        uuid: Genesys accession UUID.
        accession_number: Genebank accession number.
        institute_code: FAO WIEWS code of the holding genebank.
        crop: Genesys crop code (or crop name when unmapped).
        taxon_name: Full taxon name.
        country_code: ISO3 country of origin.
        latitude: Collecting-site latitude.
        longitude: Collecting-site longitude.
        cellid: Subsetting API grid cell of the collecting site.
        doi: Accession DOI, if any.
    """

    uuid: str
    accession_number: str | None = None
    institute_code: str | None = None
    crop: str | None = None
    taxon_name: str | None = None
    country_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    cellid: int | None = None
    doi: str | None = None

    @classmethod
    def from_accession(cls, accession: Accession) -> AccessionRecord:
        """Build a record from a Genesys accession.

        Args:
            accession: Accession returned by the Genesys SDK.
        """
        return cls(
            uuid=accession.uuid,
            accession_number=accession.accession_number,
            institute_code=accession.institute_code,
            crop=accession.crop_code,
            taxon_name=accession.taxon_name,
            country_code=accession.country_code,
            latitude=accession.latitude,
            longitude=accession.longitude,
            cellid=accession.cellid(),
            doi=accession.doi,
        )

    def label(self) -> str:
        """Return a short human-readable identifier for chat answers."""
        number = self.accession_number or self.uuid
        institute = f" ({self.institute_code})" if self.institute_code else ""

        return f"{number}{institute}"


@dataclass
class StepRecord:
    """One step applied to the selection, kept for traceability.

    Attributes:
        stage: Stage the step belongs to.
        description: What was done (tool and arguments in plain words).
        before: Number of accessions before the step.
        after: Number of accessions after the step.
    """

    stage: str
    description: str
    before: int
    after: int


@dataclass
class AccessionContext:
    """Current selection of accessions and the steps that produced it.

    Attributes:
        accessions: Selected accessions keyed by UUID (insertion order kept).
        passport_filter: Serialized Genesys filter of the passport stage.
        total_matching: Accessions matching the passport filter on the server.
        stage: Highest stage reached so far.
        steps: Applied steps, in order.
        clusters: Result of the last climate clustering: label -> cellids.
        cluster_algorithm: Algorithm of the last clustering.
        selected_cluster: Cluster label chosen by the user, if any.
    """

    accessions: dict[str, AccessionRecord] = field(default_factory=dict)
    passport_filter: dict[str, Any] = field(default_factory=dict)
    total_matching: int = 0
    stage: Stage = Stage.EMPTY
    steps: list[StepRecord] = field(default_factory=list)
    clusters: dict[str, list[int]] = field(default_factory=dict)
    cluster_algorithm: str | None = None
    selected_cluster: str | None = None

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    @property
    def is_empty(self) -> bool:
        """Whether no accession is selected."""
        return not self.accessions

    @property
    def count(self) -> int:
        """Number of selected accessions."""
        return len(self.accessions)

    @property
    def truncated(self) -> bool:
        """Whether the passport stage fetched fewer accessions than matched."""
        return self.total_matching > self._passport_count

    @property
    def _passport_count(self) -> int:
        """Number of accessions right after the passport stage."""
        # The first recorded step is always the passport selection.
        for step in self.steps:
            if step.stage == Stage.PASSPORT.value:
                return step.after

        return self.count

    def uuids(self) -> list[str]:
        """Return the UUIDs of the selected accessions."""
        return list(self.accessions)

    def records(self) -> list[AccessionRecord]:
        """Return the selected accessions in insertion order."""
        return list(self.accessions.values())

    def with_cellid(self) -> list[AccessionRecord]:
        """Return the selected accessions that have a grid cell."""
        return [record for record in self.accessions.values() if record.cellid is not None]

    def cellids_by_crop(self) -> list[CropCellIds]:
        """Group the cellids of the selection by crop, as the Subsetting API expects.

        Accessions without a crop are grouped under ``"unknown"``; accessions
        without a cellid are skipped because the API cannot place them.
        """
        grouped: dict[str, list[int]] = {}

        # Collect distinct cellids per crop, preserving first-seen order.
        for record in self.with_cellid():
            crop = (record.crop or "unknown").lower()
            cells = grouped.setdefault(crop, [])

            if record.cellid not in cells:
                cells.append(record.cellid)  # type: ignore[arg-type]

        return [CropCellIds(crop=crop, cellids=cells) for crop, cells in grouped.items()]

    def accessions_in_cells(self, cellids: list[int]) -> list[AccessionRecord]:
        """Return the selected accessions located in any of the given cells.

        Args:
            cellids: Grid cells to look up.
        """
        wanted = set(cellids)

        return [record for record in self.accessions.values() if record.cellid in wanted]

    def crops(self) -> list[str]:
        """Return the distinct crops of the selection."""
        return list(
            dict.fromkeys((record.crop or "unknown") for record in self.accessions.values())
        )

    def countries(self) -> dict[str, int]:
        """Return accession counts per country of origin."""
        counts: dict[str, int] = {}

        # Count accessions per ISO3 code; missing codes are grouped as unknown.
        for record in self.accessions.values():
            key = record.country_code or "unknown"
            counts[key] = counts.get(key, 0) + 1

        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def set_passport_selection(
        self,
        accessions: list[Accession],
        *,
        passport_filter: dict[str, Any],
        total_matching: int,
        description: str,
    ) -> None:
        """Start a new selection from a passport query, discarding previous state.

        Args:
            accessions: Accessions returned by Genesys.
            passport_filter: Serialized filter that produced them.
            total_matching: Total matching accessions on the server.
            description: Plain-words description of the query.
        """
        self.accessions = {a.uuid: AccessionRecord.from_accession(a) for a in accessions}
        self.passport_filter = passport_filter
        self.total_matching = total_matching
        self.stage = Stage.PASSPORT
        self.steps = [
            StepRecord(
                stage=Stage.PASSPORT.value,
                description=description,
                before=0,
                after=len(self.accessions),
            )
        ]
        self.clusters = {}
        self.cluster_algorithm = None
        self.selected_cluster = None

    def keep(self, uuids: list[str], *, stage: Stage, description: str) -> int:
        """Reduce the selection to the given UUIDs and record the step.

        Args:
            uuids: UUIDs to keep; unknown ones are ignored.
            stage: Stage of the step; the context stage advances to it when
                it is later in the business order.
            description: Plain-words description of the criterion applied.

        Returns:
            Number of accessions kept.
        """
        before = self.count
        wanted = set(uuids)
        self.accessions = {u: r for u, r in self.accessions.items() if u in wanted}

        # The stage only moves forward: a trait filter after clustering does
        # not send the workflow back to the traits stage.
        if STAGE_ORDER.index(stage) > STAGE_ORDER.index(self.stage):
            self.stage = stage

        self.steps.append(
            StepRecord(stage=stage.value, description=description, before=before, after=self.count)
        )

        return self.count

    def set_clusters(self, clusters: dict[int, list[int]], algorithm: str) -> None:
        """Store the result of a climate clustering.

        Args:
            clusters: Cluster label -> cellids.
            algorithm: Algorithm name used.
        """
        self.clusters = {str(label): list(cells) for label, cells in clusters.items()}
        self.cluster_algorithm = algorithm
        self.selected_cluster = None

        if STAGE_ORDER.index(Stage.CLIMATE) > STAGE_ORDER.index(self.stage):
            self.stage = Stage.CLIMATE

    def cluster_sizes(self) -> dict[str, int]:
        """Return the number of selected accessions per cluster label."""
        return {
            label: len(self.accessions_in_cells(cells)) for label, cells in self.clusters.items()
        }

    # ------------------------------------------------------------------ #
    # Summaries and serialization
    # ------------------------------------------------------------------ #

    def summary(self, *, sample: int = 5) -> dict[str, Any]:
        """Return a compact description of the state for the LLM.

        Args:
            sample: Number of example accessions to include.
        """
        return {
            "stage": self.stage.value,
            "accession_count": self.count,
            "with_coordinates": len(self.with_cellid()),
            "total_matching_passport_filter": self.total_matching,
            "truncated": self.truncated,
            "crops": self.crops(),
            "countries": self.countries(),
            "steps": [asdict(step) for step in self.steps],
            "clusters": self.cluster_sizes() if self.clusters else None,
            "cluster_algorithm": self.cluster_algorithm,
            "selected_cluster": self.selected_cluster,
            "sample": [record.label() for record in self.records()[:sample]],
        }

    def to_json(self) -> str:
        """Serialize the whole context to a JSON string."""
        payload = {
            "accessions": [asdict(record) for record in self.accessions.values()],
            "passport_filter": self.passport_filter,
            "total_matching": self.total_matching,
            "stage": self.stage.value,
            "steps": [asdict(step) for step in self.steps],
            "clusters": self.clusters,
            "cluster_algorithm": self.cluster_algorithm,
            "selected_cluster": self.selected_cluster,
        }

        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str | None) -> AccessionContext:
        """Rebuild a context from :meth:`to_json` output.

        Args:
            text: JSON string, or ``None``/empty for a fresh context.
        """
        # Missing or blank state means the conversation has no selection yet.
        if not text or not text.strip():
            return cls()

        payload = json.loads(text)
        context = cls(
            passport_filter=payload.get("passport_filter", {}),
            total_matching=payload.get("total_matching", 0),
            stage=Stage(payload.get("stage", Stage.EMPTY.value)),
            steps=[StepRecord(**step) for step in payload.get("steps", [])],
            clusters={k: list(v) for k, v in payload.get("clusters", {}).items()},
            cluster_algorithm=payload.get("cluster_algorithm"),
            selected_cluster=payload.get("selected_cluster"),
        )

        # Records are rebuilt in the stored order to keep sampling stable.
        for item in payload.get("accessions", []):
            record = AccessionRecord(**item)
            context.accessions[record.uuid] = record

        return context
