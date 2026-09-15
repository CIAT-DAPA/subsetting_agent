"""Unit tests for the request/response models of the SDK."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from subsetting_sdk.models import (
    ClusteringAlgorithm,
    ClusteringHyperparameters,
    ClusterRequest,
    ClusterResult,
    CropCellIds,
    IndicatorFilter,
    IndicatorType,
    MonthWindow,
    SubsetResult,
)


class TestMonthWindow:
    """Month windows must validate bounds and expand wrap-around ranges."""

    def test_regular_window_expands_in_order(self) -> None:
        """A window inside the year lists its months ascending."""
        assert MonthWindow(start=3, end=6).months() == [3, 4, 5, 6]

    def test_wrapping_window_crosses_year_end(self) -> None:
        """A window whose start is after its end wraps like the API does."""
        assert MonthWindow(start=11, end=2).months() == [11, 12, 1, 2]

    def test_single_month_window(self) -> None:
        """Equal start and end select exactly one month."""
        assert MonthWindow(start=7, end=7).months() == [7]

    @pytest.mark.parametrize("bad", [0, 13])
    def test_out_of_range_month_is_rejected(self, bad: int) -> None:
        """Months outside 1-12 raise a validation error."""
        with pytest.raises(ValidationError):
            MonthWindow(start=bad, end=5)

    def test_to_api_is_low_up_list(self) -> None:
        """Serialization matches the ``[low, up]`` list the API reads."""
        assert MonthWindow(start=4, end=9).to_api() == [4, 9]


class TestCropCellIds:
    """Cellid lists are deduplicated and serialized with the API keys."""

    def test_duplicates_are_removed_preserving_order(self) -> None:
        """Repeated cellids collapse to their first occurrence."""
        item = CropCellIds(crop="Bean", cellids=[3, 1, 3, 2, 1])

        assert item.cellids == [3, 1, 2]

    def test_to_api_shape(self) -> None:
        """The payload uses ``crop`` and ``cellids`` keys."""
        assert CropCellIds(crop="Bean", cellids=[1]).to_api() == {"crop": "Bean", "cellids": [1]}


class TestIndicatorFilter:
    """Filters enforce the API invariants and serialize with the API keys."""

    def test_specific_indicator_requires_crop(self) -> None:
        """Crop-specific filters without a crop would match nothing; reject them."""
        with pytest.raises(ValidationError, match="requires 'crop'"):
            IndicatorFilter(
                type=IndicatorType.SPECIFIC, name="Days optimal", indicator_periods=["abc"]
            )

    def test_inverted_range_is_rejected(self) -> None:
        """A range whose min exceeds its max is invalid."""
        with pytest.raises(ValidationError, match="min is greater than max"):
            IndicatorFilter(
                type=IndicatorType.GENERIC,
                name="Total precipitation",
                indicator_periods=["abc"],
                range=(100.0, 10.0),
            )

    def test_at_least_one_period_is_required(self) -> None:
        """An empty period list cannot be sent to the API."""
        with pytest.raises(ValidationError):
            IndicatorFilter(type=IndicatorType.GENERIC, name="x", indicator_periods=[])

    def test_to_api_uses_api_key_names(self) -> None:
        """Period ids go under ``indicator`` and the range is a list."""
        payload = IndicatorFilter(
            type=IndicatorType.GENERIC,
            name="Total precipitation",
            indicator_periods=["p1"],
            months=MonthWindow(start=5, end=9),
            range=(100.0, 500.0),
        ).to_api()

        assert payload == {
            "type": "generic",
            "name": "Total precipitation",
            "indicator": ["p1"],
            "months": [5, 9],
            "range": [100.0, 500.0],
        }

    def test_to_api_omits_missing_range_and_includes_crop(self) -> None:
        """Without a range the key is absent; specific filters carry their crop."""
        payload = IndicatorFilter(
            type=IndicatorType.SPECIFIC, name="Days optimal", indicator_periods=["p"], crop="Bean"
        ).to_api()

        assert "range" not in payload
        assert payload["crop"] == "Bean"


class TestClusterRequest:
    """The cluster request reproduces the nested body used by the web client."""

    def test_to_api_nests_analysis_block(self) -> None:
        """Algorithms and hyperparameters live under ``analysis``."""
        request = ClusterRequest(
            cellid_list=[CropCellIds(crop="Bean", cellids=[1, 2])],
            filters=[
                IndicatorFilter(
                    type=IndicatorType.GENERIC, name="Total precipitation", indicator_periods=["p"]
                )
            ],
            months=MonthWindow(start=1, end=12),
            algorithms=[ClusteringAlgorithm.AGGLOMERATIVE, ClusteringAlgorithm.DBSCAN],
            hyperparameters=ClusteringHyperparameters(n_clusters=4),
        )

        payload = request.to_api()

        assert payload["cellid_list"] == [{"crop": "Bean", "cellids": [1, 2]}]
        assert payload["months"] == [1, 12]
        assert payload["analysis"]["algorithm"] == ["agglomerative", "dbscan"]
        assert payload["analysis"]["hyperparameter"] == {
            "n_clusters": 4,
            "epsilon": 10.0,
            "minpts": 20,
            "min_cluster_size": 10,
        }

    def test_default_algorithm_is_agglomerative(self) -> None:
        """The Genesys UI default is agglomerative clustering."""
        request = ClusterRequest(
            cellid_list=[CropCellIds(crop="Bean", cellids=[1])],
            filters=[
                IndicatorFilter(type=IndicatorType.GENERIC, name="x", indicator_periods=["p"])
            ],
        )

        assert request.algorithms == [ClusteringAlgorithm.AGGLOMERATIVE]


class TestClusterResult:
    """Cluster rows are split into labels and values and grouped per algorithm."""

    def test_rows_are_parsed(self, cluster_payload: dict) -> None:
        """Each row exposes cellid, crop, labels and indicator values."""
        result = ClusterResult.from_api(cluster_payload)
        first = result.rows[0]

        assert first.cellid == 101
        assert first.crop == "bean"
        assert first.labels == {"cluster_hac": 0, "cluster_dbscan": 0}
        assert first.values == {"prec_month1": 10.0, "prec_month2": 12.5}

    def test_clusters_grouped_by_agglomerative_label(self, cluster_payload: dict) -> None:
        """Default grouping uses ``cluster_hac`` across every crop."""
        result = ClusterResult.from_api(cluster_payload)

        assert result.clusters() == {0: [101, 102, 201], 1: [103]}

    def test_clusters_filtered_by_crop(self, cluster_payload: dict) -> None:
        """Crop filtering is case-insensitive and excludes other crops."""
        result = ClusterResult.from_api(cluster_payload)

        assert result.clusters(crop="Maize") == {0: [201]}

    def test_dbscan_noise_is_excluded_by_default(self, cluster_payload: dict) -> None:
        """Label -1 is dropped unless noise is explicitly requested."""
        result = ClusterResult.from_api(cluster_payload)

        assert result.clusters(ClusteringAlgorithm.DBSCAN) == {0: [101, 201], 1: [103]}
        assert result.clusters(ClusteringAlgorithm.DBSCAN, include_noise=True)[-1] == [102]

    def test_missing_algorithm_column_yields_empty(self, cluster_payload: dict) -> None:
        """Asking for an algorithm that was not run returns no clusters."""
        result = ClusterResult.from_api(cluster_payload)

        assert result.clusters(ClusteringAlgorithm.HDBSCAN) == {}

    def test_empty_payload_is_accepted(self) -> None:
        """The API answers ``{}`` on internal failure; that parses to no rows."""
        assert ClusterResult.from_api({}).rows == []


class TestSubsetResult:
    """The subset response uses the singular ``cellid`` key for lists."""

    def test_parses_filtered_cellids_alias(self) -> None:
        """``filtered_cellids[].cellid`` is mapped to ``filtered[].cellids``."""
        result = SubsetResult.model_validate(
            {
                "filtered_cellids": [
                    {"crop": "bean", "cellid": [1, 2]},
                    {"crop": "maize", "cellid": [2, 3]},
                ],
                "quantile": [],
                "proportion": [],
            }
        )

        assert result.filtered[0].cellids == [1, 2]
        assert result.all_cellids() == [1, 2, 3]
