
import pytest

from app.evaluation.metrics import (
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_hit_rate_is_one_when_relevant_chunk_is_retrieved():
    assert hit_rate_at_k(
        ["chunk-a", "chunk-b"],
        {"chunk-b"},
        k=2,
    ) == 1.0


def test_hit_rate_is_zero_when_no_relevant_chunk_is_retrieved():
    assert hit_rate_at_k(
        ["chunk-a", "chunk-b"],
        {"chunk-c"},
        k=2,
    ) == 0.0


def test_recall_at_k_counts_relevant_chunks():
    assert recall_at_k(
        ["chunk-a", "chunk-c", "chunk-b"],
        {"chunk-a", "chunk-b", "chunk-d"},
        k=3,
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank_rewards_higher_ranking():
    assert reciprocal_rank_at_k(
        ["chunk-a", "chunk-b", "chunk-c"],
        {"chunk-c"},
        k=3,
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_no_relevant_chunk_is_found():
    assert reciprocal_rank_at_k(
        ["chunk-a", "chunk-b"],
        {"chunk-c"},
        k=2,
    ) == 0.0


def test_metrics_reject_non_positive_k():
    with pytest.raises(ValueError, match="greater than zero"):
        hit_rate_at_k(["chunk-a"], {"chunk-a"}, k=0)


def test_metrics_reject_empty_relevant_ids():
    with pytest.raises(ValueError, match="cannot be empty"):
        recall_at_k(["chunk-a"], set(), k=3)
