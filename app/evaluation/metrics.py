
from collections.abc import Sequence


def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return 1.0 if any relevant chunk appears in the top k."""
    _validate_k(k)

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty.")

    return float(bool(set(retrieved_ids[:k]) & relevant_ids))


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of relevant chunks found in the top k."""
    _validate_k(k)

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty.")

    retrieved_relevant = set(retrieved_ids[:k]) & relevant_ids
    return len(retrieved_relevant) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant chunk."""
    _validate_k(k)

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty.")

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
