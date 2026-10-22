
from collections import defaultdict
from collections.abc import Sequence

from app.retrieval.models import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    top_k: int = 5,
    rank_constant: int = 60,
) -> list[RetrievalResult]:
    """Combine ranked retrieval lists using Reciprocal Rank Fusion."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if rank_constant <= 0:
        raise ValueError("rank_constant must be greater than zero.")

    fused_scores: dict[str, float] = defaultdict(float)
    result_by_id: dict[str, RetrievalResult] = {}

    for results in result_lists:
        seen_in_list: set[str] = set()

        for rank, result in enumerate(results, start=1):
            if result.chunk_id in seen_in_list:
                continue

            seen_in_list.add(result.chunk_id)
            fused_scores[result.chunk_id] += (
                1.0 / (rank_constant + rank)
            )

            # Preserve the result metadata from the first occurrence.
            result_by_id.setdefault(result.chunk_id, result)

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            chunk_id,
        ),
    )

    return [
        RetrievalResult(
            chunk_id=result_by_id[chunk_id].chunk_id,
            document_id=result_by_id[chunk_id].document_id,
            document_name=result_by_id[chunk_id].document_name,
            text=result_by_id[chunk_id].text,
            page_number=result_by_id[chunk_id].page_number,
            section=result_by_id[chunk_id].section,
            score=fused_scores[chunk_id],
        )
        for chunk_id in ranked_ids[:top_k]
    ]
