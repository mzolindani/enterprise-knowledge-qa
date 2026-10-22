
import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import RetrievalResult


def make_result(chunk_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document-001",
        document_name="safety-policy.pdf",
        text=f"Text for {chunk_id}",
        page_number=1,
        section="Safety",
        score=score,
    )


def test_rrf_combines_results_from_multiple_retrievers():
    bm25_results = [
        make_result("shared"),
        make_result("lexical-only"),
    ]
    vector_results = [
        make_result("shared"),
        make_result("semantic-only"),
    ]

    results = reciprocal_rank_fusion(
        [bm25_results, vector_results],
    )

    assert results[0].chunk_id == "shared"
    assert {result.chunk_id for result in results} == {
        "shared",
        "lexical-only",
        "semantic-only",
    }
    assert results[0].score > results[1].score


def test_rrf_respects_top_k():
    results = reciprocal_rank_fusion(
        [[make_result("a"), make_result("b"), make_result("c")]],
        top_k=2,
    )

    assert len(results) == 2


def test_rrf_handles_empty_result_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=0)


def test_rrf_rejects_invalid_rank_constant():
    with pytest.raises(ValueError, match="rank_constant"):
        reciprocal_rank_fusion([], rank_constant=0)


def test_rrf_does_not_double_count_duplicate_chunks_in_one_list():
    results = reciprocal_rank_fusion(
        [[make_result("a"), make_result("a"), make_result("b")]],
    )

    scores = {result.chunk_id: result.score for result in results}

    assert len(results) == 2
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 63)
