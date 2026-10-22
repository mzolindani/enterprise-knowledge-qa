
from app.generation.models import GeneratedAnswer
from app.generation.ollama_provider import OllamaAnswerProvider
from app.generation.validator import validate_answer
from app.retrieval.hybrid_service import HybridRetrievalService

_INSUFFICIENT_EVIDENCE = (
    "I could not verify a sufficiently supported answer from the "
    "available documents."
)


class KnowledgeQAService:
    """Retrieve evidence, generate an answer, and validate its support."""

    def __init__(
        self,
        retriever: HybridRetrievalService,
        answer_provider: OllamaAnswerProvider,
        retrieval_k: int = 5,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero.")

        self.retriever = retriever
        self.answer_provider = answer_provider
        self.retrieval_k = retrieval_k

    def ask(self, question: str) -> GeneratedAnswer:
        """Return an answer only when its cited evidence passes validation."""
        if not question.strip():
            raise ValueError("question cannot be empty.")

        evidence = self.retriever.search(
            question,
            top_k=self.retrieval_k,
        )

        generated = self.answer_provider.generate(
            question,
            evidence,
        )

        if not evidence:
            return GeneratedAnswer(
                question=question,
                answer=_INSUFFICIENT_EVIDENCE,
                citations=(),
                model=generated.model,
                grounded=False,
            )

        # Only accept citations that refer to retrieved chunks.
        evidence_by_id = {
            (item.document_id, item.chunk_id): item
            for item in evidence
        }

        cited_evidence = []
        seen_ids = set()
        invalid_citation = False

        for citation in generated.citations:
            citation_id = (
                citation.document_id,
                citation.chunk_id,
            )

            item = evidence_by_id.get(citation_id)

            if item is None:
                invalid_citation = True
                break

            if citation_id not in seen_ids:
                cited_evidence.append(item)
                seen_ids.add(citation_id)

        # No citation, or a citation that points outside the retrieved
        # evidence, means the answer cannot be accepted.
        if invalid_citation or not cited_evidence:
            return GeneratedAnswer(
                question=question,
                answer=_INSUFFICIENT_EVIDENCE,
                citations=(),
                model=generated.model,
                grounded=False,
            )

        validation = validate_answer(
            generated.answer,
            cited_evidence,
        )

        if not validation.valid:
            return GeneratedAnswer(
                question=question,
                answer=_INSUFFICIENT_EVIDENCE,
                citations=(),
                model=generated.model,
                grounded=False,
            )

        return GeneratedAnswer(
            question=generated.question,
            answer=generated.answer,
            citations=generated.citations,
            model=generated.model,
            grounded=True,
        )
