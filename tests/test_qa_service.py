
import pytest

from app.generation import GeneratedAnswer, KnowledgeQAService
from app.generation.models import SourceCitation
from app.retrieval.models import RetrievalResult


def make_result():
    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="document-001",
        document_name="safety-policy.pdf",
        text="Employees must report incidents to their manager.",
        page_number=2,
        section="Incident Reporting",
        score=0.85,
    )


class FakeRetriever:
    def __init__(self, results=None):
        self.results = [] if results is None else results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeAnswerProvider:
    def __init__(self):
        self.calls = []

    def generate(self, question, evidence):
        self.calls.append((question, evidence))
        citations = tuple(
            SourceCitation(
                document_id=item.document_id,
                document_name=item.document_name,
                page_number=item.page_number,
                chunk_id=item.chunk_id,
                excerpt=item.text,
            )
            for item in evidence
        )

        return GeneratedAnswer(
            question=question,
            answer="Employees must report incidents to their manager.",
            citations=citations,
            model="test-model",
            grounded=bool(citations),
        )


def test_ask_retrieves_evidence_and_generates_answer():
    evidence = [make_result()]
    retriever = FakeRetriever(evidence)
    provider = FakeAnswerProvider()

    service = KnowledgeQAService(
        retriever=retriever,
        answer_provider=provider,
        retrieval_k=3,
    )

    result = service.ask("How should employees report incidents?")

    assert result.answer == (
        "Employees must report incidents to their manager."
    )
    assert result.grounded is True
    assert len(result.citations) == 1
    assert retriever.calls == [
        ("How should employees report incidents?", 3)
    ]
    assert provider.calls == [
        ("How should employees report incidents?", evidence)
    ]


def test_ask_handles_no_retrieved_evidence():
    retriever = FakeRetriever()
    provider = FakeAnswerProvider()

    service = KnowledgeQAService(retriever, provider)

    result = service.ask("What is the company revenue?")

    assert result.citations == ()
    assert result.grounded is False
    assert provider.calls == [("What is the company revenue?", [])]


def test_ask_rejects_blank_question():
    service = KnowledgeQAService(
        FakeRetriever(),
        FakeAnswerProvider(),
    )

    with pytest.raises(ValueError, match="question"):
        service.ask("   ")


def test_service_rejects_invalid_retrieval_k():
    with pytest.raises(ValueError, match="retrieval_k"):
        KnowledgeQAService(
            FakeRetriever(),
            FakeAnswerProvider(),
            retrieval_k=0,
        )

def test_ask_rejects_answer_with_unsupported_deadline():
    evidence = [
        RetrievalResult(
            chunk_id="policy-001",
            document_id="policy-doc",
            document_name="incident_policy.pdf",
            text=(
                "Employees must report workplace safety incidents to "
                "their line manager immediately. The manager must record "
                "the incident in the company safety register."
            ),
            page_number=1,
            section="Incident Reporting",
            score=0.9,
        )
    ]

    class HallucinatingProvider:
        def generate(self, question, evidence):
            return GeneratedAnswer(
                question=question,
                answer=(
                    "The manager must report the incident within "
                    "24 hours of the incident. [Source 1]"
                ),
                citations=(
                    SourceCitation(
                        document_id="policy-doc",
                        document_name="incident_policy.pdf",
                        page_number=1,
                        chunk_id="policy-001",
                        excerpt=evidence[0].text,
                    ),
                ),
                model="test-model",
                grounded=True,
            )

    service = KnowledgeQAService(
        FakeRetriever(evidence),
        HallucinatingProvider(),
    )

    result = service.ask("What must happen after an incident?")

    assert result.grounded is False
    assert result.citations == ()
    assert "could not verify" in result.answer.lower()
    assert "24 hours" not in result.answer.lower()


def test_ask_returns_supported_answer_after_validation():
    evidence = [make_result()]

    service = KnowledgeQAService(
        FakeRetriever(evidence),
        FakeAnswerProvider(),
    )

    result = service.ask("How should employees report incidents?")

    assert result.answer == (
        "Employees must report incidents to their manager."
    )
    assert result.grounded is True
    assert len(result.citations) == 1


def test_ask_rejects_supported_answer_without_citations():
    evidence = [make_result()]

    class NoCitationProvider:
        def generate(self, question, evidence):
            return GeneratedAnswer(
                question=question,
                answer="Employees must report incidents to their manager.",
                citations=(),
                model="test-model",
                grounded=False,
            )

    service = KnowledgeQAService(
        FakeRetriever(evidence),
        NoCitationProvider(),
    )

    result = service.ask("How should employees report incidents?")

    assert result.grounded is False
    assert result.citations == ()
    assert "could not verify" in result.answer.lower()


def test_ask_rejects_citation_to_unretrieved_chunk():
    evidence = [make_result()]

    class FabricatedCitationProvider:
        def generate(self, question, evidence):
            return GeneratedAnswer(
                question=question,
                answer=(
                    "Employees must report incidents to their manager."
                ),
                citations=(
                    SourceCitation(
                        document_id="unknown-document",
                        document_name="fabricated.pdf",
                        page_number=99,
                        chunk_id="unknown-chunk",
                        excerpt="Fabricated supporting text.",
                    ),
                ),
                model="test-model",
                grounded=True,
            )

    service = KnowledgeQAService(
        FakeRetriever(evidence),
        FabricatedCitationProvider(),
    )

    result = service.ask("How should employees report incidents?")

    assert result.grounded is False
    assert result.citations == ()
    assert "could not verify" in result.answer.lower()
