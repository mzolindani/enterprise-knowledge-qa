
import pytest

from app.generation import OllamaAnswerProvider
from app.retrieval.models import RetrievalResult


def make_result(
    chunk_id="chunk-001",
    text="Employees must report incidents to their manager.",
):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document-001",
        document_name="safety-policy.pdf",
        text=text,
        page_number=2,
        section="Incident Reporting",
        score=0.85,
    )


class FakeOllamaClient:
    def __init__(self, content="Employees must report incidents [Source 1]."):
        self.content = content
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "message": {
                "content": self.content,
            }
        }


def test_generate_returns_answer_and_citation():
    client = FakeOllamaClient()
    provider = OllamaAnswerProvider(client=client)

    result = provider.generate(
        "How should employees report incidents?",
        [make_result()],
    )

    assert "report incidents" in result.answer.lower()
    assert result.model == "qwen2.5:3b"
    assert result.grounded is True
    assert len(result.citations) == 1
    assert result.citations[0].document_name == "safety-policy.pdf"
    assert result.citations[0].page_number == 2
    assert result.citations[0].chunk_id == "chunk-001"


def test_generate_does_not_call_model_without_evidence():
    client = FakeOllamaClient()
    provider = OllamaAnswerProvider(client=client)

    result = provider.generate(
        "What is the company revenue?",
        [],
    )

    assert result.grounded is False
    assert result.citations == ()
    assert "could not find sufficient information" in result.answer.lower()
    assert client.calls == []


def test_generate_rejects_blank_question():
    provider = OllamaAnswerProvider(client=FakeOllamaClient())

    with pytest.raises(ValueError, match="question"):
        provider.generate("   ", [make_result()])


def test_generate_only_attaches_cited_sources():
    client = FakeOllamaClient(
        content="Employees must report incidents [Source 2]."
    )
    provider = OllamaAnswerProvider(client=client)

    result = provider.generate(
        "How should employees report incidents?",
        [
            make_result("chunk-001", "Unrelated office policy."),
            make_result("chunk-002", "Employees report incidents."),
        ],
    )

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "chunk-002"
    assert result.grounded is True


def test_generate_rejects_blank_model_name():
    with pytest.raises(ValueError, match="model"):
        OllamaAnswerProvider(model="   ", client=FakeOllamaClient())



def test_prompt_requires_supported_facts_and_preserves_requirements():
    client = FakeOllamaClient()
    provider = OllamaAnswerProvider(client=client)

    provider.generate(
        "Who must employees report incidents to?",
        [
            make_result(
                text=(
                    "Employees must report incidents to their line manager. "
                    "The manager must record the incident in the safety "
                    "register. Serious incidents must be escalated to the "
                    "health and safety officer."
                )
            )
        ],
    )

    prompt = client.calls[0]["messages"][1]["content"].lower()

    assert "explicitly supported by the evidence" in prompt
    assert "preserve relevant requirements" in prompt
    assert "escalation requirements" in prompt
    assert "[source 1]" in prompt


def test_uncited_answer_is_not_marked_grounded():
    client = FakeOllamaClient(
        content="Employees should report incidents to their manager."
    )
    provider = OllamaAnswerProvider(client=client)

    result = provider.generate(
        "How should employees report incidents?",
        [make_result()],
    )

    assert result.citations == ()
    assert result.grounded is False
