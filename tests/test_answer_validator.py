
from app.generation.validator import validate_answer
from app.retrieval.models import RetrievalResult


def make_evidence(text):
    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="document-001",
        document_name="incident_policy.pdf",
        text=text,
        page_number=1,
        section="Incident Reporting",
        score=0.9,
    )


def test_accepts_answer_with_supported_policy_facts():
    evidence = [
        make_evidence(
            "Employees must report workplace safety incidents to their "
            "line manager immediately. The manager must record the "
            "incident in the company safety register. Serious incidents "
            "must be escalated to the health and safety officer."
        )
    ]

    result = validate_answer(
        "Employees must report workplace safety incidents to their "
        "line manager immediately.",
        evidence,
    )

    assert result.valid is True
    assert result.unsupported_claims == ()


def test_rejects_invented_24_hour_deadline():
    evidence = [
        make_evidence(
            "Employees must report workplace safety incidents to their "
            "line manager immediately. The manager must record the "
            "incident in the company safety register. Serious incidents "
            "must be escalated to the health and safety officer."
        )
    ]

    result = validate_answer(
        "The manager must report the incident to the safety department "
        "within 24 hours of the incident.",
        evidence,
    )

    assert result.valid is False
    assert len(result.unsupported_claims) == 1
    assert "24 hours" in result.unsupported_claims[0]


def test_rejects_answer_without_evidence():
    result = validate_answer(
        "Employees must report incidents immediately.",
        [],
    )

    assert result.valid is False
    assert result.unsupported_claims


def test_rejects_blank_answer():
    result = validate_answer("  ", [make_evidence("Incident policy.")])

    assert result.valid is False
    assert result.reason == "The answer is empty."



def test_rejects_answer_that_negates_a_policy_requirement():
    evidence = [
        make_evidence(
            "Employees must report workplace safety incidents to their "
            "line manager immediately."
        )
    ]

    result = validate_answer(
        "Employees must not report workplace safety incidents to their "
        "line manager immediately.",
        evidence,
    )

    assert result.valid is False
    assert len(result.unsupported_claims) == 1
