
import re
from collections.abc import Sequence

from app.generation.models import AnswerValidation
from app.retrieval.models import RetrievalResult

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "in", "is", "it", "must", "of", "on", "or", "that",
    "the", "their", "then", "this", "to", "was", "were", "will",
    "with", "within",
}

_NEGATION_TOKENS = {
    "not", "never", "no", "cannot", "can't", "couldn't",
    "didn't", "doesn't", "don't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't", "shouldn't",
    "mustn't",
}


def _tokens(text: str) -> list[str]:
    """Return lowercase word and number tokens."""
    return re.findall(r"\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b", text.lower())


def _content_tokens(text: str) -> set[str]:
    """Return meaningful tokens, excluding common function words."""
    return {
        token
        for token in _tokens(text)
        if token not in _STOP_WORDS and not token.isdigit()
    }


def _has_negation(text: str) -> bool:
    """Detect common explicit negation words in a sentence."""
    return bool(set(_tokens(text)) & _NEGATION_TOKENS)


def validate_answer(
    answer: str,
    evidence: Sequence[RetrievalResult],
) -> AnswerValidation:
    """Apply conservative lexical checks to answer sentences."""
    if not answer.strip():
        return AnswerValidation(
            valid=False,
            supported_claims=(),
            unsupported_claims=(),
            reason="The answer is empty.",
        )

    if not evidence:
        return AnswerValidation(
            valid=False,
            supported_claims=(),
            unsupported_claims=(answer.strip(),),
            reason="No evidence was provided for validation.",
        )

    evidence_text = " ".join(item.text for item in evidence)

    evidence_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", evidence_text)
        if sentence.strip()
    ]

    evidence_tokens = set(_tokens(evidence_text))
    evidence_content = _content_tokens(evidence_text)

    # Remove source markers before evaluating answer text.
    answer_text = re.sub(
        r"\[Source\s+\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer_text)
        if sentence.strip()
    ]

    supported: list[str] = []
    unsupported: list[str] = []

    for sentence in sentences:
        sentence_tokens = set(_tokens(sentence))
        sentence_content = _content_tokens(sentence)

        unsupported_numbers = {
            token
            for token in sentence_tokens
            if token.isdigit() and token not in evidence_tokens
        }

        if not sentence_content:
            unsupported.append(sentence)
            continue

        overlap = sentence_content & evidence_content
        overlap_ratio = len(overlap) / len(sentence_content)

        # Check whether this answer sentence reverses the negation
        # of a sufficiently similar evidence sentence.
        comparable_answer_tokens = (
            sentence_content - _NEGATION_TOKENS
        )
        contradicts_evidence = False

        if comparable_answer_tokens:
            for evidence_sentence in evidence_sentences:
                evidence_sentence_content = (
                    _content_tokens(evidence_sentence)
                    - _NEGATION_TOKENS
                )

                if not evidence_sentence_content:
                    continue

                comparison_ratio = len(
                    comparable_answer_tokens
                    & evidence_sentence_content
                ) / len(comparable_answer_tokens)

                if (
                    comparison_ratio >= 0.6
                    and _has_negation(sentence)
                    != _has_negation(evidence_sentence)
                ):
                    contradicts_evidence = True
                    break

        if (
            unsupported_numbers
            or overlap_ratio < 0.6
            or contradicts_evidence
        ):
            unsupported.append(sentence)
        else:
            supported.append(sentence)

    valid = bool(supported) and not unsupported

    if valid:
        reason = "All answer sentences passed the lexical checks."
    else:
        reason = (
            "One or more sentences failed the lexical checks. "
            "Review them against the original evidence."
        )

    return AnswerValidation(
        valid=valid,
        supported_claims=tuple(supported),
        unsupported_claims=tuple(unsupported),
        reason=reason,
    )
