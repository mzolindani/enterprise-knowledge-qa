
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from app.embeddings import LocalSentenceTransformerProvider
from app.evaluation.metrics import (
    hit_rate_at_k,
    reciprocal_rank_at_k,
)
from app.ingestion.pipeline import DocumentIndexingPipeline
from app.retrieval import (
    BM25Retriever,
    ChromaVectorStore,
    HybridRetrievalService,
)


def create_policy_pdf(path: Path, title: str, text: str) -> None:
    """Create a small PDF for the retrieval benchmark."""
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (50, 50, 550, 750),
            f"{title}\n\n{text}",
            fontsize=12,
        )
        pdf.save(path)


def main() -> None:
    """Run a small end-to-end retrieval benchmark."""
    with TemporaryDirectory(prefix="knowledge_qa_eval_") as temp_dir:
        root = Path(temp_dir)

        documents = {
            "incident_reporting.pdf": (
                "Incident Reporting Policy",
                "Employees must report workplace safety incidents "
                "to their line manager immediately.",
            ),
            "incident_records.pdf": (
                "Incident Records Policy",
                "Managers must record every reported incident in "
                "the company safety register.",
            ),
            "serious_incidents.pdf": (
                "Serious Incident Escalation Policy",
                "Serious workplace safety incidents must also be "
                "escalated to the health and safety officer.",
            ),
        }

        for filename, (title, text) in documents.items():
            create_policy_pdf(root / filename, title, text)

        vector_store = ChromaVectorStore(
            collection_name="knowledge_qa_eval"
        )
        bm25 = BM25Retriever()
        provider = LocalSentenceTransformerProvider()

        pipeline = DocumentIndexingPipeline(
            vector_store=vector_store,
            embedding_provider=provider,
            bm25_retriever=bm25,
        )

        for filename in documents:
            indexed = pipeline.index_file(root / filename)
            if indexed == 0:
                raise RuntimeError(
                    f"No chunks indexed for {filename}"
                )

        retriever = HybridRetrievalService(
            vector_store=vector_store,
            bm25_retriever=bm25,
            embedding_provider=provider,
        )

        cases = [
            {
                "question": (
                    "Who must employees report workplace safety "
                    "incidents to?"
                ),
                "expected_document": "incident_reporting.pdf",
            },
            {
                "question": (
                    "Where must managers record reported incidents?"
                ),
                "expected_document": "incident_records.pdf",
            },
            {
                "question": (
                    "Who must serious safety incidents be escalated to?"
                ),
                "expected_document": "serious_incidents.pdf",
            },
        ]

        hits_at_1 = 0
        hits_at_3 = 0
        reciprocal_ranks = []
        k = 3

        print("\nRetrieval Evaluation")
        print("=" * 60)

        for case in cases:
            results = retriever.search(case["question"], top_k=k)
            retrieved_names = [
                result.document_name for result in results
            ]
            relevant = {case["expected_document"]}

            hit_at_1 = hit_rate_at_k(
                retrieved_names, relevant, k=1
            )
            hit_at_3 = hit_rate_at_k(
                retrieved_names, relevant, k=3
            )
            reciprocal_rank = reciprocal_rank_at_k(
                retrieved_names, relevant, k=3
            )

            hits_at_1 += int(hit_at_1)
            hits_at_3 += int(hit_at_3)
            reciprocal_ranks.append(reciprocal_rank)

            print(f"\nQuestion: {case['question']}")
            print(f"Expected: {case['expected_document']}")
            print(f"Retrieved: {retrieved_names}")
            print(f"Hit@1: {int(hit_at_1)}")
            print(f"Hit@3: {int(hit_at_3)}")
            print(f"Reciprocal rank@3: {reciprocal_rank:.3f}")

        total = len(cases)

        print("\nSummary")
        print("-" * 60)
        print(f"Questions evaluated: {total}")
        print(f"Hit Rate@1: {hits_at_1 / total:.2%}")
        print(f"Hit Rate@3: {hits_at_3 / total:.2%}")
        print(
            f"MRR@3: {sum(reciprocal_ranks) / total:.3f}"
        )
       


if __name__ == "__main__":
    main()
