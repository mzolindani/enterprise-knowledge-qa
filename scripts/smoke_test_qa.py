
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from app.embeddings import LocalSentenceTransformerProvider
from app.generation import KnowledgeQAService, OllamaAnswerProvider
from app.ingestion import ingest_file
from app.ingestion.pipeline import DocumentIndexingPipeline
from app.retrieval import (
    BM25Retriever,
    ChromaVectorStore,
    HybridRetrievalService,
)


def create_test_pdf(file_path: Path) -> None:
    """Create a small policy document for the smoke test."""
    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Company Incident Reporting Policy",
        fontsize=16,
    )

    page.insert_textbox(
        (72, 110, 520, 700),
        (
            "Employees must report workplace safety incidents "
            "to their line manager immediately. The manager must "
            "record the incident in the company safety register. "
            "Serious incidents must also be escalated to the "
            "health and safety officer."
        ),
        fontsize=11,
    )


    document.save(file_path)
    document.close()


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "incident_policy.pdf"

        create_test_pdf(pdf_path)

        document = ingest_file(pdf_path)
        print("\nExtracted PDF text:")
        print(document.full_text)

        vector_store = ChromaVectorStore(
            collection_name="qa_smoke_test",
        )
        bm25_retriever = BM25Retriever()
        embedding_provider = LocalSentenceTransformerProvider()

        indexing_pipeline = DocumentIndexingPipeline(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            bm25_retriever=bm25_retriever,
        )

        indexed_count = indexing_pipeline.index_file(pdf_path)
        print(f"Indexed chunks: {indexed_count}")

        retrieval_service = HybridRetrievalService(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            embedding_provider=embedding_provider,
        )

        answer_provider = OllamaAnswerProvider(
            model="qwen2.5:3b",
        )

        qa_service = KnowledgeQAService(
            retriever=retrieval_service,
            answer_provider=answer_provider,
        )

        question = (
            "Who must employees report workplace safety incidents to, "
            "and what must happen afterward?"
        )

        
        # Temporary diagnostics: inspect retrieval and raw generation.
        evidence = retrieval_service.search(question, top_k=5)
        generated = answer_provider.generate(question, evidence)

        from app.generation.validator import validate_answer

        validation = validate_answer(generated.answer, evidence)

        print("\nRetrieved evidence:")
        for item in evidence:
            print(f"- {item.text}")

        print("\nRaw model answer:")
        print(generated.answer)

        print("\nValidation result:")
        print(f"Valid: {validation.valid}")
        print(f"Supported claims: {validation.supported_claims}")
        print(f"Unsupported claims: {validation.unsupported_claims}")
        print(f"Reason: {validation.reason}")

        # Run the normal QA flow as before.
        result = qa_service.ask(question)


        print("\nQuestion:")
        print(result.question)

        print("\nAnswer:")
        print(result.answer)

        print(f"\nModel: {result.model}")
        print(f"Grounded flag: {result.grounded}")

        print("\nCitations:")
        for citation in result.citations:
            print(f"- Document: {citation.document_name}")
            print(f"  Page: {citation.page_number}")
            print(f"  Chunk: {citation.chunk_id}")
            print(f"  Excerpt: {citation.excerpt}")


if __name__ == "__main__":
    main()
