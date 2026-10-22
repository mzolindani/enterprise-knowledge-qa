
import pymupdf

from app.embeddings import LocalSentenceTransformerProvider
from app.ingestion.pipeline import DocumentIndexingPipeline
from app.retrieval import (
    BM25Retriever,
    ChromaVectorStore,
    HybridRetrievalService,
)


def create_policy_pdf(path, text):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (50, 50, 550, 750),
            text,
            fontsize=12,
        )
        pdf.save(path)


def test_indexing_and_hybrid_search_end_to_end(tmp_path):
    pdf_path = tmp_path / "safety-policy.pdf"
    create_policy_pdf(
        pdf_path,
        (
            "Workplace safety policy. Employees must report "
            "incidents to their manager. All incidents must be "
            "recorded in the incident reporting system."
        ),
    )

    vector_store = ChromaVectorStore()
    bm25 = BM25Retriever()
    provider = LocalSentenceTransformerProvider()

    pipeline = DocumentIndexingPipeline(
        vector_store=vector_store,
        embedding_provider=provider,
        bm25_retriever=bm25,
    )

    indexed_count = pipeline.index_file(pdf_path)

    assert indexed_count > 0
    assert vector_store.count() == indexed_count
    assert bm25.count() == indexed_count

    hybrid = HybridRetrievalService(
        vector_store=vector_store,
        bm25_retriever=bm25,
        embedding_provider=provider,
    )

    results = hybrid.search(
        "Where must employees report workplace incidents?",
        top_k=3,
    )

    assert results
    assert results[0].document_name == "safety-policy.pdf"
    assert any(
        "incident" in result.text.lower()
        for result in results
    )
    assert all(result.page_number is not None for result in results)


def test_reindexing_removes_obsolete_chunks_from_both_indexes(
    tmp_path,
):
    pdf_path = tmp_path / "company-policy.pdf"

    create_policy_pdf(
        pdf_path,
        (
            "The legacy project code is ORCHID-731. "
            "Employees must follow the old reporting process."
        ),
    )

    vector_store = ChromaVectorStore(
    collection_name="reindex_test_company_policy"
)
    bm25 = BM25Retriever()
    provider = LocalSentenceTransformerProvider()

    pipeline = DocumentIndexingPipeline(
        vector_store=vector_store,
        embedding_provider=provider,
        bm25_retriever=bm25,
    )

    pipeline.index_file(pdf_path)

    old_results = bm25.search("ORCHID-731", top_k=5)
    assert old_results

    create_policy_pdf(
        pdf_path,
        (
            "The updated company policy requires employees "
            "to follow the new reporting process."
        ),
    )

    updated_count = pipeline.index_file(pdf_path)

    print("Updated chunk count:", updated_count)
    print("Total ChromaDB chunks:", vector_store.count())
    print(
        "ChromaDB records:",
        vector_store.collection.get(
            include=["documents", "metadatas"]
        ),
    )

    assert updated_count > 0
    assert vector_store.count() == updated_count
    assert bm25.count() == updated_count
    assert bm25.search("ORCHID-731", top_k=5) == []

