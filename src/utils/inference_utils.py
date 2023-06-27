from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import os


def get_tokenizer():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    return tokenizer


def get_retriever(dataset=None):

    if dataset:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq", indexed_dataset=dataset
        )

    else:
        dataset_path = "./custom_dataset/my_knowledge_dataset"
        index_path = "./custom_dataset/my_knowledge_dataset_hnsw_index.faiss"
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq",
            index_name="custom",
            passages_path=dataset_path,
            index_path=index_path,
        )

    return retriever


def get_model(retriever):
    model = RagSequenceForGeneration.from_pretrained(
        "facebook/rag-sequence-nq", retriever=retriever
    )

    return model
