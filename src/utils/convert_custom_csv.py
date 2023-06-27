import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import faiss
import torch
from datasets import Features, Sequence, Value, load_dataset

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)

""" Full credit goes to 
https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/use_own_knowledge_dataset.py """


logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(
    documents: dict,
    ctx_encoder: DPRContextEncoder,
    ctx_tokenizer: DPRContextEncoderTokenizerFast,
) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"],
        documents["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )["input_ids"]
    embeddings = ctx_encoder(
        input_ids.to(device=device), return_dict=True
    ).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def csv_to_dataset(
    csv_path,
    output_dir_path,
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
):

    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage

    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    assert os.path.isfile(csv_path), "Please provide a valid path to a csv file"

    # You can load a Dataset object this way
    dataset = load_dataset(
        "csv",
        data_files=csv_path,
        split="train",
        delimiter="\t",
        column_names=["title", "text"],
    )

    # Then split the documents into passages of 100 words
    dataset = dataset.map(
        split_documents, batched=True, num_proc=processing_args.num_proc
    )

    # And compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    ).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    )
    new_features = Features(
        {
            "text": Value("string"),
            "title": Value("string"),
            "embeddings": Sequence(Value("float32")),
        }
    )  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,
    )

    # And finally save your dataset
    passages_path = os.path.join(output_dir_path, "custom_knowledge_dataset")
    dataset.save_to_disk(passages_path)
    # from datasets import load_from_disk
    # dataset = load_from_disk(passages_path)  # to reload the dataset

    ######################################
    logger.info("Step 2 - Index the dataset")
    ######################################

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(
        index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT
    )
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(
        output_dir_path, "custom_knowledge_dataset_hnsw_index.faiss"
    )
    dataset.get_index("embeddings").save(index_path)

    return dataset


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={
            "help": "The dimension of the embeddings to pass to the HNSW Faiss index."
        },
    )
    m: int = field(
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )
