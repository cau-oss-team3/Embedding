#!/usr/bin/env python3

import sys
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


def simple_chunk(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        length_function = len,
        is_separator_regex = False
    )
    return text_splitter.split_documents(doc)


def semantic_chunk(doc):
    embeddings = HuggingFaceEmbeddings(
        #model_name="jhgan/ko-sroberta-multitask",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': False},
        model_kwargs={'device': 'cuda:0'}, # use NVIDIA GPU
        #model_kwargs={'device': 'cpu'}, # use CPU
    )
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type = "percentile", breakpoint_threshold_amount = args.threshold
        #breakpoint_threshold_type = "standard_deviation", breakpoint_threshold_amount = 1.25
        #breakpoint_threshold_type = "standard_deviation", breakpoint_threshold_amount = 0.6
    )
    return text_splitter.split_documents(doc)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument('file', help='PDF file')
    cli.add_argument('-m', '--mode', default='semantic',
        help='chunking mode (semantic/simple)')
    cli.add_argument('-t', '--threshold', default=80,
        help='breakpoint threshold of percentile type')
    cli.add_argument('-s', '--chunk_size', default=1000,
        help='size of chunk for text splitter')
    cli.add_argument('-o', '--chunk_overlap', default=0,
        help='overlap of chunk for text splitter')
    args = cli.parse_args()

    loader = PyPDFLoader(args.file)
    doc = loader.load()
    print('### Document Length:', len(doc), file=sys.stderr)

    if args.mode == 'semantic':
        chunks = semantic_chunk(doc)
    else:
        chunks = simple_chunk(doc)

    print('### Chunk Length:', len(chunks), file=sys.stderr)
    idx = 0
    for chunk in chunks:
        idx += 1
        print(f'### CHUNK {idx}:')
        print(chunk.page_content.replace('\n', ' '))

