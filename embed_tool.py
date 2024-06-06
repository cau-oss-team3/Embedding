#!/usr/bin/env python3

import os
import sys
import pickle
from openai import OpenAI
# from env import OPEN_AI_KEY

OPENAI_EMB_MODEL = "text-embedding-3-small"
OPEN_AI_KEY = "sk-proj-xxx" ## insert api key here

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_KEY)

def get_embedding(text, client):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=OPENAI_EMB_MODEL).data[0].embedding
    )

def load_chunks(path):
    chunks = []
    file = open(path, "r")
    for line in file:
        if line.startswith("### CHUNK "): continue
        line = " ".join(line.strip().split())
        chunks.append(line.strip())
    print("### Total Chunks:", len(chunks), file=sys.stderr)
    return chunks

def add_embed(embed, text):
    oai = get_openai_client()
    text = " ".join(text.split())
    code = get_embedding(text, oai)
    embed[text] = code

def gen_embed(chunks):
    embed = {}
    cnt = 0
    for text in chunks:
        add_embed(embed, text)
        print("\r%d / %d" % (cnt, len(chunks)), end="")
        # print(f"\r{cnt} / {len(chunks)}", end="")
        cnt += 1
    print(" Done...")
    return embed

def save_embed(embed, path):
    with open(path, "wb") as f:
        pickle.dump(embed, f)

def load_embed(path):
    with open(path, "rb") as f:
        embed = pickle.load(f)
    return embed

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_RAG(embed, code, nbest=1):
    score = {}
    for k in embed:
        score[k] = cosine_similarity(code, embed[k])
    score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return score[0:nbest]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('ERROR: %s <chunks.txt> <embed.pkl>'%os.path.basename(sys.argv[0]))
        sys.exit(2)

    chunks = load_chunks(sys.argv[1])
    embed = gen_embed(chunks)
    save_embed(embed, sys.argv[2])

