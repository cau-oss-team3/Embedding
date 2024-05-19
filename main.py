import sys
import numpy as np
from openai import OpenAI
from env import OPEN_AI_KEY

import random

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_EMB_MODEL = "text-embedding-3-small"

RAG_FILE = "RAG.txt"


oai = OpenAI(api_key=OPEN_AI_KEY)


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_KEY)  # ###에 api key 입력


def get_embedding(text, client):
    codes = [0] * 1536
    for i in range(1536):
        codes[i] = random.random()
    return codes


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


"""
def get_embedding(text, client):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=OPENAI_EMB_MODEL).data[0].embedding
    )

def test_emb():
    response = get_embedding(
        "Some students had criticised the selection of Mr Youngkin as that year's speaker, both for his opposition to a racial literacy requirement being considered by VCU as well as for saying that encampments on college campuses should not be allowed."
    )
    print(response, len(response))
    return response
"""


def make_chunk(txt):
    txt = txt.replace("\r", "").replace("\n", " ")  # .replace("\f", " ")
    chunk = txt.splitlines()

    return chunk


def search_RAG(RAG_dict, code, nbest=1):
    score = {}
    for k in RAG_dict:
        score[k] = cosine_similarity(code, RAG_dict[k])
    score = sorted(score.items(), key=lambda x: x[1])
    return score[0:nbest]


#######################################

chunks = make_chunk(open(RAG_FILE).read())

n = 0

# make RAG DB
RAG_dict = {}
for c in chunks:
    c = " ".join(c.split())
    print(n, ">", c)
    n += 1

    code = get_embedding(c, oai)
    RAG_dict[c] = code


input = "blah blah"
input_code = get_embedding(input, oai)

# search RAG_dict by similarity
best_sentences = search_RAG(RAG_dict, input_code, 1)
print(best_sentences)

### result: best sentence found
print(best_sentences[0][0])
