import sys
import numpy as np
import json
from openai import OpenAI
from env import OPEN_AI_KEY

import random

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_EMB_MODEL = "text-embedding-3-small"

# 텍스트 파일을 RAG로 생성
# TODO: RAG 문서 선정 후 JSON 파일로 저장
RAG_FILE = "RAG.txt"


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_KEY)


def get_embedding(text, client):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=OPENAI_EMB_MODEL).data[0].embedding
    )


# 코사인 유사도 검사: 근접하는 문장일수록 값이 1에 가까워짐
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


"""
# 가짜 임베딩 코드 생성기: 랜덤으로 코드 생성
def get_embedding(text, client):
    codes = [0] * 1536
    for i in range(1536):
        codes[i] = random.random()
    return codes

def test_emb():
    response = get_embedding(
        "Some students had criticised the selection of Mr Youngkin as that year's speaker, both for his opposition to a racial literacy requirement being considered by VCU as well as for saying that encampments on college campuses should not be allowed."
    )
    print(response, len(response))
    return response
"""


oai = get_openai_client()


# 문서에서 청크 생성: 줄바꿈과 공백 제거 후 각 문단을 청크로 변환
def make_chunk(txt):
    chunk = []
    prgh = ""
    txt = txt.split("\n")
    for l in txt:
        l = l.strip()
        prgh = prgh + " " + l
        if l == "":
            if prgh != " ":
                chunk.append(prgh)
            prgh = ""

    return chunk


# RAG 생성
def add_RAG(RAG_DB, chunk):
    for c in chunk:
        c = " ".join(c.split())
        code = get_embedding(c, oai)
        RAG_DB[c] = code


# input을 RAG에서 검색
def search_RAG(RAG_DB, code, nbest=1):
    score = {}
    for k in RAG_DB:
        score[k] = cosine_similarity(code, RAG_DB[k])
    score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return score[0:nbest]


# RAG를 json 파일로 저장
def save_RAG(RAG_DB, path):
    with open(path, "w") as f:
        f.write(json.dumps(RAG_DB, indent=4))


# json 파일 RAG를 불러옴
def load_RAG(path):
    with open(path, "r") as f:
        RAG = json.load(f)
    return RAG


#######################################

chunk = make_chunk(open(RAG_FILE).read())

# make RAG DB
RAG_DB = {}
## add_RAG(RAG_DB, chunk)
## save_RAG(RAG_DB, "RAG_DB.json")
RAG_DB = load_RAG("RAG_DB.json")

# 예시 문장
input = "SDL abstracts the layers of multimedia input, making it possible to run on multiple OS."
input_code = get_embedding(input, oai)

# search RAG_DB by similarity
best_sentences = search_RAG(RAG_DB, input_code, 1)
print(best_sentences)

### result: best sentence found
print(best_sentences[0][0])
