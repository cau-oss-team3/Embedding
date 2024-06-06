#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
from openai import OpenAI

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_EMB_MODEL = "text-embedding-3-small"
OPEN_AI_KEY = "sk-proj-xxx" ## insert api key here

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_KEY)


def load_embed(path):
    with open(path, "rb") as f:
        embed = pickle.load(f)
    return embed


def get_embedding(text, client):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=OPENAI_EMB_MODEL).data[0].embedding
    )

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_RAG(embed, code, nbest=1):
    score = {}
    for k in embed:
        score[k] = cosine_similarity(code, embed[k])
    score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return score[0:nbest]

def get_best_RAG(query, client: OpenAI):
    code = get_embedding(query, client)
    best_chunk = search_RAG(RAG_DB, code, 3)
    return " ".join(map(lambda it: it[0], best_chunk))


def get_qna_answer(
    question,
    client: OpenAI,
):
    question = question.replace("\n", " ")
    
    ## Question or History 중심
    prompt_for_rag = question + \
                    " Please check the question and find the related informations are in this database."
    RAG_text = get_best_RAG(prompt_for_rag, client)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are a guide specializing in backend with the expertise of backend programmin expert. " + \
                           "Your task is to suggest today's study direction and the next short-term goal to the user, " + \
                           "tailoring your advice to the user’s current knowledge level in backend and desired learning outcomes. " + \
                           "Your message is delivered just before the study session begins. " + \
                           "Ensure your advice is specific, actionable, and motivating, encouraging the user to persist in their studies." + \
                           f"Question: {question}\nPlease provide detailed guidance for the next steps in learning, taking into account the current knowledge and the desired learning outcome." + \
                           "You should not exceed 200 words and please provide the response in Korean."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": RAG_text
            },
            """
            {
                "role": "assistant",
                "content": history_text
            },
            """
        ],
        temperature=0.75,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.75,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('ERROR: %s <embed.pkl>'%os.path.basename(sys.argv[0]))
        sys.exit(2)

    RAG_DB = load_embed(sys.argv[1])

    oai = get_openai_client()
    prompt_for_rag = "Software for Python might be focused on quality, readability, coherence."
    print(get_best_RAG(prompt_for_rag, oai))

