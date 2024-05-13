from typing import Union


from fastapi import FastAPI
from fastapi import APIRouter, Depends
from openai import OpenAI

router = APIRouter(prefix="/prompt", tags=["prompt"])

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_EMB_MODEL = "text-embedding-3-small"

app = FastAPI()


def get_openai_client() -> OpenAI:
    return OpenAI(api_key="sk-proj-###")  # ###에 api key 입력


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


def get_embedding(text, client: OpenAI = Depends(get_openai_client)):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=OPENAI_EMB_MODEL).data[0].embedding
    )


"""
1536개의 floating point 코드를 VectorDB에 담아야 함
"""


@app.get("/test-emb/")
def test_emb():
    response = get_embedding(
        "Some students had criticised the selection of Mr Youngkin as that year's speaker, both for his opposition to a racial literacy requirement being considered by VCU as well as for saying that encampments on college campuses should not be allowed."
    )
    print(response, len(response))
    return response
