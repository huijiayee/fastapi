from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()


@app.get("/")
def check_root():
    return {"Hello": "World"}


@app.get("/get_embed")
def get_embedding(text):
    # model = SentenceTransformer(
    #     'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = SentenceTransformer(
        'sentence-transformers/paraphrase-MiniLM-L6-v2')
    text = text.replace("\n", " ")
    embeddings = model.encode(text).tolist()
    return embeddings
