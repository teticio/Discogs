BASELINE = False

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import faiss
import pickle
import uvicorn
import cv2 as cv
import numpy as np
from tensorflow import keras

from fastapi import FastAPI, File, UploadFile
from starlette.concurrency import run_in_threadpool

app = FastAPI()

release_info = pickle.load(open('../release_info.p', 'rb'))
if not BASELINE:
    embedding = keras.models.load_model('../embedding_model')
    embedding_vectors = pickle.load(open('../embedding_vectors.p', 'rb'))
else:
    embedding = keras.models.load_model('../image_model')
    embedding_vectors = pickle.load(open('../image_vectors.p', 'rb'))

index_to_release = dict(enumerate(embedding_vectors))
vectors = np.array([embedding_vectors[_] for _ in embedding_vectors],
                   dtype=np.float32)
if not BASELINE:
    index = faiss.IndexFlatIP(vectors.shape[1])
    faiss.normalize_L2(vectors)
else:
    index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

IMG_SIZE = 224  #######


def preprocess_image(image):
    nparr = np.fromstring(image, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def predict(image, topk=1):
    x = embedding.predict(np.array([preprocess_image(image)]))
    if not BASELINE:
        x = x / np.linalg.norm(x[0])
    y = index.search(x, topk)
    return [float(_) for _ in y[0][0][:topk]
            ], [index_to_release[_] for _ in y[1][0][:topk]]


@app.post("/uploadfile/")
async def create_upload_file(topk: int, file: UploadFile = File(...)):
    image = await file.read()
    result = await run_in_threadpool(predict, image, topk)
    ids = result[1]
    proximities = result[0]
    artists = [release_info[str(_)][0] for _ in ids]
    titles = [release_info[str(_)][1] for _ in ids]
    return {
        "ids": ids,
        "proximities": proximities,
        "artists": artists,
        "titles": titles
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)