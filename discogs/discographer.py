import faiss
import pickle
import cv2 as cv
import numpy as np
from tensorflow import keras

from fastapi import FastAPI, File, UploadFile
from starlette.concurrency import run_in_threadpool

app = FastAPI()

release_info = pickle.load(open('../release_info.p', 'rb'))
embedding = keras.models.load_model('../embedding_model')
embedding_vectors = pickle.load(open('../embedding_vectors.p', 'rb'))

index_to_release = dict(enumerate(embedding_vectors))
vectors = np.array([embedding_vectors[_] for _ in embedding_vectors], dtype=np.float32)
index = faiss.IndexFlatIP(vectors.shape[1])
faiss.normalize_L2(vectors)
index.add(vectors)

IMG_SIZE = 224  #######


def preprocess_image(image):
    nparr = np.fromstring(image, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def predict(image):
    x = embedding.predict(np.array([preprocess_image(image)]))
    x = x / np.linalg.norm(x[0])
    y = index.search(x, 1)
    return y[0][0][0], index_to_release[y[1][0][0]]


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    image = await file.read()
    result = await run_in_threadpool(predict, image)
    id = result[1]
    proximity = result[0]
    if id in release_info:  ################
        artist = release_info[id][0]
        title = release_info[id][1]
    else:
        artist = title = None
    return {
        "id": id,
        "proximity": float(proximity),
        "artist": artist,
        "title": title
    }
