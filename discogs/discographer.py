import pickle
import cv2 as cv
import numpy as np
from tensorflow import keras

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

app = FastAPI()

embedding = keras.models.load_model('../embedding_model')
embedding_vectors = pickle.load(open('../embedding_vectors.p', 'rb'))
release_info = pickle.load(open('../release_info.p', 'rb'))

IMG_SIZE = 224  #######


def preprocess_image(image):
    nparr = np.fromstring(image, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def predict(image):
    x = embedding.predict(np.array([preprocess_image(image)]))
    proximities = sorted([
        (np.dot(embedding_vectors[_], x[0]) /
         np.linalg.norm(embedding_vectors[_]) / np.linalg.norm(x[0]), _)
        for _ in embedding_vectors
    ], reverse=True)
    return proximities[0]


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
