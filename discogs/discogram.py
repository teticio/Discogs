BASELINE = False

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import faiss
import pickle
import cv2 as cv
import numpy as np
from tensorflow import keras

import logging
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

token = open('credentials', 'rt').read()
updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher

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
    if image.shape[0] > image.shape[1]:
        image = image[(image.shape[0] - image.shape[1]) //
                      2:(image.shape[0] + image.shape[1]) // 2, :, :]
    else:
        image = image[:, (image.shape[1] - image.shape[0]) //
                      2:(image.shape[1] + image.shape[0]) // 2, :]
    image = cv.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def predict(image, topk=1):
    x = embedding.predict(np.array([preprocess_image(image)]))
    if not BASELINE:
        x = x / np.linalg.norm(x[0])
    y = index.search(x, topk)
    return y


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Send me a video or photo")


def handle_video(update, context):
    file = updater.bot.get_file(update.message.video.file_id)
    filename = update.message.video.file_name
    file.download(filename)
    try:
        proximities = np.zeros((len(embedding_vectors), ))
        video = cv.VideoCapture(filename)
        while True:
            for skip in range(5):
                flag, frame = video.read()
            if not flag:
                break
            y = predict(frame, 1000)  # only consider top 1000
            for _ in range(len(y[0][0])):
                proximities[y[1][0][_]] += y[0][0][_]
        id = index_to_release[np.argmax(proximities)]
        artist = release_info[str(id)][0]
        title = release_info[str(id)][1]
        url = release_info[str(id)][2]
        logger.info(f'{artist} - {title}')
        context.bot.send_message(chat_id=update.effective_chat.id, text=url)
    except Exception as e:
        # os.remove(filename)
        raise (e)
    # os.remove(filename)


def handle_photo(update, context):
    filename = 'photo.jpg'
    proximities = np.zeros((len(embedding_vectors), ))
    for photo in update.message.photo:
        try:
            file = updater.bot.get_file(photo.file_id)
            file.download(filename)
            frame = cv.imread(filename, cv.IMREAD_COLOR)
            y = predict(frame, 1000)  # only consider top 1000
            for _ in range(len(y[0][0])):
                proximities[y[1][0][_]] += y[0][0][_]
        except Exception as e:
            # os.remove(filename)
            raise (e)
        # os.remove(filename)
    id = index_to_release[np.argmax(proximities)]
    artist = release_info[str(id)][0]
    title = release_info[str(id)][1]
    url = release_info[str(id)][2]
    logger.info(f'{artist} - {title}')
    context.bot.send_message(chat_id=update.effective_chat.id, text=url)


start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)
video_handler = MessageHandler(Filters.video & (~Filters.command),
                               handle_video)
dispatcher.add_handler(video_handler)
photo_handler = MessageHandler(Filters.photo & (~Filters.command),
                               handle_photo)
dispatcher.add_handler(photo_handler)

if __name__ == "__main__":
    updater.start_polling()
    updater.idle()
    updater.stop()
