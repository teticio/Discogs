import os
import faiss
import pickle
import random
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import Model, layers, metrics, optimizers, callbacks

train_val_split = 0.8
releases = [
    int(_[6:-4]) for _ in os.listdir('../thumbs') if _[6:-4].isnumeric()
]

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.01),
        preprocessing.RandomTranslation(height_factor=(-.05, 0.05),
                                        width_factor=(-0.05, 0.05)),
        preprocessing.RandomZoom(height_factor=(-0.05, 0.05),
                                 width_factor=(-0.05, 0.05)),
        preprocessing.RandomContrast(factor=0.3),
    ],
    name="img_augmentation",
)


def preprocess_image(release, augment=True):
    if type(release) == int:
        release = f'../thumbs/thumb_{release}.jpg'
    img = cv.imread(release, cv.IMREAD_COLOR)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if augment:
        img = img_augmentation(tf.expand_dims(img, axis=0))
        img = img[0].numpy().astype("uint8")
    return img


# https://keras.io/examples/vision/siamese_network/


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights))

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def generator(batch_size, validation=False):
    index = 0
    random.shuffle(releases)
    indices = releases[
        int(train_val_split *
            len(releases)):] if validation else releases[:int(train_val_split *
                                                              len(releases))]

    image_vectors = pickle.load(open('../image_vectors.p', 'rb'))
    vectors = np.array([image_vectors[str(_)] for _ in indices],
                       dtype=np.float32)
    del image_vectors

    quantiser = faiss.IndexFlatL2(vectors.shape[1])
    similar = faiss.IndexIVFFlat(quantiser, vectors.shape[1], 100,
                                 faiss.METRIC_L2)
    similar.train(vectors)
    #similar = quantiser
    similar.add(vectors)
    vector = np.array([vectors[0]])
    D = I = None

    while True:
        anchors = []
        positives = []
        negatives = []
        for _ in range(batch_size):
            # choose one of 100-200 most similar as negative
            vector = np.array([vectors[index]])
            D, I = similar.search(vector, 200, D=D, I=I)
            while True:
                not_index = I[0][random.randint(100, 200 - 1)]
                # not_index = random.randint(0, len(indices) - 1)
                if not_index != index:
                    break
            not_index = 0
            anchors += [
                np.array([preprocess_image(indices[index], augment=True)])
            ]
            positives += [
                np.array([preprocess_image(indices[index], augment=False)])
            ]
            negatives += [
                np.array([preprocess_image(indices[not_index], augment=False)])
            ]
            index = index + 1
            if index == len(indices):
                index = 0
                random.shuffle(indices)
        yield [anchors, positives, negatives]


def dump_embeddings():
    embedding.save('../embedding_model')
    print(f'Dumping embeddings')
    embedding_vectors = {}
    batch_size = 1024
    for i in tqdm(range(0, len(releases), batch_size)):
        batch = []
        for j in range(0, min(batch_size, len(releases) - i)):
            batch += [preprocess_image(releases[i + j], augment=False)]
        vectors = embedding.predict(np.array(batch))
        for j in range(0, min(batch_size, len(releases) - i)):
            embedding_vectors[releases[i + j]] = vectors[j]
    pickle.dump(embedding_vectors, open('../embedding_vectors.p', 'wb'))
    del embedding_vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dump_embeddings',
                        action='store_true',
                        help='Dump embeddings')
    args = parser.parse_args()

    base_cnn = tf.keras.applications.EfficientNetB0(include_top=False,
                                                    weights="imagenet",
                                                    pooling="avg")

    layer = layers.Flatten()(base_cnn.output)
    layer = layers.Dense(1024)(layer)
    layer = layers.LeakyReLU(0.2)(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.Dropout(rate=0.5)(layer)
    layer = layers.Dense(512)(layer)
    output = layer

    embedding = Model(base_cnn.input, output, name="Embedding")
    IMG_SIZE = 224  ####### embedding.get_input_shape_at(0)[1]

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "top_conv":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name="anchor", shape=(IMG_SIZE, IMG_SIZE, 3))
    positive_input = layers.Input(name="positive",
                                  shape=(IMG_SIZE, IMG_SIZE, 3))
    negative_input = layers.Input(name="negative",
                                  shape=(IMG_SIZE, IMG_SIZE, 3))

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances)

    siamese_model = SiameseModel(siamese_network, margin=0.1)  ##########
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))

    # dummy call before we can load weights
    siamese_model([np.zeros((IMG_SIZE, IMG_SIZE, 3))] * 3)
    if os.path.isfile('../discographer.h5'):
        print('Loading weights')
        siamese_model.load_weights('../discographer.h5')

    if args.dump_embeddings:
        dump_embeddings()

    batch_size = 1024
    siamese_model.fit(
        generator(batch_size),
        epochs=500,
        steps_per_epoch=train_val_split * len(releases) / batch_size / 10,
        validation_data=generator(batch_size, validation=True),
        validation_steps=(1 - train_val_split) * len(releases) / batch_size /
        10,
        callbacks=[
            callbacks.ModelCheckpoint('../discographer.h5',
                                      monitor='val_loss',
                                      mode='min',
                                      save_best_only=True),
        ])
