import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras_facenet import model as resnet
from mtcnn.mtcnn import MTCNN
from utils import utils
from embeddings_dao import EmbeddingsDao
import cv2
import numpy as np


class FaceRecognition:
    def __init__(self):
        self.threshold = 1.00
        self.model_path = 'models/model-20170512-110547.ckpt-250000'
        self.embedding_dao = EmbeddingsDao()
        self.detector = MTCNN()
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.compat.v1.Session()
            self.x = tf.compat.v1.placeholder('float', [None, 160, 160, 3])
            self.embeddings = tf.nn.l2_normalize(resnet.inference(self.x, 0.6, phase_train=False)[0], 1, 1e-10)
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, self.model_path)

    def get_embeddings(self, path):
        # face_img = utils.preprocess_img(path)
        detector = MTCNN()
        img = cv2.imread(path, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(img_rgb)
        x1, y1, width, height = result[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_rgb[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)
        face_img = face / 255.0
        embeddings = self.sess.run(self.embeddings, feed_dict={self.x: face_img})
        return embeddings[0], result[0]['box']

    def check_embeddings(self, name, path):
        embedding, rect = self.get_embeddings(path)
        embedding_db = self.embedding_dao.get_embedding(name.lower())
        dist = utils.compute_dist(embedding, embedding_db)
        if dist < self.threshold:
            utils.draw_rectangle(rect, name)
            response = "Same"
        else:
            response = "Different"

        return response

    def save_embeddings(self, name, path):
        embedding, rect = self.get_embeddings(path)
        self.embedding_dao.save_embedding(name.lower(), embedding)
