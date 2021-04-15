import numpy as np
import base64
import os
import cv2
import datetime


def decode_image(image):
    header, encoded = image.split(",", 1)
    binary_image = base64.b64decode(encoded)
    return binary_image


def save_image(image):
    image_name = "photo.jpeg"
    binary_image = decode_image(image)

    with open(os.path.join('./', image_name), 'wb') as f:
        f.write(binary_image)

    return image_name


def preprocess_img(path, detector):
    img = cv2.imread(path, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)
    x1, y1, width, height = result[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img_rgb[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0
    return face


def compute_dist(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    return dist


def get_date_time():
    today = datetime.datetime.now()
    time = str(today.time()).split('.')[0]
    day = today.day
    month = today.month
    year = today.year
    date = str(day) + '/' + str(month) + '/' + str(year)
    return time, date


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 10, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width - width*0.3:
            return scale / 10, textSize[0]
    return 1


def draw_rectangle(rect, name):
    time, date = get_date_time()
    image_name = "photo.jpeg"
    save_image = "save_img.jpg"
    image_path = os.path.join('./', image_name)
    text = name + ' - ' + time
    img1 = cv2.imread(image_path, 1)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    new_img = cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    scale, size = get_optimal_font_scale(text, img.shape[0])
    new_img = cv2.rectangle(new_img, (rect[0], rect[1] + rect[3] + 4), (rect[0] + size[0], rect[1] + rect[3] + 4 + size[1]),
                            (0, 255, 0), -1)
    new_img = cv2.putText(new_img, text, (rect[0], rect[1] + rect[3] + 4 + size[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    save_path = os.path.join('./static/', save_image)
    cv2.imwrite(save_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
