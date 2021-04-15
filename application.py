from utils import utils

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from face_recognition import FaceRecognition

application = Flask(__name__)
CORS(application)
face_recognition = FaceRecognition()


@application.route('/', methods=['GET'])
@cross_origin()
def home_page():
    return render_template('mark_attendance.html')


@application.route('/_photo_cap')
@cross_origin()
def photo_cap():
    photo_base64 = request.args.get('photo_cap')
    name = request.args.get('name')
    is_add = request.args.get('isAdd')
    path = utils.save_image(photo_base64)
    if is_add == 'false':
        response = face_recognition.check_embeddings(name, path)
        if response == 'Same':
            return render_template('marked_attendance.html')
        else:
            return render_template('not_found.html')
    else:
        face_recognition.save_embeddings(name=name, path=path)
        response = 'Added employee ' + name
    return render_template('added.html')


@application.route('/add_new')
@cross_origin()
def add_new():
    return render_template('add_new.html')


if __name__ == '__main__':
    application.run(port=8000)
