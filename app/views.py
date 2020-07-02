from app import app
from flask import request, redirect, jsonify
from flask import render_template
from werkzeug.utils import secure_filename
import sys
import os
from pprint import pprint
import base64
import urllib.request
import json
import http.client
from cv2 import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import keras.backend as K
import dlib


app.config["IMAGE_STATIC"] = 'app/static/img/object_detected'
app.config["MODEL"] = 'app/core'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024


@app.route("/")
def index():
    return render_template("public/index.html")


@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>
    """


def allowed_image(filename):
    
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


img_size = 224


def resize_img(im):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    return new_im, ratio, top, left


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    
    if request.method == "POST":
        if request.files:
            img = request.files["image"]
            execution_path = os.getcwd()

            model_path = os.path.join(execution_path, app.config["MODEL"], 'model.h5')
            cat_bounding_box_model = os.path.join(execution_path, app.config["MODEL"], 'bbs_1.h5')
            dog_bounding_box_model = os.path.join(execution_path, app.config["MODEL"], 'dogHeadDetector.dat')
            filename = secure_filename(img.filename)
            uploaded_path = os.path.join(execution_path, app.config["IMAGE_STATIC"], 'input', filename)
            img.save(uploaded_path)
            model = load_model(model_path)

            # dimensions of our images
            img_width, img_height = 150, 150
            saved_img = image.load_img(uploaded_path, target_size=(img_width, img_height))
            x = image.img_to_array(saved_img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict_classes(images, batch_size=10)
            classes = classes[0][0]
            if classes:
                res = 'Dog'
                detector = dlib.cnn_face_detection_model_v1(dog_bounding_box_model)
                dog_img = cv2.imread(uploaded_path)
                dog_img = cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB)
                dets = detector(dog_img, upsample_num_times=1)
                img_result = dog_img.copy()
                for i, d in enumerate(dets):
                    x1, y1 = d.rect.left(), d.rect.top()
                    x2, y2 = d.rect.right(), d.rect.bottom()
                    cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255, 0, 0),
                                  lineType=cv2.LINE_AA)
                cv2.imwrite(uploaded_path, img_result)

            else:
                res = 'Cat'
                bbs_model = load_model(cat_bounding_box_model)
                cat_img = cv2.imread(uploaded_path)
                bounding_boxed_img = cat_img.copy()
                cat_img, ratio, top, left = resize_img(cat_img)
                inputs = (cat_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
                pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))
                ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)
                cv2.rectangle(bounding_boxed_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255),
                              thickness=2)
                cv2.imwrite(uploaded_path, bounding_boxed_img)


            K.clear_session()
            # return jsonify({
            #     'mes': 'ok',
            #     'result': res
            #     }), 201

            img_temp_path = '/static/img/object_detected/input/' + filename
            string_array = [res, img_temp_path]
            return render_template('public/upload_image.html', value=string_array)
        return jsonify({
            'result': 'error',
            'read_text': ''}), 201   

    return render_template("public/upload_image.html")

   
