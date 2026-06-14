import numpy as np
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from flask_cors import CORS
from huggingface_hub import hf_hub_download

#  Configuration 
threshold   = 0.30
image_size  = 224
class_labels = ['meningioma', 'glioma', 'pituitary tumor', 'noTumor']
#MODEL_PATH  = os.environ.get('MODEL_PATH', 'AT_RESu_net_all_STD_0.0927_th=0.38.hdf5')
FRONTEND_PORT = os.environ.get('FRONTEND_PORT', '3000')


MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    hf_hub_download(
        repo_id="VaradPatil8815/wellscanhealthcare-brain-tumor",
        filename="AT_RESu_net_all_STD_0.0927_th=0.38.hdf5"
    )
)

#  Custom metrics 
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= threshold, tf.float32)
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / union

def dice_coefficient(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

custom_objects = {'mean_iou': mean_iou, 'dice_coefficient': dice_coefficient}

#  Model loading 
model = load_model(MODEL_PATH, custom_objects=custom_objects)

#  Flask app 
basepath      = os.path.dirname(__file__)
app           = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join(basepath, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app, resources={r"/*": {"origins": f"http://localhost:{FRONTEND_PORT}"}}, supports_credentials=True)


def preprocessed_img(path):
    v = cv2.imread(path)
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x1, y1, w, h = cv2.boundingRect(contours)
    v = cv2.resize(v[y1:y1+h, x1:x1+w], (image_size, image_size))
    plt.imsave(path, v, cmap='gray')
    v = (v - v.mean()) / v.std()
    v = np.expand_dims(v, axis=-1)
    v = np.expand_dims(v, axis=0)
    return v

def save_segmented_image(segmented_array, filename):
    segmented_array = (segmented_array >= threshold).astype(np.uint8).squeeze()
    segmented_path  = os.path.join(UPLOAD_FOLDER, 'segmented_' + filename)
    plt.imsave(segmented_path, segmented_array, cmap='gray')
    return segmented_array

def tumor_location(v_path, mask, filename):
    labeled_mask, num_features = label(mask)
    if num_features > 0:
        sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
        largest = np.argmax(sizes) + 1
        coords  = np.argwhere(labeled_mask == largest)
        min_y, min_x = np.min(coords, axis=0)
        max_y, max_x = np.max(coords, axis=0)
    else:
        min_y = min_x = max_y = max_x = 0
    v    = cv2.imread(v_path)
    v    = cv2.rectangle(v, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
    path = os.path.join(UPLOAD_FOLDER, 'bbox_' + filename)
    plt.imsave(path, v)

#  Routes 
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file provided'}), 400

    filename  = secure_filename(f.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(file_path)

    img = preprocessed_img(file_path)
    Y_pred1, Y_pred2 = model.predict(img)
    y_pred    = np.argmax(Y_pred1, axis=1)
    rounded   = np.round(Y_pred1 * 100, decimals=4)
    predicted = class_labels[y_pred[0]] + ' ' + str(rounded[0][y_pred[0]]) + '%'

    mask = save_segmented_image(Y_pred2, filename)
    tumor_location(file_path, mask, filename)

    return jsonify({
        'prediction':    predicted,
        'original_img':  url_for('uploaded_file', filename='bbox_'      + filename),
        'segmented_img': url_for('uploaded_file', filename='segmented_' + filename),
    })

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
