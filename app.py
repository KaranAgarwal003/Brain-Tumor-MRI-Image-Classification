import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import render_template
app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = os.path.join('model', 'brain_mri_model.h5')
model = load_model(MODEL_PATH)

# Class labels
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_badge_class(label):
    label = label.lower()
    if 'glioma' in label:
        return 'badge-glioma'
    elif 'meningioma' in label:
        return 'badge-meningioma'
    elif 'pituitary' in label:
        return 'badge-pituitary'
    else:
        return 'badge-no'



@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded", badge_class='badge-no')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected", badge_class='badge-no')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            # Preprocess image
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            result = class_labels[predicted_class_index]
            badge_class = get_badge_class(result)

            return render_template('result.html', prediction=result, badge_class=badge_class, image_filename=filename)

        else:
            return render_template('index.html', prediction="Invalid file type. Upload .jpg/.jpeg/.png", badge_class='badge-no')

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", badge_class='badge-no')

if __name__ == '__main__':
    app.run(debug=True)
