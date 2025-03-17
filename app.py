import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To suppress unnecessary TensorFlow warnings

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from threading import Thread
import keras.initializers
import tensorflow as tf
import numpy as np



app = Flask(__name__, template_folder='templates')

#run_with_ngrok(app)

# Custom Initializer
def variance_scaling_fixed(**kwargs):
    kwargs.pop('dtype', None)  # Remove dtype argument
    return keras.initializers.VarianceScaling(**kwargs)

tf.keras.utils.get_custom_objects().update({'VarianceScaling': variance_scaling_fixed})

custom_objects = {
    'Zeros': keras.initializers.Zeros
}

# Load Trained Model
MODEL_PATH = "models/skin_disease_model.h5"
#MODEL_PATH = "models/model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects, safe_mode=False)

# Disease Class Info
CLASS_INFO = {
    'nv': {'name': 'Melanocytic Nevi', 'description': 'Benign neoplasms of melanocytes with various dermatoscopic appearances.',
           'causes': 'Genetic predisposition, UV exposure.', 'treatment': 'No treatment required; removal if necessary.',
           'diagnosis': 'Clinical examination, dermatoscopy, biopsy if atypical.'},

    'mel': {'name': 'Melanoma', 'description': 'A malignant tumor derived from melanocytes that can be invasive.',
            'causes': 'UV exposure, genetic mutations.', 'treatment': 'Surgical excision, immunotherapy.',
            'diagnosis': 'Dermatoscopy, biopsy, histopathological examination.'},

    'bkl': {'name': 'Benign Keratosis', 'description': 'Includes seborrheic keratosis, solar lentigo, and lichen-planus-like keratoses.',
            'causes': 'Aging, sun exposure.', 'treatment': 'Cryotherapy, laser therapy, topical treatments.',
            'diagnosis': 'Clinical examination, dermatoscopy, biopsy if needed.'},

    'bcc': {'name': 'Basal Cell Carcinoma', 'description': 'A common epithelial skin cancer that rarely metastasizes but can be locally destructive.',
            'causes': 'Chronic sun exposure, genetic predisposition.', 'treatment': 'Surgical excision, Mohs surgery.',
            'diagnosis': 'Dermatoscopy, biopsy, histopathological confirmation.'},

    'akiec': {'name': 'Actinic Keratosis & Intraepithelial Carcinoma', 'description': 'Non-invasive precursors of squamous cell carcinoma that may progress if untreated.',
              'causes': 'Chronic sun exposure, HPV infection.', 'treatment': 'Cryotherapy, photodynamic therapy.',
              'diagnosis': 'Clinical examination, dermatoscopy, biopsy if needed.'},

    'vasc': {'name': 'Vascular Lesions', 'description': 'Includes cherry angiomas, angiokeratomas, pyogenic granulomas, and hemorrhages.',
             'causes': 'Aging, hormonal changes, trauma.', 'treatment': 'Laser therapy, electrosurgery, observation.',
             'diagnosis': 'Clinical examination, dermatoscopy, biopsy if uncertain.'},

    'df': {'name': 'Dermatofibroma', 'description': 'A benign skin lesion resulting from minor trauma or an inflammatory reaction.',
           'causes': 'Skin injury, inflammatory response.', 'treatment': 'No treatment needed; removal if symptomatic.',
           'diagnosis': 'Clinical examination, dermatoscopy, biopsy if atypical.'}
}


# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess Image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # ✅ Improved Preprocessing for Better Prediction
    img_array = tf.image.adjust_brightness(img_array, delta=0.1)   # Brightness adjustment
    img_array = tf.image.adjust_contrast(img_array, 2)            # Contrast adjustment
    img_array = tf.image.random_flip_left_right(img_array)        # Flip image horizontally
    img_array = tf.image.random_flip_up_down(img_array)           # Flip image vertically

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.route('/',  methods=['GET','POST'])
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict using Model
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_label = list(CLASS_INFO.keys())[class_index]
        confidence = float(np.max(prediction))

       # Get Disease Info
        disease_info = CLASS_INFO.get(predicted_label, {})

        return jsonify({
            'filename': filename,
            'prediction': disease_info.get('name', 'Unknown Disease'),
            'confidence': round(confidence * 100, 2),
            'description': disease_info.get('description', 'No description available.'),
            'causes': disease_info.get('causes', 'Unknown causes.'),
            'diagnosis': disease_info.get('diagnosis', 'Diagnosis not available.'),
            'treatment': disease_info.get('treatment', 'No treatment available.')
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)

