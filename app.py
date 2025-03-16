import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To suppress unnecessary TensorFlow warnings

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import keras.initializers
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder='templates/index.html')


def variance_scaling_fixed(**kwargs):
    kwargs.pop('dtype', None)  # Remove dtype argument
    return keras.initializers.VarianceScaling(**kwargs)

tf.keras.utils.get_custom_objects().update({'VarianceScaling': variance_scaling_fixed})

custom_objects = {
    'Zeros': initializers.Zeros
}

# Load the trained model (handle the VarianceScaling error)
MODEL_PATH = "./models/model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects, safe_mode=False)

# Class Information (Disease Details)
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

# Upload folder for storing uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Image size (224, 224) as per the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

@app.route('/')
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

        # Preprocess the image and make prediction
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_label = list(CLASS_INFO.keys())[class_index]
        confidence = float(np.max(prediction))

        # Get disease details
        disease_info = CLASS_INFO[predicted_label]

        return jsonify({
            'filename': filename,
            'prediction': disease_info['name'],
            'confidence': round(confidence * 100, 2),
            'description': disease_info['description'],
            'causes': disease_info['causes'],
            'treatment': disease_info['treatment'],
            'diagnosis': disease_info['diagnosis']
        })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

