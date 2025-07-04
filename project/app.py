from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model (1).h5')

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:
            result = 'Healthy'
        elif predicted_class == 1:
            result = 'Osteopenia'
        elif predicted_class == 2:
            result = 'Osteoporosis'

        return render_template('result.html', result=result, filepath=filename)

    else:
        return render_template('error.html', error='Invalid file format. Please upload an image file.')

if __name__ == '__main__':
    app.run(debug=True)
