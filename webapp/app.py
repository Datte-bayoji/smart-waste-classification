from flask import Flask, render_template, request
import os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL = '../waste_model.h5'  # adjust path if needed
IMG_SIZE = (224, 224)

# load model and class map
model = load_model(MODEL)
with open('../idx_to_class.json') as f:
    idx_to_class = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            # predict
            img = image.load_img(filepath, target_size=IMG_SIZE)
            x = image.img_to_array(img)/255.0
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
            label = idx_to_class[str(idx)]
            prob = float(preds[idx])

            return render_template('index.html', filename=filename, label=label, prob=prob)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
