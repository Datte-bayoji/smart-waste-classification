# predict.py
import sys, json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL = 'waste_model.h5'
IMG_SIZE = (224,224)

model = load_model(MODEL)
with open('idx_to_class.json') as f:
    idx_to_class = json.load(f)

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = idx_to_class[str(idx)]
    print("Prediction:", label, "Prob:", float(preds[idx]))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
