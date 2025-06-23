import joblib
from PIL import Image
import numpy as np

model = joblib.load('model/plant_disease_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

def preprocess_image(image_path):
    img = Image.open(image_path).resize((64, 64)).convert("RGB")
    img_array = np.array(img).flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_array)
    return img_scaled

def predict_disease(image_path):
    processed = preprocess_image(image_path)
    prediction = model.predict(processed)
    confidence = model.predict_proba(processed).max() * 100
    label = label_encoder.inverse_transform(prediction)[0]
    return {"label": label, "confidence": round(confidence, 2)}
