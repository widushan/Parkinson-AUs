from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import pickle
from io import BytesIO
from PIL import Image
import base64

# Load model and scaler
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_au_features(landmarks):
    if not landmarks:
        return np.zeros(9)
    l = landmarks.landmark
    left_eye = l[159]
    left_brow = l[70]
    right_eye = l[386]
    right_brow = l[300]
    cheek_left = l[205]
    lip_corner_left = l[61]
    lip_corner_right = l[291]
    outer_brow_left = l[66]
    au_01 = np.mean([abs(left_brow.y - left_eye.y), abs(right_brow.y - right_eye.y)])
    au_06 = abs(cheek_left.y - left_eye.y)
    au_12 = abs(lip_corner_left.x - lip_corner_right.x)
    au_04 = abs(left_brow.y - right_brow.y)
    au_07 = abs(left_eye.y - l[145].y)
    au_09 = 0.0
    au_02 = abs(outer_brow_left.y - left_eye.y)
    return np.array([au_01, au_06, au_12, au_04, au_07, au_09, au_01, au_02, au_04])

@app.route('/')
def home():
    return "Backend is running. Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    user_age = data.get('age', 60)
    user_gender = data.get('gender', 1)
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    au_features = np.zeros(9)
    if results.multi_face_landmarks:
        au_features = extract_au_features(results.multi_face_landmarks[0])
    input_features = np.append(au_features, [user_age, user_gender])
    input_scaled = scaler.transform([input_features])
    pred = rf_model.predict(input_scaled)[0]
    pred_proba = rf_model.predict_proba(input_scaled)[0][1]
    return jsonify({'status': 'PD' if pred == 1 else 'Healthy', 'probability': float(pred_proba)})

if __name__ == '__main__':
    app.run(debug=True)