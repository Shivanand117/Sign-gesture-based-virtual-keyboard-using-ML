from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand to avoid confusion
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# === Absolute paths for model and gesture map ===
MODEL_PATH = r"D:\praveen\proo\Machine Learning and Mediapipe Assisted Sign\gesture_model.h5"
GESTURE_MAP_PATH = r"D:\praveen\proo\Machine Learning and Mediapipe Assisted Sign\gesture_map.json"

# Load the trained model and gesture map
model = load_model(MODEL_PATH)
with open(GESTURE_MAP_PATH, 'r') as f:
    gesture_map = json.load(f)

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to the wrist position."""
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    normalized = []
    for lm in landmarks:
        norm_lm = [lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]]
        norm_lm = [x * 100 for x in norm_lm]  # Scale up for better precision
        normalized.extend(norm_lm)
    return np.array(normalized)

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = normalize_landmarks(hand_landmarks.landmark)
                landmarks = landmarks.reshape(1, 21, 3)
                
                prediction = model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                if confidence > 0.85:
                    gesture = gesture_map[str(predicted_class)]
                    text = f"Gesture: {gesture} ({confidence:.2f})"
                    cv2.putText(frame, text, (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    bar_length = int(confidence * 200)
                    cv2.rectangle(frame, (10, 60), (10 + bar_length, 70), 
                                (0, 255, 0), -1)
                else:
                    cv2.putText(frame, "Uncertain gesture", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        cv2.putText(frame, "Show hand gesture in the frame", (10, 440), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        cv2.destroyAllWindows()

