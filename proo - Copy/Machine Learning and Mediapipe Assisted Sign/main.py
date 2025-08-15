import cv2
import mediapipe as mp
import numpy as np
import keyboard
import pyttsx3
import json
import tensorflow as tf
from collections import deque
import time
import os
from data_collection import DataCollector
from train_model import train_gesture_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GestureKeyboard:
    def __init__(self, model_path='gesture_model.h5', gesture_map_path='gesture_map.json'):
        try:
            logging.info("Initializing GestureKeyboard...")
            
            # Initialize MediaPipe
            logging.info("Setting up MediaPipe...")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils

            # Check if model files exist
            self.model = None
            self.gesture_map = {}
            logging.info(f"Checking for model files at {model_path} and {gesture_map_path}")
            
            if os.path.exists(model_path) and os.path.exists(gesture_map_path):
                logging.info("Loading existing model...")
                self.model = tf.keras.models.load_model(model_path)
                with open(gesture_map_path, 'r') as f:
                    self.gesture_map = {int(k): v for k, v in json.load(f).items()}
            else:
                logging.info("Model files not found. Starting training mode...")
                self.training_mode()
                logging.info("Training complete. Loading new model...")
                self.model = tf.keras.models.load_model(model_path)
                with open(gesture_map_path, 'r') as f:
                    self.gesture_map = {int(k): v for k, v in json.load(f).items()}

            # Initialize text-to-speech engine
            logging.info("Initializing text-to-speech engine...")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)

            # Initialize buffers
            self.gesture_buffer = deque(maxlen=30)
            self.text_buffer = ""
            self.last_gesture_time = time.time()
            self.gesture_cooldown = 1.0  # seconds
            
            logging.info("GestureKeyboard initialization complete!")
            
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def training_mode(self):
        """Collect data and train the model."""
        try:
            logging.info("Starting training mode...")
            print("\n=== Starting Data Collection ===")
            print("We'll collect gesture data for each letter and special commands.")
            input("Press Enter when you're ready to start...")

            # Initialize data collector
            collector = DataCollector()
            
            # Create dataset directory if it doesn't exist
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
                logging.info("Created dataset directory")
            
            # Collect data for basic gestures first
            basic_gestures = ['A', 'B', 'C', 'D', 'E']  # Start with a few gestures for testing
            for gesture in basic_gestures:
                logging.info(f"Collecting data for gesture: {gesture}")
                print(f"\nCollecting data for gesture: {gesture}")
                print("Position your hand and press 'c' to capture (collect about 30 samples)")
                print("Press 'q' when done with this gesture")
                collector.collect_gesture_data(gesture, num_samples=30)

            logging.info("Data collection complete")
            print("\n=== Data Collection Complete ===")
            print("Training the model...")
            
            # Train the model
            train_gesture_model('dataset')
            
            logging.info("Model training complete")
            print("\n=== Training Complete ===")
            print("Model has been trained and saved!")
            
        except Exception as e:
            logging.error(f"Error during training mode: {str(e)}", exc_info=True)
            raise

    def process_landmarks(self, hand_landmarks):
        """Convert hand landmarks to model input format."""
        try:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        except Exception as e:
            logging.error(f"Error during landmark processing: {str(e)}", exc_info=True)
            raise

    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks using the trained model."""
        try:
            # Reshape landmarks to match model input shape
            X = landmarks.reshape(1, 21, 3)
            
            # Get model prediction
            prediction = self.model.predict(X, verbose=0)
            gesture_index = np.argmax(prediction[0])
            confidence = prediction[0][gesture_index]
            
            if confidence > 0.8:  # Only return gesture if confidence is high enough
                return self.gesture_map[gesture_index]
            return None
        except Exception as e:
            logging.error(f"Error during gesture prediction: {str(e)}", exc_info=True)
            raise

    def handle_gesture(self, gesture):
        """Handle recognized gesture and update text buffer."""
        try:
            if gesture == 'SPACE':
                self.text_buffer += ' '
                self.engine.say("space")
            elif gesture == 'DELETE':
                if self.text_buffer:
                    self.text_buffer = self.text_buffer[:-1]
                    self.engine.say("delete")
            else:
                self.text_buffer += gesture
                self.engine.say(gesture)
            
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during gesture handling: {str(e)}", exc_info=True)
            raise

    def run(self):
        """Main application loop."""
        try:
            cap = cv2.VideoCapture(0)
            
            # Set up window
            cv2.namedWindow('Gesture Keyboard', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gesture Keyboard', 1280, 720)

            print("\n=== Gesture Keyboard Started ===")
            print("Make hand gestures to type")
            print("Press ESC to exit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                # Draw hand landmarks and process gestures
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        # Process landmarks for gesture recognition
                        if time.time() - self.last_gesture_time > self.gesture_cooldown:
                            landmarks = self.process_landmarks(hand_landmarks)
                            gesture = self.predict_gesture(landmarks)

                            if gesture:
                                self.handle_gesture(gesture)
                                self.last_gesture_time = time.time()

                # Display text buffer
                cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 100), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Text: {self.text_buffer}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                # Display help text
                help_text = "Press 'ESC' to exit | Make hand gestures to type"
                cv2.putText(
                    frame,
                    help_text,
                    (20, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Gesture Keyboard', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Error during application loop: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    keyboard = GestureKeyboard()
    keyboard.run()
