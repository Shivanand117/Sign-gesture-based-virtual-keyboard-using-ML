import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollector:
    def __init__(self, data_dir='dataset'):
        try:
            logging.info(f"Initializing DataCollector with data_dir: {data_dir}")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.data_dir = data_dir
            
            # Create dataset directory if it doesn't exist
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                logging.info(f"Created data directory: {data_dir}")
        except Exception as e:
            logging.error(f"Error initializing DataCollector: {str(e)}", exc_info=True)
            raise

    def collect_gesture_data(self, gesture_name, num_samples=100):
        """Collect hand gesture data for training."""
        try:
            gesture_dir = os.path.join(self.data_dir, gesture_name)
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)
                logging.info(f"Created directory for gesture: {gesture_name}")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera")
            
            sample_count = 0
            
            logging.info(f"Starting data collection for gesture: {gesture_name}")
            print(f"Collecting data for gesture: {gesture_name}")
            print("Press 'c' to capture a sample, 'q' to quit")

            while sample_count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame from camera")
                    continue

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                # Display sample count
                cv2.putText(
                    frame,
                    f"Samples: {sample_count}/{num_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Data Collection', frame)
                key = cv2.waitKey(1)

                if key == ord('c') and results.multi_hand_landmarks:
                    # Save landmark data
                    landmarks_data = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        landmarks_data.append(landmarks)

                    # Save to file
                    filename = os.path.join(gesture_dir, f'sample_{sample_count}.json')
                    with open(filename, 'w') as f:
                        json.dump(landmarks_data, f)
                    
                    sample_count += 1
                    logging.info(f"Saved sample {sample_count} for gesture {gesture_name}")
                    print(f"Saved sample {sample_count}")

                elif key == ord('q'):
                    logging.info(f"Data collection for gesture {gesture_name} stopped by user")
                    break

            cap.release()
            cv2.destroyAllWindows()
            logging.info(f"Completed data collection for gesture {gesture_name}")
            
        except Exception as e:
            logging.error(f"Error collecting data for gesture {gesture_name}: {str(e)}", exc_info=True)
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            raise

if __name__ == "__main__":
    try:
        collector = DataCollector()
        
        # Collect data for each letter
        gestures = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        gestures.extend(['SPACE', 'DELETE'])
        
        for gesture in gestures:
            print(f"\nCollecting data for gesture: {gesture}")
            input("Press Enter to start collecting samples for this gesture...")
            collector.collect_gesture_data(gesture, num_samples=100)
    except Exception as e:
        logging.error("Error in main data collection process", exc_info=True)
        raise
