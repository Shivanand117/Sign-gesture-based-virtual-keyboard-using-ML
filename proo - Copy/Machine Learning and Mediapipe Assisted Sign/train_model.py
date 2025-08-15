import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Reshape, Dropout, Flatten
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_gesture_data(data_dir):
    """Load gesture data from JSON files."""
    try:
        logging.info(f"Loading gesture data from {data_dir}")
        X = []
        y = []
        gesture_map = {}
        
        # Get list of gesture directories
        gestures = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Create gesture mapping
        for i, gesture in enumerate(sorted(gestures)):
            gesture_map[i] = gesture
            
        # Load data for each gesture
        for gesture_idx, gesture in enumerate(sorted(gestures)):
            gesture_dir = os.path.join(data_dir, gesture)
            sample_files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
            
            for sample_file in sample_files:
                with open(os.path.join(gesture_dir, sample_file), 'r') as f:
                    landmarks_data = json.load(f)
                    # Take only the first hand's landmarks if multiple hands are detected
                    if landmarks_data:
                        landmarks = np.array(landmarks_data[0])
                        # Reshape to (21, 3) without extra dimension
                        landmarks = landmarks.reshape(21, 3)
                        X.append(landmarks)
                        y.append(gesture_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Save gesture mapping
        with open('gesture_map.json', 'w') as f:
            json.dump(gesture_map, f)
            
        logging.info(f"Loaded {len(X)} samples across {len(gestures)} gestures")
        return X, y, len(gestures)
    
    except Exception as e:
        logging.error(f"Error loading gesture data: {str(e)}", exc_info=True)
        raise

def create_model(num_classes):
    """Create and compile the gesture recognition model."""
    try:
        logging.info("Creating model architecture")
        model = Sequential([
            Input(shape=(21, 3)),  # 21 landmarks, 3 coordinates each
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Flatten(),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info("Model created successfully")
        return model
    
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}", exc_info=True)
        raise

def train_gesture_model(data_dir, epochs=100):
    """Train the gesture recognition model."""
    try:
        # Load and preprocess data
        X, y, num_classes = load_gesture_data(data_dir)
        
        # Create and compile model
        model = create_model(num_classes)
        
        # Train model
        logging.info("Starting model training")
        history = model.fit(
            X, y,
            epochs=epochs,
            validation_split=0.2,
            verbose=1,
            batch_size=32,
        )
        
        # Save model
        model.save('gesture_model.h5')
        logging.info("Model saved successfully")
        
        return history
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        train_gesture_model('dataset')
    except Exception as e:
        logging.error("Error in main training process", exc_info=True)
        raise
