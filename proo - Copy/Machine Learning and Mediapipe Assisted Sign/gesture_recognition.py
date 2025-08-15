import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pandas as pd

class GestureRecognizer:
    def __init__(self):
        self.model = self._build_model()
        self.gesture_map = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
            5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z', 26: 'SPACE', 27: 'DELETE'
        }

    def _build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(30, 63), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(28, activation='softmax')  # 26 letters + space + delete
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_data(self, landmarks_sequence):
        # Convert landmarks sequence to model input format
        return np.array(landmarks_sequence).reshape(1, 30, 63)

    def predict_gesture(self, landmarks_sequence):
        if len(landmarks_sequence) < 30:
            return None
        
        X = self.preprocess_data(landmarks_sequence)
        prediction = self.model.predict(X)
        gesture_index = np.argmax(prediction[0])
        return self.gesture_map.get(gesture_index)

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )
        return history

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load_weights(path)
