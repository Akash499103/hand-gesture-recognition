import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# ── MediaPipe Setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ── Gesture Labels ───────────────────────────────────────────
gesture_labels = ['fist', 'open_palm', 'thumbs_up', 'peace', 'pointing']

# ── Extract Landmarks ────────────────────────────────────────
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks), result.multi_hand_landmarks[0]
    return None, None

# ── Build CNN Model ──────────────────────────────────────────
def build_model(num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ── Collect Training Data ────────────────────────────────────
def collect_data(samples_per_gesture=200):
    cap = cv2.VideoCapture(0)
    data = []
    labels = []

    for idx, gesture in enumerate(gesture_labels):
        print(f"\nGet ready for gesture: {gesture.upper()}")
        print("Press SPACE to start collecting...")

        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f"Prepare: {gesture} | Press SPACE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Collecting Data", frame)
            if cv2.waitKey(1) == 32:  # SPACE
                break

        count = 0
        while count < samples_per_gesture:
            ret, frame = cap.read()
            landmarks, hand_lms = extract_landmarks(frame)
            if landmarks is not None:
                data.append(landmarks)
                labels.append(idx)
                count += 1
                if hand_lms:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"{gesture}: {count}/{samples_per_gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Collecting Data", frame)
            cv2.waitKey(1)

        print(f"Collected {samples_per_gesture} samples for {gesture}")

    cap.release()
    cv2.destroyAllWindows()
    return np.array(data), np.array(labels)

# ── Train Model ──────────────────────────────────────────────
def train_model():
    print("Collecting training data...")
    X, y = collect_data()

    y_cat = to_categorical(y, num_classes=len(gesture_labels))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = build_model(len(gesture_labels))
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    model.save('gesture_model.h5')
    print("Model saved as gesture_model.h5")
    return model

# ── Real-Time Prediction ─────────────────────────────────────
def predict_realtime(model):
    cap = cv2.VideoCapture(0)
    print("\nStarting real-time gesture recognition. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        landmarks, hand_lms = extract_landmarks(frame)

        if landmarks is not None:
            pred = model.predict(landmarks.reshape(1, -1), verbose=0)
            gesture_idx = np.argmax(pred)
            confidence = np.max(pred) * 100
            gesture_name = gesture_labels[gesture_idx]

            if hand_lms:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    choice = input("1. Train new model\n2. Load existing model\nChoice: ")

    if choice == '1':
        model = train_model()
    else:
        model = tf.keras.models.load_model('gesture_model.h5')
        print("Model loaded.")

    predict_realtime(model)