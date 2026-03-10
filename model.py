import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import face_recognition
from PIL import Image

MODEL_PATH = "model.pkl"


# -------- Extract embedding from uploaded image --------
def extract_embedding_for_image(image_stream):

    image = Image.open(image_stream)
    image = image.convert("RGB")
    image_np = np.array(image)

    face_locations = face_recognition.face_locations(image_np)

    if len(face_locations) == 0:
        return None

    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    return face_encodings[0]


# -------- Load trained model --------
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# -------- Predict --------
def predict_with_model(clf, emb):
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf


# -------- Training --------
def train_model_background(dataset_dir, progress_callback=None):

    X = []
    y = []

    student_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    total_students = max(1, len(student_dirs))
    processed = 0

    for sid in student_dirs:
        folder = os.path.join(dataset_dir, sid)
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for fn in files:
            path = os.path.join(folder, fn)

            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 0:
                continue

            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) == 0:
                continue

            X.append(face_encodings[0])
            y.append(int(sid))

        processed += 1
        if progress_callback:
            pct = int((processed / total_students) * 80)
            progress_callback(pct, f"Processed {processed}/{total_students} students")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    X = np.array(X)
    y = np.array(y)

    if progress_callback:
        progress_callback(85, "Training RandomForest...")

    clf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    if progress_callback:
        progress_callback(100, "Training complete")