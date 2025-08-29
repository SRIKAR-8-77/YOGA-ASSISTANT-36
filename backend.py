import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, auth, firestore
from pydantic import BaseModel
import json

# --- Import custom modules ---
from accuracy_calculator import calculate_pose_accuracy
from nlp_processor import analyze_feedback_text

# --- Firebase Admin SDK Initialization ---
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Authentication Dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        return auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid credentials: {e}")


# --- (Helper functions and model loading remain the same) ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def extract_features(landmarks, visibility_threshold=0.6):
    features = {}
    lm_pose = mp.solutions.pose.PoseLandmark
    for i, lm in enumerate(landmarks):
        features[f'landmark_{i}_x'], features[f'landmark_{i}_y'] = lm.x, lm.y

    def get_coords_if_visible(lm_enum):
        lm = landmarks[lm_enum.value]
        return [lm.x, lm.y] if lm.visibility > visibility_threshold else None

    right_shoulder, right_elbow, right_wrist = get_coords_if_visible(lm_pose.RIGHT_SHOULDER), get_coords_if_visible(
        lm_pose.RIGHT_ELBOW), get_coords_if_visible(lm_pose.RIGHT_WRIST)
    right_hip, right_knee, right_ankle = get_coords_if_visible(lm_pose.RIGHT_HIP), get_coords_if_visible(
        lm_pose.RIGHT_KNEE), get_coords_if_visible(lm_pose.RIGHT_ANKLE)
    left_shoulder, left_elbow, left_wrist = get_coords_if_visible(lm_pose.LEFT_SHOULDER), get_coords_if_visible(
        lm_pose.LEFT_ELBOW), get_coords_if_visible(lm_pose.LEFT_WRIST)
    left_hip, left_knee, left_ankle = get_coords_if_visible(lm_pose.LEFT_HIP), get_coords_if_visible(
        lm_pose.LEFT_KNEE), get_coords_if_visible(lm_pose.LEFT_ANKLE)
    features['angle_right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist) if all(
        [right_shoulder, right_elbow, right_wrist]) else np.nan
    features['angle_left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist) if all(
        [left_shoulder, left_elbow, left_wrist]) else np.nan
    features['angle_right_shoulder'] = calculate_angle(right_hip, right_shoulder, right_elbow) if all(
        [right_hip, right_shoulder, right_elbow]) else np.nan
    features['angle_left_shoulder'] = calculate_angle(left_hip, left_shoulder, left_elbow) if all(
        [left_hip, left_shoulder, left_elbow]) else np.nan
    features['angle_right_hip'] = calculate_angle(right_shoulder, right_hip, right_knee) if all(
        [right_shoulder, right_hip, right_knee]) else np.nan
    features['angle_left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee) if all(
        [left_shoulder, left_hip, left_knee]) else np.nan
    features['angle_right_knee'] = calculate_angle(right_hip, right_knee, right_ankle) if all(
        [right_hip, right_knee, right_ankle]) else np.nan
    features['angle_left_knee'] = calculate_angle(left_hip, left_knee, left_ankle) if all(
        [left_hip, left_knee, left_ankle]) else np.nan
    return features


model = load_model("my_model_71.51.keras")
label_encoder = joblib.load("label_encoder_71.51.pkl")
scaler = joblib.load("scaler_71.51.pkl")
column_means = joblib.load("col_means_71.51.pkl")
mp_pose = mp.solutions.pose


def process_video_logic(video_path):
    FRAME_SKIP = 15
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prediction_buffer, STABILITY_THRESHOLD = [], 3
    current_pose, pose_durations, pose_counts = "No Pose Detected", {}, {}
    pose_accuracies = {}
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        stable_pose = current_pose

        if frame_count % FRAME_SKIP == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(image_rgb)

            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks.landmark)
                feature_df = pd.DataFrame([features], columns=column_means.index)
                feature_df.fillna(column_means, inplace=True)
                feature_scaled = scaler.transform(feature_df).astype("float32")
                prediction = model.predict(feature_scaled, verbose=0)

                # --- 👇 UPDATED: Increased threshold for stricter classification ---
                CONFIDENCE_THRESHOLD = 0.60  # Require 85% confidence
                prediction_confidence = np.max(prediction)

                predicted_pose = "No Pose Detected"
                if prediction_confidence >= CONFIDENCE_THRESHOLD:
                    class_idx = np.argmax(prediction)
                    predicted_pose = label_encoder.inverse_transform([class_idx])[0]

                prediction_buffer.append(predicted_pose)
                if len(prediction_buffer) > STABILITY_THRESHOLD:
                    prediction_buffer.pop(0)

                if len(prediction_buffer) == STABILITY_THRESHOLD and len(set(prediction_buffer)) == 1:
                    stable_pose = prediction_buffer[0]
                    if stable_pose != "No Pose Detected":
                        accuracy_result = calculate_pose_accuracy(features, stable_pose)
                        if stable_pose not in pose_accuracies:
                            pose_accuracies[stable_pose] = []
                        pose_accuracies[stable_pose].append(accuracy_result['accuracy'])

        pose_durations[current_pose] = pose_durations.get(current_pose, 0) + (1 / fps)

        if stable_pose != current_pose:
            if stable_pose != "No Pose Detected":
                pose_counts[stable_pose] = pose_counts.get(stable_pose, 0) + 1
            current_pose = stable_pose

    cap.release()
    summary_data = []
    for pose, duration in pose_durations.items():
        if pose != "No Pose Detected":
            count = pose_counts.get(pose, 0)
            accuracies = pose_accuracies.get(pose, [])
            avg_accuracy = round(np.mean(accuracies)) if accuracies else 0

            summary_data.append({
                "Yoga Pose": pose,
                "Total Time (s)": int(duration),
                "Repetitions": count,
                "Average Accuracy (%)": avg_accuracy
            })

    total_time = sum(item['Total Time (s)'] for item in summary_data)
    return {"summary": summary_data, "total_time": total_time}


# --- FastAPI App and Endpoints ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# --- Pydantic Models for API Data ---
class JournalItem(BaseModel):
    entry_text: str


# === INDEPENDENT FEATURE ENDPOINTS ===
@app.post("/analyze-session/")
async def analyze_video(user: dict = Depends(get_current_user), file: UploadFile = File(...)):
    user_id = user['uid']
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(await file.read())
            video_path = tfile.name
        results = process_video_logic(video_path)
        session_data = {"date": datetime.datetime.utcnow(), "summary": results["summary"],
                        "total_time": results["total_time"]}
        session_ref = db.collection("users").document(user_id).collection("sessions").add(session_data)
        return {"message": "Analysis complete", "sessionId": session_ref[1].id}
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)


@app.post("/add-journal-entry/")
async def submit_journal(journal: JournalItem, user: dict = Depends(get_current_user)):
    user_id = user['uid']
    print(f"📥 Received journal entry from {user_id}: {journal.entry_text}")
    analysis = analyze_feedback_text(journal.entry_text)
    print(f"🔎 Sentiment analysis: {analysis}")
    journal_data = {
        "date": datetime.datetime.utcnow(),
        "entry_text": journal.entry_text,
        "sentiment": analysis.get("sentiment"),
        "sentiment_score": analysis.get("sentiment_score")
    }
    db.collection("users").document(user_id).collection("journal").add(journal_data)
    print(f"✅ Saved journal entry for {user_id}")
    return {"message": "Journal entry saved successfully", "analysis": journal_data}


# --- DATA FETCHING ENDPOINTS ---
@app.get("/get-sessions/")
async def get_sessions(user: dict = Depends(get_current_user)):
    user_id = user['uid']
    sessions_ref = db.collection("users").document(user_id).collection("sessions").order_by("date",
                                                                                            direction=firestore.Query.DESCENDING).stream()
    sessions = []
    for session in sessions_ref:
        s_data = session.to_dict()
        s_data['id'] = session.id
        s_data['date'] = s_data['date'].isoformat()
        sessions.append(s_data)
    return sessions


@app.get("/get-journal-entries/")
async def get_journal_entries(user: dict = Depends(get_current_user)):
    user_id = user['uid']
    print(f"📤 Fetching journal entries for {user_id}")
    journal_ref = db.collection("users").document(user_id).collection("journal").order_by("date",
                                                                                          direction=firestore.Query.DESCENDING).stream()
    entries = []
    for entry in journal_ref:
        e_data = entry.to_dict()
        e_data['id'] = entry.id
        e_data['date'] = e_data['date'].isoformat()
        entries.append(e_data)
    print(f"📊 Found {len(entries)} journal entries")
    return entries


@app.get("/ping")
async def ping():
    print("✅ Backend received a ping!", flush=True)
    return {"message": "pong"}