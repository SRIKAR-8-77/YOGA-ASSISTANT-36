# measure_all_angles.py
import cv2
import mediapipe as mp
import numpy as np
import os
import json

# --- Paste your helper functions here ---
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def extract_features(landmarks, visibility_threshold=0.6):
    features = {}
    lm_pose = mp_pose.PoseLandmark
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


# --- Main Measurement Logic ---
def measure_all_poses_in_folder(folder_path):
    pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    all_poses_data = {}

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return

    print(f"Scanning folder: {folder_path}\n")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if not results.pose_landmarks:
            print(f"Warning: No pose detected in {filename}")
            continue

        features = extract_features(results.pose_landmarks.landmark)

        # Get the pose name from the filename (e.g., "phalakasana.jpg" -> "phalakasana")
        pose_key = os.path.splitext(filename)[0]

        all_poses_data[pose_key] = {
            "description": pose_key.replace("_", " ").title(),
            "angles": {}
        }

        for key, value in features.items():
            if "angle" in key and not np.isnan(value):
                all_poses_data[pose_key]["angles"][key] = {
                    "ideal": int(value),
                    "threshold": 15  # Default threshold, you can adjust this later
                }

    # Print the final, complete JSON object
    print("--- COMPLETE POSE TEMPLATES (copy this entire block) ---")
    print(json.dumps(all_poses_data, indent=2))
    print("---------------------------------------------------------")


# --- IMPORTANT: Change this to the path of your reference image folder ---
FOLDER_PATH = "../PythonProject3/reference_images"
measure_all_poses_in_folder('../PythonProject3/reference_images')