import json
import numpy as np

# --- Load the Pose Templates Database ---
# This file must be in the same directory.
try:
    with open('pose_templates.json', 'r') as f:
        POSE_TEMPLATES = json.load(f)
except FileNotFoundError:
    print("Error: `pose_templates.json` not found. Please create it.")
    POSE_TEMPLATES = {}


def calculate_pose_accuracy(user_features: dict, detected_pose_name: str):
    """
    Calculates the accuracy of a user's pose by comparing their joint angles
    to a pre-defined template.

    Args:
        user_features (dict): A dictionary of the user's calculated angles and landmarks.
        detected_pose_name (str): The name of the pose detected by the classification model.

    Returns:
        dict: A dictionary containing the overall accuracy percentage and specific feedback.
    """
    # Sanitize the pose name to match the JSON keys (e.g., "Warrior II" -> "warrior_ii")
    pose_key = detected_pose_name.lower().replace(" ", "_")

    if pose_key not in POSE_TEMPLATES:
        return {
            "accuracy": 100,  # Default to 100 if no template exists
            "feedback": "No template available for this pose.",
            "details": []
        }

    template = POSE_TEMPLATES[pose_key]
    template_angles = template.get('angles', {})

    total_error = 0
    angles_compared = 0
    feedback_details = []

    for angle_name, template_values in template_angles.items():
        user_angle = user_features.get(angle_name)

        if user_angle is not None and not np.isnan(user_angle):
            ideal_angle = template_values['ideal']
            threshold = template_values['threshold']

            error = abs(user_angle - ideal_angle)

            if error <= threshold:
                normalized_error = 0
                feedback_details.append({
                    "angle": angle_name,
                    "status": "correct",
                    "message": f"Your {angle_name.replace('_', ' ')} is correct."
                })
            else:
                normalized_error = min(100, ((error - threshold) / 90) * 100)
                direction = "extend" if user_angle < ideal_angle else "bend"
                diff = round(abs(user_angle - ideal_angle))
                feedback_details.append({
                    "angle": angle_name,
                    "status": "incorrect",
                    "message": f"Try to {direction} your {angle_name.replace('_', ' ')} by about {diff} degrees."
                })

            total_error += normalized_error
            angles_compared += 1

    if angles_compared == 0:
        return {
            "accuracy": 0,
            "feedback": "Could not compare any angles for this pose.",
            "details": []
        }

    overall_accuracy = 100 - (total_error / angles_compared)

    return {
        "accuracy": round(overall_accuracy),
        "feedback": "Great form!" if overall_accuracy > 85 else "Good effort, a few adjustments can improve your form.",
        "details": feedback_details
    }