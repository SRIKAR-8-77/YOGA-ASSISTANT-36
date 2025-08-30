<br>
<h1 align="center">ZENFLOW</h1>
<h2 align="center">Yoga Pose AI Assistant</h2>
<p align="center">A real-time yoga assistant powered by computer vision and AI.</p>

About the Project

This repository contains the backend and machine learning components for a Yoga Pose AI Assistant. The application uses MediaPipe to detect human poses from images or video feeds and a custom-trained neural network to classify the yoga pose. It provides real-time feedback on pose accuracy and offers personalized guidance from an AI-powered coach.


<img width="2795" height="1299" alt="Screenshot 2025-08-31 023908" src="https://github.com/user-attachments/assets/8ffae7de-048c-42ff-b97a-7ce01264a61a" />

<img width="788" height="1128" alt="Screenshot 2025-08-31 023921" src="https://github.com/user-attachments/assets/00f7c99d-38c2-40de-8c00-72130a3d6dc4" />

<img width="1583" height="957" alt="Screenshot 2025-08-31 023938" src="https://github.com/user-attachments/assets/dec9dfdf-0182-4f07-9381-8d3b39dc74c4" />

<img width="1568" height="1056" alt="Screenshot 2025-08-31 023947" src="https://github.com/user-attachments/assets/a8f1d13c-e6cb-461a-9020-dd31cabc5444" />

</p>

✨ Key Features

->Pose Classification: Identifies and classifies a wide range of yoga poses using a Keras model.

->Accuracy Assessment: Compares the user's pose to an ideal template and provides a numerical accuracy score.

->Personalized Feedback: Delivers specific, actionable feedback on how to improve a pose (e.g., "Bend your knee by 5 degrees").

->AI Coach Integration: An AI assistant powered by Google's Gemini API provides personalized yoga and wellness advice.

->Real-time Interaction: The backend is built with FastAPI to handle real-time video stream processing and chat interactions.

->User Authentication: Uses Firebase for secure user authentication and to store user-specific data and chat history.

## 💻 Technical Stack

Our application is built on a robust and modern stack, leveraging industry-leading tools for each component of the system.

| Category           | Technology                                             |
|-------------------|--------------------------------------------------------|
| Backend            | Python with FastAPI                                     |
| Pose Detection     | MediaPipe                                              |
| Machine Learning   | TensorFlow/Keras (.keras), scikit-learn (.pkl)        |
| AI/NLP             | Google's Gemini API, Hugging Face transformers        |
| Database           | Google Firestore (via Firebase Admin SDK)             |
| Environment        | dotenv                                                 |

## 📂 Repository Structure

The repository is organized to separate core application logic, machine learning models, and data utilities. This makes it easy to navigate and understand each file’s purpose.

```plaintext
.
├── 🖥️ backend.py                — Main API for all app functionality
├── 🔌 firebase_client.py        — Handles Firebase connection
├── 🤖 yoga_assistant.py         — CrewAI setup for the AI coach
├── 📊 accuracy_calculator.py    — Core logic for pose scoring and feedback
├── 📝 nlp_processor.py          — Utilities for natural language tasks
├── 📐 measure_all_angles.py     — Script for generating pose templates
├── 🧪 test.py                   — Standalone script to test LLM connection
│
├── 🏋️ my_model_71.51.keras      — Trained Keras pose classification model
├── 📏 scaler_71.51.pkl          — StandardScaler for data normalization
└── 📓 YOGA_NN_NOTEBOOK.ipynb    — Jupyter Notebook for model training & evaluation


🚀 Getting Started
Prerequisites
Python 3.8+

Set up a Firebase Project, enable Firestore and Authentication, and generate a serviceAccountKey.json.

Obtain a Google Gemini API Key from Google AI Studio.

Some NLP features may require a Hugging Face token.

Installation
Clone the repository and install the required dependencies:

# Clone the repository
git clone <repository_url>

# Navigate into the project directory
cd <repository_name>

# Install Python dependencies
pip install -r requirements.txt

Configuration
Create a .env file in the root directory and add your API keys:

GEMINI_API_KEY=your_google_gemini_api_key
HUGGING_FACE_HUB_TOKEN=your_huggingface_token

Running the Application
To start the FastAPI backend server:

uvicorn backend:app --reload
