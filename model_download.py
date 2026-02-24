# download_models.py
import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    
    if os.path.exists(filepath):
        print(f"{filename} already exists. Skipping.")
        return
    
    urllib.request.urlretrieve(url, filepath)
    print(f"Downloaded {filename} successfully!")

# Pose model
pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
download_file(pose_url, "pose_landmarker.task")

# Hand model
hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
download_file(hand_url, "hand_landmarker.task")

print("\nAll models downloaded!")