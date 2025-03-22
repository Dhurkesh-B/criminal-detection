import os
import json
import cv2
import face_recognition
import numpy as np
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv


# Directories
face_mark_folder = "face_mark_json"
criminal_info_folder = "criminal_info_json"
image_attendance_folder = "ImageAttendance"
load_dotenv()

# Create directories if they don’t exist
for folder in [face_mark_folder, criminal_info_folder, image_attendance_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)

def capture_images(person_name, num_images=3):
    cap = cv2.VideoCapture(0)
    count = 0
    captured_images = []

    while count < num_images:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        img_path = os.path.join(image_attendance_folder, f"{person_name}_{count}.jpg")
        cv2.imwrite(img_path, img)
        captured_images.append(img_path)
        count += 1

        cv2.imshow("Capturing Images", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_images

def process_video(person_name, video_file, num_images=3):
    video_path = os.path.join(image_attendance_folder, f"{person_name}_temp_video.mp4")
    video_file.save(video_path)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_images)  # Evenly space the 3 frames
    count = 0
    captured_images = []
    frame_idx = 0

    while count < num_images and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if frame_idx % frame_interval == 0:  # Extract frame at interval
            img_path = os.path.join(image_attendance_folder, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            captured_images.append(img_path)
            count += 1
        
        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    os.remove(video_path)
    return captured_images

def process_photos(person_name, photo_files):
    captured_images = []
    for i, photo_file in enumerate(photo_files):
        photo_path = os.path.join(image_attendance_folder, f"{person_name}_{i}.jpg")
        photo_file.save(photo_path)
        captured_images.append(photo_path)
    return captured_images

def save_to_json(data, folder, filename):
    json_path = os.path.join(folder, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {json_path}")

def process_and_store_encodings(person_name, captured_images):
    encodings_list = []

    # Process each image to detect and encode faces
    for img_path in captured_images:
        cur_img = cv2.imread(img_path)
        cur_img_rgb = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(cur_img_rgb)
        
        if encodings:
            encodings_list.append(encodings[0].tolist())  # Convert numpy array to list for JSON
        else:
            print(f"Warning: No face detected in {img_path}")

    if not encodings_list:
        return False

    # Save encodings to face_mark_json
    encoding_data = {
        "name": person_name,
        "encodings": encodings_list
    }
    save_to_json(encoding_data, face_mark_folder, person_name)
    return True

@app.route('/add_suspect', methods=['POST'])
def add_suspect():
    person_namekey = request.form.get('namekey')
    suspect_data = {
        'Full Name': request.form.get('full_name'),
        'Age': int(request.form.get('age')),
        'complaint': request.form.get('complaint'),
        'Nationality': request.form.get('nationality')
    }
    input_method = request.form.get('input_method')

    # Save suspect info
    save_to_json(suspect_data, criminal_info_folder, person_namekey)

    # Process based on input method
    if input_method == 'video' and 'video' in request.files:
        video_file = request.files['video']
        captured_images = process_video(person_namekey, video_file)
    elif input_method == 'photo':
        photo_files = [request.files.get(f'photo{i+1}') for i in range(3)]
        if None in photo_files or any(not f for f in photo_files):
            return jsonify({"message": "Please upload exactly 3 photos"}), 400
        captured_images = process_photos(person_namekey, photo_files)
    elif input_method == 'live':
        captured_images = capture_images(person_namekey)
    else:
        return jsonify({"message": "Invalid input method or missing file"}), 400

    # Process and store face encodings
    if process_and_store_encodings(person_namekey, captured_images):
        return jsonify({"message": "Suspect added and face encodings stored successfully"}), 200
    else:
        return jsonify({"message": "Failed to detect or store face encodings"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)