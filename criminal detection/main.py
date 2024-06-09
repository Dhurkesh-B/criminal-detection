import os
import shutil
import json
import cv2
import face_recognition
import numpy as np
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
import geocoder
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
from os.path import basename
import sys


face_mark_folder = "face_mark_json"
criminal_info_folder = "criminal_info_json"

if not os.path.exists(face_mark_folder):
    os.makedirs(face_mark_folder)

if not os.path.exists(criminal_info_folder):
    os.makedirs(criminal_info_folder)

# Dictionary to keep track of saved images
image_count = {}

def capture_images(person_name, num_images=5):
    cap = cv2.VideoCapture(0)
    count = 0
    captured_images = []

    while count < num_images:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        img_path = os.path.join("ImageAttendance", f"{person_name}_{count}.jpg")
        cv2.imwrite(img_path, img)
        captured_images.append(img_path)
        count += 1

        cv2.imshow("Capturing Images", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_images


def save_to_json(data, folder, filename):
    json_path = os.path.join(folder, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {json_path}")


def main():
    print("1. Check the camera for the suspect")
    print("2. Add a new suspect")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        run_recognition_program()
    elif choice == '2':
        add_new_suspect()
    else:
        print("Invalid choice. Exiting.")


def add_new_suspect():
    person_namekey = input("Enter the new person's namekey: ")
    di = {}
    di['Full Name'] = input('Enter the Full Name of the suspect: ')
    di['Age'] = int(input('Enter the age of the suspect: '))
    di['complaint'] = input('Enter the complaint: ')
    di['Nationality'] = input('Enter the Nationality: ')

    save_to_json(di, criminal_info_folder, person_namekey)

    # Capture images
    captured_images = capture_images(person_namekey)
    print('bro over here')
    # Encode the captured images and save the encodings
    print(captured_images)
    face_encodings = []
    for img_path in captured_images:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            face_encodings.append(encodings[0])
    print('finally got here bro')

    face_mark_data = {
        "namekey": person_namekey,
        "encodings": [enc.tolist() for enc in face_encodings]
    }
    print('bro i came here')

    save_to_json(face_mark_data, face_mark_folder, person_namekey)

    # Save images to ImageAttendance folder
    for img_path in captured_images:
        destination_path = os.path.join("ImageAttendance", os.path.basename(img_path))
        shutil.move(img_path, destination_path)

    print("New suspect added successfully. The program will now exit.")
    sys.exit(0)

def run_recognition_program():
    current_directory = os.getcwd()
    output_folder = os.path.join(current_directory, "RecognizedFaces")
    photo_detect_folder = os.path.join(current_directory, "photo-detect")
    pdf_folder = os.path.join(current_directory, "pdf-data")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(photo_detect_folder):
        os.makedirs(photo_detect_folder)

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    face_encodings_dict = {}

    # Load all face encodings from face_mark_json folder
    for filename in os.listdir(face_mark_folder):
        if filename.endswith(".json"):
            with open(os.path.join(face_mark_folder, filename), 'r') as f:
                data = json.load(f)
                face_encodings_dict[data['namekey']] = np.array(data['encodings'])

    cap = cv2.VideoCapture(0)
    checked = []
    while True:
        success, frame = cap.read()

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                [enc for enc_list in face_encodings_dict.values() for enc in enc_list], face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(face_encodings_dict.keys())[first_match_index // 5]  # Assuming each person has 5 images
                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                if not name in checked and name != "Unknown":
                    # Load criminal info
                    with open(os.path.join(criminal_info_folder, f"{name}.json"), 'r') as f:
                        criminal_info = json.load(f)

                    print(f"Match found: {criminal_info}")



                    # Create and send PDF report
                    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # location = get_current_location()
                    # create_pdf_data(name, location, current_time, criminal_info)

                    print("Face recognized and data saved. The program will now exit.")
                    checked.append(name)
                    # Save recognized image

                attendance(name, frame)


        cv2.imshow("webcam", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exiting program...")
            break
    cap.release()
    cv2.destroyAllWindows()

def attendance(name, img):
    now = datetime.now()
    dnow = now.strftime('%H:%M:%S')

    # Initialize count for the person if not present in the dictionary
    if name not in image_count:
        image_count[name] = 1
    else:
        image_count[name] += 1

    # Save only three images for each person
    if image_count[name] <= 3:
        # Save the recognized face image in the "photo-detect" folder with the correct name
        img_path = os.path.join("photo-detect", f'{name}{image_count[name]}.jpg')

        print(f"Saving {name} image to {img_path}")

        # Save the detected face image as a screenshot
        cv2.imwrite(img_path, img)

        print(f"Image saved successfully.")

        # Create PDF data after saving three images
        if image_count[name] == 3:
            location_info = get_current_location()
            create_pdf_data(name, location_info, dnow)

def get_current_location():
    location = geocoder.ip('me')
    if location:
        return location.address
    else:
        return "Location not available"


def sendMail(pdf_name, useremail):
    sender_email = 'dhurkeshmyself@gmail.com'
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = useremail
    msg['Subject'] = 'SUSPECT-DETECTED'

    greet_content = """Hello Sir,
            The suspect has been detected recently
            check the pdf that is attached below with the location"""
    body = MIMEText(greet_content, 'plain')
    msg.attach(body)

    with open(pdf_name, 'rb') as fd:
        part = MIMEApplication(fd.read(), basename(pdf_name))
        part['Content-Disposition'] = f'attachment; filename="{basename(pdf_name)}"'
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, 'iifx ufam jpzm orzz')
    server.send_message(msg)
    server.quit()


def create_pdf_data(name, place, time):
    pdf_folder = "pdf-data"
    pdf_path = os.path.join(pdf_folder, f'{name}_data.pdf')
    print(os.getcwd(),name)

    file_path='criminal_info_json\\'+name+'.json'
    with open(file_path,'r') as fd:
        criminal_data = json.load(fd)
    print(criminal_data)

    # Create a PDF document
    pdf = canvas.Canvas(pdf_path, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    # Add name, place, and time to the first page
    pdf.drawString(100, 750, f"Name: {name}")
    pdf.drawString(100, 730, f"Location: {place}")
    pdf.drawString(100, 710, f"Time: {time}")

    # Add additional information
    pdf.drawString(100, 690, f"Full Name: {criminal_data['Full Name']}")
    pdf.drawString(100, 670, f"Age: {criminal_data['Age']}")
    pdf.drawString(100, 650, f"Complaint: {criminal_data['complaint']}")
    pdf.drawString(100, 630, f"Nationality: {criminal_data['Nationality']}")

    for i in range(1, 4):  # Assuming 3 images
        img_path = os.path.join("photo-detect", f'{name}{i}.jpg')
        if os.path.exists(img_path):
            pdf.showPage()
            with Image.open(img_path) as img_pil:
                img_format = img_pil.format
                img_width, img_height = img_pil.size
            x_centered = (letter[0] - img_width) / 2
            y_centered = (letter[1] - img_height) / 2
            pdf.drawInlineImage(img_path, x_centered, y_centered, width=img_width, height=img_height)

    pdf.save()
    print(f"PDF created and saved at {pdf_path}")

    user_email = 'user_email@example.com'
    sendMail(pdf_path, user_email)


if __name__ == "__main__":
    main()
