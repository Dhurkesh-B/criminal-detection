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
from dotenv import load_dotenv

face_mark_folder = "face_mark_json"
criminal_info_folder = "criminal_info_json"
load_dotenv()
# Dictionary to keep track of saved images
image_count = {}

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
                face_encodings_dict[data['name']] = np.array(data['encodings'])

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
    sender_email = os.getenv('senderEmail') #Replace it with your gmail
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
    server.login(sender_email, os.getenv('passkey'))
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

    user_email = os.getenv('reciverEmail')
    sendMail(pdf_path, user_email)


if __name__ == "__main__":
    
    run_recognition_program()
