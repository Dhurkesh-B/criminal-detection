import cv2
import os
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

# Absolute paths to folders
def run_recognition_program(info_data):
    current_directory = os.getcwd()
    path = os.path.join(current_directory, "ImageAttendance")
    output_folder = os.path.join(current_directory, "RecognizedFaces")
    photo_detect_folder = os.path.join(current_directory, "photo-detect")
    pdf_folder = os.path.join(current_directory, "pdf-data")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(photo_detect_folder):
        os.makedirs(photo_detect_folder)

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    className = []
    encodeKnownList = []

    mylist = os.listdir(path)

    for img in mylist:
        curImg = cv2.imread(os.path.join(path, img))
        className.append(os.path.splitext(img)[0])

        # Encoding faces during initialization
        curImgRGB = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(curImgRGB)[0]
        encodeKnownList.append(enc)

    print("Encoding completed")

    # Dictionary to store the count of images saved for each person
    image_count = {}



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
            img_path = os.path.join(photo_detect_folder, f'{name}{image_count[name]}.jpg')

            print(f"Saving {name} image to {img_path}")

            # Save the detected face image as a screenshot
            cv2.imwrite(img_path, img)

            print(f"Image saved successfully.")

            # Create PDF data after saving three images
            if image_count[name] == 3:
                location_info = get_current_location()
                createpdfdata(name, location_info, dnow)

    def get_current_location():
        # Use the geocoder library to get the location based on the user's IP address
        location = geocoder.ip('me')

        if location:
            # Return the full address obtained from the geocoding service
            return location.address
        else:
            return "Location not available"

    def sendMail(pdf_name, useremail):
        sender_email = 'your_gmail@gmail.com'  # Replace with your email address
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
        server.login(sender_email, 'your_gmail_passkey')  # Replace with your Gmail App Password
        server.send_message(msg)
        server.quit()

    def createpdfdata(name, place, time):
        pdf_path = os.path.join(pdf_folder, f'{name}_data.pdf')

        # Create a PDF document
        pdf = canvas.Canvas(pdf_path, pagesize=letter)

        # Set font
        pdf.setFont("Helvetica", 12)

        # Convert Unicode characters to UTF-8
        name = name.encode('utf-8').decode('utf-8')
        place = place.encode('utf-8').decode('utf-8')
        time = time.encode('utf-8').decode('utf-8')

        # Add name, place, and time to the first page
        pdf.drawString(100, 750, f"Name: {name}")
        pdf.drawString(100, 730, f"Location: {place}")
        pdf.drawString(100, 710, f"Time: {time}")

        # Check if additional information is available in info_data dictionary
        if name in info_data:
            # Add additional information to the first page
            pdf.drawString(100, 690, f"Full Name: {info_data[name]['Full Name']}")
            pdf.drawString(100, 670, f"Age: {info_data[name]['Age']}")
            pdf.drawString(100, 650, f"Complaint: {info_data[name]['complaint']}")
            pdf.drawString(100, 630, f"Nationality: {info_data[name]['Nationality']}")

        for i in range(1, 4):  # Assuming 3 images
            img_path = os.path.join(photo_detect_folder, f'{name}{i}.jpg')

            if os.path.exists(img_path):
                # Add a new page for each image
                pdf.showPage()

                # Open the image using PIL to get its format, width, and height
                with Image.open(img_path) as img_pil:
                    img_format = img_pil.format
                    img_width, img_height = img_pil.size

                # Calculate the position to center the image on the page
                x_centered = (letter[0] - img_width) / 2
                y_centered = (letter[1] - img_height) / 2

                # Add image to the new page, centered
                pdf.drawInlineImage(img_path, x_centered, y_centered, width=img_width, height=img_height)

        # Save the PDF in the "pdf-data" folder
        pdf.save()

        print(f"PDF created and saved at {pdf_path}")

        # Example usage:
        # Replace 'user_email@example.com' with the actual email address of the user
        user_email = 'user_email@example.com'
        sendMail(pdf_path, user_email)

    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurLoc = face_recognition.face_locations(imgS)
        faceCurEncode = face_recognition.face_encodings(imgS, faceCurLoc)

        for faceEncode, faceLoc in zip(faceCurEncode, faceCurLoc):
            matches = face_recognition.compare_faces(encodeKnownList, faceEncode)
            dist = face_recognition.face_distance(encodeKnownList, faceEncode)
            matchIndex = np.argmin(dist)

            if matches[matchIndex]:
                name = className[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                attendance(name, img)

        cv2.imshow("webcam", img)

        # Check if the 'q' key is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exiting program...")
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
