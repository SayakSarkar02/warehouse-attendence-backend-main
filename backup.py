from flask import Flask, request, jsonify
import base64
import os
import numpy as np
from PIL import Image
import cv2
import face_recognition
import imageio.v2 as imageio
import flask_cors
import io
import csv
from datetime import datetime
from flask import send_from_directory

app = Flask(__name__)
flask_cors.CORS(app)


attendancecvv = "Class 10th A.csv"
studentsfolder = "Class 10th A"
# Load the registered students
known_face_encodings = []
known_face_names = []

for student_dir in os.listdir(studentsfolder):
    student_path = os.path.join(studentsfolder, student_dir)
    if os.path.isdir(student_path):
        name = os.path.basename(student_dir)
        for filename in os.listdir(student_path):
            image_path = os.path.join(student_path, filename)
            image = imageio.imread(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            else:
                print(f"No faces found in the image {image_path}")

os.makedirs(studentsfolder, exist_ok=True)

def update_attendance_csv(student_name, present):
    csv_path = attendancecvv

    # Create the file if it doesn't exist
    if not os.path.exists(csv_path):
        open(csv_path, 'w').close()

    # Open the CSV file in read mode
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        attendance_data = list(reader)

    # Find the row with the given student_name
    student_row = None
    for i, row in enumerate(attendance_data):
        if row[1] == student_name:
            student_row = i
            break

    # If the student is found, update the attendance value
    if student_row is not None:
        current_date = datetime.now().strftime('%Y-%m-%d')
        attendance_data[student_row][0] = current_date
        attendance_data[student_row][2] = 'Present' if present else 'Absent'
    else:
        # If the student is not found, create a new row
        new_row = [datetime.now().strftime('%Y-%m-%d'), student_name, 'Present' if present else 'Absent']
        attendance_data.append(new_row)

    # Open the CSV file in write mode and rewrite the updated data
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if it doesn't exist in the attendance_data list
        if not attendance_data or attendance_data[0] != ['date', 'name', 'attendance']:
            writer.writerow(['date', 'name', 'attendance'])

        writer.writerows(attendance_data)

@app.route('/register', methods=['POST'])
def register_student():
    # Get the name from the form data
    name = request.form.get('name')

    # Create a directory for the new student if it doesn't exist
    student_dir = os.path.join(studentsfolder, name)
    os.makedirs(student_dir, exist_ok=True)

    # Process the 4 images from the form data
    for i in range(5):
        image_data = request.form.get(f'image{i}')

        # Check if the image data is present
        if not image_data:
            return f"No image data found for image{i}", 400

        try:
            # Decode the base64 data
            image_bytes = base64.b64decode(image_data)

            # Save the image to the student's directory
            image_path = os.path.join(student_dir, f'{i}.jpg')
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            for filename in os.listdir(student_dir):
                image_path = os.path.join(student_dir, filename)
                image = imageio.imread(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

        except Exception as e:
            print(f"Error saving image{i}: {e}")
            return f"Error saving image{i}", 500
        
    update_attendance_csv(name, present=False)

    # Return a success message
    return f"Student '{name}' registered successfully with 4 images"

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get the base64-encoded image data from the request
    image_data = request.form.get('image_data').replace("data:image/jpeg;base64,", "")

    # Check if image data is present
    if not image_data:
        return "No image data found in request", 400

    try:
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert PIL Image to numpy array
        image_np = np.array(pil_image)

        # Convert the numpy array to a BGR image
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Check if the image decoding was successful
        if image is None:
            return "Failed to decode image", 400
        
        # Print debug info
        # print(f"Decoded image to BGR image of shape {image.shape}")

    except Exception as e:
        print(f"Error decoding image: {e}")
        return "Error decoding image", 500
    
    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the uploaded image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized_names = []

    # Recognize faces in the uploaded image
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                recognized_names.append(name)
                update_attendance_csv(name, present=True)  # Update attendance CSV
            else:
                recognized_names.append("Unknown")
        else:
            recognized_names.append("Unknown")

    # Return the recognized names as a response
    # print(f"Recognized names: {recognized_names}")
    return {
        "recognized_names": recognized_names
    }

@app.route('/attendance', methods=['GET'])
def get_attendance_data():
    csv_path = attendancecvv

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        return jsonify([])

    attendance_data = []

    # Read the attendance data from the CSV file
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header row
        attendance_data = [dict(zip(header, row)) for row in reader]

    # Get all registered students
    studfolder = os.listdir(studentsfolder)
    registered_names = set([os.path.basename(student_dir) for student_dir in studfolder])

    # Add absent status for registered students not in attendance CSV
    for student_name in registered_names:
        found = False
        for entry in attendance_data:
            if entry['name'] == student_name:
                found = True
                break
        if not found:
            attendance_data.append({'date': '', 'name': f"{student_name}", 'attendance': 'Absent'})

    # Update attendance_data to include the full path for present students
    for entry in attendance_data:
        entry['name'] = f"{studentsfolder}/{entry['name']}"

    return jsonify(attendance_data)

@app.route('/mark-absent', methods=['POST'])
def mark_absent():
    student_name = request.form.get('student_name')

    if not student_name:
        return jsonify({'error': 'Student name is required'}), 400

    csv_path = attendancecvv

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Attendance data not found'}), 404

    # Open the CSV file in read mode
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header row
        attendance_data = list(reader)

    # Open the CSV file in write mode
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write the header row

        # Write attendance data, marking the specified student as absent
        for row in attendance_data:
            if row[1] == student_name:
                row[2] = 'Absent'
            writer.writerow(row)

    return jsonify({'message': 'Attendance updated successfully'}), 200

# Endpoint to initialize or clear attendance data
@app.route('/clearall', methods=['GET'])
def clear():
    csv_path = attendancecvv

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Attendance data not found'}), 404

    # Initialize or clear attendance data
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'name', 'attendance'])  # Write header row
        
        # Get all registered students
        for student_dir in os.listdir(studentsfolder):
            student_name = os.path.basename(student_dir)
            writer.writerow([datetime.now().strftime('%Y-%m-%d'), student_name, 'Absent'])

    return jsonify({'message': 'Attendance data initialized or cleared successfully'}), 200

@app.route('/<string:studentsfolder>/<path:filename>', methods=['GET'])
def serve_student_image(studentsfolder, filename):
    folder_path = os.path.join(studentsfolder)
    return send_from_directory(folder_path, filename)


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True, port=4003, host="0.0.0.0")