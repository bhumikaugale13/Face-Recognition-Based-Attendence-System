Face Recognition Based Attendance System

📌 Project Description

The Face Recognition Based Attendance System is a smart attendance management application that uses face recognition technology to automatically mark student attendance. The system contains two dashboards:

Public Dashboard (Student Panel)

Protected Dashboard (Admin Panel)

Students can register themselves, train the face recognition model, and mark attendance. However, attendance will only be recorded if the student is approved by the admin.

🚀 Features
👨‍🎓 Student Dashboard (Public)

Students can perform the following actions:

Register themselves in the system

Upload their face images

Train the face recognition model

Mark attendance using face recognition

Get a message if attendance is already marked for the day

⚠️ Note:
Attendance will be marked only if the admin has approved the student.

🔐 Admin Dashboard (Protected)

The admin panel allows the administrator to manage the system.

Admin functionalities include:

Approve student registrations

Delete student records

Update student information

View all registered students

Download attendance records as a CSV file

Manage attendance data

Logout securely

⚙️ System Workflow

Student registers in the system.

Student uploads face images.

The system trains the face recognition model.

Admin reviews and approves the student.

Student marks attendance using face recognition.

System checks:

If the student is approved

If attendance is already marked for that day

Attendance is recorded successfully.

Admin can download the attendance report in CSV format.

🛠 Technologies Used

Python

Flask

OpenCV

Face Recognition Library

HTML

CSS

JavaScript

CSV (for attendance records)

📂 Project Structure
Face-Recognition-Attendance-System/
│
├── app.py
├── model.py
├── templates/
├── static/
├── dataset/
├── attendance/
└── README.md
▶️ How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/face-recognition-attendance-system.git
2. Install Required Libraries
pip install flask opencv-python face-recognition pandas numpy
3. Run the Application
python app.py

📊 Attendance Export

The admin can download attendance data in CSV format, which can be opened in:

Microsoft Excel

Google Sheets

Any spreadsheet software

🔒 Security

Admin panel is protected

Only approved students can mark attendance

Duplicate attendance for the same day is prevented

👩‍💻 Author
BHUMIKA UGALE
