import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import pickle
import time

class FaceRecognitionSystem:
    def __init__(self):
        self.dataset_path = "archive (1)"
        self.faces_dir = os.path.join(self.dataset_path, "Faces", "Faces")
        self.csv_path = os.path.join(self.dataset_path, "Dataset.csv")
        self.model_path = "face_recognition_model.h5"
        self.attendance_file = "attendance.csv"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognition_model = None
        self.attendance_data = []
        self.class_names = []
        self.image_size = (128, 128)
        self.enrolled_students_file = "enrolled_students.pkl"
        self.enrolled_students = self.load_enrolled_students()
        
    def load_enrolled_students(self):
        """Load enrolled students data from file"""
        if os.path.exists(self.enrolled_students_file):
            with open(self.enrolled_students_file, 'rb') as f:
                return pickle.load(f)
        return {}  # Return empty dict if no enrolled students
        
    def save_enrolled_students(self):
        """Save enrolled students data to file"""
        with open(self.enrolled_students_file, 'wb') as f:
            pickle.dump(self.enrolled_students, f)
            
    def enroll_student(self):
        """Register a new student by capturing their face"""
        print("\n=== Student Enrollment ===")
        student_name = input("Enter student name: ")
        
        if student_name in self.enrolled_students:
            print(f"Student {student_name} is already enrolled!")
            return
            
        # Try different camera indices
        camera_indices = [0, 1, 2]
        video_capture = None
        
        for index in camera_indices:
            try:
                print(f"Trying to open camera at index {index}...")
                video_capture = cv2.VideoCapture(index)
                
                # Wait a bit for the camera to initialize
                time.sleep(2)
                
                if video_capture.isOpened():
                    # Set camera properties
                    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    video_capture.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Try to read a frame to confirm camera is working
                    ret, frame = video_capture.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Successfully opened camera at index {index}")
                        break
                    else:
                        print(f"Camera at index {index} opened but failed to read frame")
                        video_capture.release()
                else:
                    print(f"Failed to open camera at index {index}")
            except Exception as e:
                print(f"Error with camera at index {index}: {e}")
                if video_capture is not None:
                    video_capture.release()
                continue
        
        if video_capture is None or not video_capture.isOpened():
            print("Error: Could not open any camera. Please check your camera connection.")
            return
            
        print("\nPosition your face in the camera and press SPACE to capture")
        print("Press 'q' to quit")
        
        face_samples = []
        sample_count = 0
        required_samples = 5  # Number of face samples to capture
        
        while sample_count < required_samples:
            ret, frame = video_capture.read()
            if not ret or frame is None or frame.size == 0:
                print("Failed to grab frame, retrying...")
                continue
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Draw face detection box and instructions
            for (x, y, w, h) in faces:
                # Draw box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw sample count
                cv2.putText(frame, f"Sample {sample_count + 1}/{required_samples}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Add instructions to the frame
            cv2.putText(frame, "Press SPACE to capture", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Student Enrollment', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and len(faces) > 0:  # Space bar pressed and face detected
                # Get the largest face (assuming it's the closest/most prominent)
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = self.preprocess_face(face_roi)
                face_samples.append(processed_face[0])  # Remove batch dimension
                sample_count += 1
                print(f"Captured sample {sample_count}/{required_samples}")
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        if sample_count == required_samples:
            # Store the face samples
            self.enrolled_students[student_name] = np.array(face_samples)
            self.save_enrolled_students()
            print(f"\nSuccessfully enrolled {student_name}!")
        else:
            print("\nEnrollment cancelled or incomplete.")
            
    def train_model(self):
        """Train a deep learning model for face recognition using MobileNetV2"""
        if not self.enrolled_students:
            print("No enrolled students found. Please enroll students first.")
            return
            
        print("Training face recognition model...")
        
        # Prepare training data from enrolled students
        X_enrolled = []
        y_enrolled = []
        self.class_names = list(self.enrolled_students.keys())
        num_classes = len(self.class_names)
        
        print(f"Preparing training data for {num_classes} enrolled students...")
        
        for idx, (name, faces) in enumerate(self.enrolled_students.items()):
            print(f"Processing {name} with {len(faces)} face samples")
            for face in faces:
                # Add original face
                X_enrolled.append(face)
                y_enrolled.append(idx)  # Use index as class label
                
                # Add horizontally flipped face
                X_enrolled.append(cv2.flip(face, 1))
                y_enrolled.append(idx)
                
                # Add slightly rotated faces
                for angle in [-15, -10, -5, 5, 10, 15]:
                    rows, cols = face.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                    rotated = cv2.warpAffine(face, M, (cols, rows))
                    X_enrolled.append(rotated)
                    y_enrolled.append(idx)
                
                # Add brightness variations
                for alpha in [0.8, 1.2]:  # brightness factor
                    brightened = cv2.convertScaleAbs(face, alpha=alpha, beta=0)
                    X_enrolled.append(brightened)
                    y_enrolled.append(idx)
                
        X_enrolled = np.array(X_enrolled)
        y_enrolled = np.array(y_enrolled)
        
        print(f"Augmented enrolled samples to {len(X_enrolled)} images")
        
        # Load the existing dataset for negative samples
        print("\nLoading existing dataset for negative samples...")
        try:
            df = pd.read_csv(self.csv_path)
            dataset_classes = df['label'].unique().tolist()
            print(f"Found {len(dataset_classes)} classes in the dataset")
            
            X_dataset = []
            y_dataset = []
            max_neg_samples = len(X_enrolled)  # Balance the dataset
            
            for idx, row in df.iterrows():
                if len(X_dataset) >= max_neg_samples:
                    break
                    
                image_name = row['id']
                image_path = os.path.join(self.faces_dir, image_name)
                
                if os.path.exists(image_path):
                    try:
                        img = load_img(image_path, target_size=self.image_size)
                        img_array = img_to_array(img)
                        img_array = img_array / 255.0
                        X_dataset.append(img_array)
                        y_dataset.append(num_classes)  # Use num_classes as the "unknown" class
                        
                        # Add augmented negative samples
                        X_dataset.append(cv2.flip(img_array, 1))
                        y_dataset.append(num_classes)
                        
                        for angle in [-15, 15]:
                            rows, cols = img_array.shape[:2]
                            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                            rotated = cv2.warpAffine(img_array, M, (cols, rows))
                            X_dataset.append(rotated)
                            y_dataset.append(num_classes)
                            
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")
            
            X_dataset = np.array(X_dataset)
            y_dataset = np.array(y_dataset)
            print(f"Loaded {len(X_dataset)} negative samples from the dataset")
            
            # Combine datasets
            X = np.concatenate([X_enrolled, X_dataset])
            y = np.concatenate([y_enrolled, y_dataset])
            
            # Convert labels to one-hot encoding
            y = tf.keras.utils.to_categorical(y, num_classes=num_classes + 1)
            
            # Shuffle the data
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create base model using MobileNetV2
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.image_size[0], self.image_size[1], 3)
            )
            
            # Freeze the base model layers
            base_model.trainable = False
            
            # Create the model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(num_classes + 1, activation='softmax')  # +1 for unknown class
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train the model
            print("\nTraining the model...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Save the model
            model.save(self.model_path)
            self.face_recognition_model = model
            
            print("\nModel training completed!")
            print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2f}")
            
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
        
    def load_attendance_data(self):
        """Load existing attendance data"""
        if os.path.exists(self.attendance_file):
            self.attendance_data = pd.read_csv(self.attendance_file)
        else:
            self.attendance_data = pd.DataFrame(columns=['Name', 'Date', 'Time'])
            
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')
        
        # Check if attendance already marked for today
        today_attendance = self.attendance_data[
            (self.attendance_data['Name'] == name) & 
            (self.attendance_data['Date'] == date)
        ]
        
        if today_attendance.empty:
            new_attendance = pd.DataFrame({
                'Name': [name],
                'Date': [date],
                'Time': [time]
            })
            self.attendance_data = pd.concat([self.attendance_data, new_attendance], ignore_index=True)
            self.attendance_data.to_csv(self.attendance_file, index=False)
            print(f"Marked attendance for {name}")
            
    def preprocess_face(self, face_img):
        """Preprocess face image for recognition"""
        # Resize to match training data
        face_img = cv2.resize(face_img, self.image_size)
        # Convert to RGB (OpenCV uses BGR)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Normalize
        face_img = face_img / 255.0
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
            
    def run_recognition(self):
        """Run real-time face recognition"""
        if not self.face_recognition_model:
            print("Model not trained. Please train the model first.")
            return
            
        print("Initializing camera...")
        
        # Try different camera indices with more detailed error handling
        camera = None
        for camera_index in [0, 1, 2]:
            try:
                print(f"Attempting to open camera {camera_index}...")
                camera = cv2.VideoCapture(camera_index)
                
                # Wait longer for camera initialization
                time.sleep(3)
                
                if not camera.isOpened():
                    print(f"Failed to open camera {camera_index}")
                    continue
                
                # Try multiple times to read a frame
                for _ in range(5):
                    ret, frame = camera.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Successfully read frame from camera {camera_index}")
                        print(f"Frame shape: {frame.shape}")
                        
                        # Set camera properties after successful read
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
                        camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness
                        
                        # Display test frame to verify camera
                        cv2.imshow('Camera Test', frame)
                        cv2.waitKey(1000)  # Show test frame for 1 second
                        cv2.destroyWindow('Camera Test')
                        break
                    else:
                        print(f"Failed to read frame from camera {camera_index}, attempt {_ + 1}/5")
                        time.sleep(1)
                
                if ret:  # If we successfully got a frame, use this camera
                    break
                else:
                    print(f"Could not read frames from camera {camera_index}")
                    camera.release()
                    
            except Exception as e:
                print(f"Error initializing camera {camera_index}: {str(e)}")
                if camera is not None:
                    camera.release()
                    camera = None
        
        if camera is None or not camera.isOpened():
            print("Failed to initialize any camera. Please check your camera connection and permissions.")
            return
            
        # Initialize face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Starting face recognition...")
        print("Press 'q' to quit")
        
        # Initialize attendance tracking
        attendance_marked = set()
        last_attendance_time = {}  # Track last attendance time for each person
        
        # Create a named window
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', 800, 600)
        
        frame_count = 0
        try:
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("Failed to grab frame. Retrying...")
                    frame_count += 1
                    if frame_count > 10:  # After 10 failed attempts, try to reinitialize camera
                        print("Too many failed attempts. Trying to reinitialize camera...")
                        camera.release()
                        camera = cv2.VideoCapture(0)
                        frame_count = 0
                    continue
                
                # Reset frame count on successful capture
                frame_count = 0
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Preprocess face for recognition
                    face_roi = self.preprocess_face(face_roi)
                    
                    # Get prediction
                    prediction = self.face_recognition_model.predict(face_roi, verbose=0)[0]
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Get predicted class and confidence
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[predicted_class] * 100
                    
                    # Check if face is recognized with high confidence
                    if confidence > 70:  # Increased threshold for better accuracy
                        if predicted_class < len(self.class_names):  # Check if it's a known student
                            name = self.class_names[predicted_class]
                            
                            # Draw name and confidence
                            cv2.putText(frame, f"{name} ({confidence:.1f}%)", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Mark attendance if not already marked
                            current_time = datetime.now()
                            if name not in attendance_marked:
                                self.mark_attendance(name)
                                attendance_marked.add(name)
                                last_attendance_time[name] = current_time
                                print(f"Attendance marked for {name}")
                            elif (current_time - last_attendance_time[name]).total_seconds() > 300:  # 5 minutes
                                self.mark_attendance(name)
                                last_attendance_time[name] = current_time
                                print(f"Attendance marked for {name} (after 5 minutes)")
                        else:
                            cv2.putText(frame, "Unknown", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Add a status message to the frame
                cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Face Recognition', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error during recognition: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()
        
    def run(self):
        """Run the complete system"""
        while True:
            print("\n=== Face Recognition Attendance System ===")
            print("1. Enroll New Student")
            print("2. Train Model")
            print("3. Start Recognition")
            print("4. View Enrolled Students")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                self.enroll_student()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                self.load_attendance_data()
                self.run_recognition()
            elif choice == '4':
                if self.enrolled_students:
                    print("\nEnrolled Students:")
                    for name in self.enrolled_students.keys():
                        print(f"- {name}")
                else:
                    print("\nNo students enrolled yet.")
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run() 



# OWNED BY MANISH KUMAR https://github.com/manish8557/attendance_automation/edit/main/face_recognition_system.py
