#Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.screenmanager import ScreenManager, Screen

import cv2
import tensorflow as tf
from layers import L1Dist
import os 
import numpy as np
import json
from datetime import datetime

# Camera Screen for face verification
class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        
        #Main layout component
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text='Verify',on_press=self.verify ,size_hint=(1,.1))
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1,.1))

        #add items to layout
        layout = BoxLayout(orientation= 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)
        
        self.add_widget(layout)

        #Load tensorflow/keras model
        try:
            self.model = tf.keras.models.load_model('siamesemodelV2.h5', custom_objects={'L1Dist':L1Dist})
        except Exception as e:
            Logger.error(f"Error loading model: {e}")
            self.verification_label.text = 'Model loading error'
            return

        #setup video capture device
        try:
            # Try different camera backends for Windows compatibility
            self.capture = None
            
            # Try DirectShow backend first (usually more reliable on Windows)
            try:
                self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if self.capture.isOpened():
                    Logger.info("Camera opened with DirectShow backend")
                else:
                    self.capture = None
            except:
                pass
            
            # Fallback to default backend
            if self.capture is None:
                self.capture = cv2.VideoCapture(0)
                if self.capture.isOpened():
                    Logger.info("Camera opened with default backend")
                else:
                    self.capture = None
            
            # Final check
            if self.capture is None or not self.capture.isOpened():
                Logger.error("Could not open camera with any backend")
                self.verification_label.text = 'Camera initialization error'
                return
                
            # Set camera properties for better performance
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
        except Exception as e:
            Logger.error(f"Error initializing camera: {e}")
            self.verification_label.text = 'Camera error'
            return
            
        Clock.schedule_interval(self.update, 1.0/33.0)

    #Run continuously to get web cam feed
    def update(self, *args):
        # Check if capture is available
        if not hasattr(self, 'capture') or self.capture is None:
            return
            
        try:
            #Read frame from openCv
            ret, frame = self.capture.read()
            
            # Check if frame was successfully captured
            if not ret or frame is None:
                return

            #cut down frame to 250x250px
            frame = frame[120:120+250,200:200+250,:] 

            #Flip horizontall and convert image to texture
            buf = cv2.flip(frame, 0).tostring()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture
            
        except Exception as e:
            Logger.warning(f"Camera frame read error: {e}")
            # Try to reinitialize camera if errors persist
            pass

    def preprocess(self, file_path):
        #read in image from file path
        byte_img = tf.io.read_file(file_path)
        #load in the image
        img = tf.io.decode_jpeg(byte_img)
        #preprocessing step - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        #Scale image to be between 0 and 1
        img = img / 255.0

        return img
    
    #verification function to verify person
    def verify(self, *args):

        #specify threshold
        detection_threshold = 0.5
        verification_threshold = 0.8

        #Capyure input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        
        # Check if frame was successfully captured
        if not ret or frame is None:
            self.verification_label.text = 'Camera Error - No frame captured'
            return [], False
            
        frame = frame[120:120+250,200:200+250,:] 
        cv2.imwrite(SAVE_PATH, frame)

        #Build result array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            #Make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        #Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        #Verification threshold: Proportion of positive prediction / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        #Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        #Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))

        # Redirect to welcome screen if verified
        if verified:
            self.manager.current = 'welcome'

        return results, verified
    
    def cleanup_camera(self):
        """Clean up camera resources"""
        try:
            if hasattr(self, 'capture') and self.capture is not None:
                self.capture.release()
                self.capture = None
        except Exception as e:
            Logger.error(f"Error cleaning up camera: {e}")

# Welcome Screen for successful verification with live recognition
class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)
        
        # Initialize face database
        self.face_database = self.load_face_database()
        self.training_mode = False
        self.training_name = ""
        self.training_count = 0
        self.max_training_images = 10
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Top section with camera and controls
        top_section = BoxLayout(orientation='horizontal', size_hint=(1, 0.8), spacing=10)
        
        # Camera section
        camera_layout = BoxLayout(orientation='vertical', size_hint=(0.7, 1))
        self.camera_label = Label(text='Live Face Recognition', font_size='20sp', size_hint=(1, 0.1))
        self.live_cam = Image(size_hint=(1, 0.9))
        camera_layout.add_widget(self.camera_label)
        camera_layout.add_widget(self.live_cam)
        
        # Control panel
        control_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing=10)
        
        # Status label
        self.status_label = Label(
            text='Ready for Recognition',
            font_size='16sp',
            text_size=(None, None),
            halign='center',
            size_hint=(1, 0.2)
        )
        
        # Training section
        training_section = BoxLayout(orientation='vertical', size_hint=(1, 0.4), spacing=5)
        training_title = Label(text='Train New Face:', font_size='18sp', size_hint=(1, 0.2))
        
        # Name input (simplified - using button for demo)
        self.name_input_label = Label(text='Click to set name', font_size='14sp', size_hint=(1, 0.2))
        name_button = Button(text='Set Name', size_hint=(1, 0.2), on_press=self.set_training_name)
        
        self.train_button = Button(
            text='Start Training',
            font_size='16sp',
            size_hint=(1, 0.2),
            on_press=self.toggle_training,
            disabled=True
        )
        
        self.training_progress = Label(text='', font_size='12sp', size_hint=(1, 0.2))
        
        training_section.add_widget(training_title)
        training_section.add_widget(self.name_input_label)
        training_section.add_widget(name_button)
        training_section.add_widget(self.train_button)
        training_section.add_widget(self.training_progress)
        
        # Control buttons
        button_section = BoxLayout(orientation='vertical', size_hint=(1, 0.4), spacing=5)
        
        # Database status button
        status_button = Button(
            text='Show Database',
            font_size='14sp',
            size_hint=(1, 0.2),
            on_press=self.show_database_status
        )
        
        clear_button = Button(
            text='Clear Database',
            font_size='14sp',
            size_hint=(1, 0.2),
            on_press=self.clear_database
        )
        
        back_button = Button(
            text='Back to Camera',
            font_size='16sp',
            size_hint=(1, 0.2),
            on_press=self.go_back_to_camera
        )
        
        button_section.add_widget(status_button)
        button_section.add_widget(clear_button)
        button_section.add_widget(back_button)
        
        # Add to control panel
        control_panel.add_widget(self.status_label)
        control_panel.add_widget(training_section)
        control_panel.add_widget(button_section)
        
        # Add to top section
        top_section.add_widget(camera_layout)
        top_section.add_widget(control_panel)
        
        # Recognition results section
        results_section = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        results_title = Label(text='Recognized Faces:', font_size='16sp', size_hint=(1, 0.3))
        self.results_label = Label(
            text='No faces detected',
            font_size='14sp',
            text_size=(None, None),
            halign='left',
            valign='top',
            size_hint=(1, 0.7)
        )
        
        results_section.add_widget(results_title)
        results_section.add_widget(self.results_label)
        
        # Add all sections to main layout
        main_layout.add_widget(top_section)
        main_layout.add_widget(results_section)
        
        self.add_widget(main_layout)
        
        # Load model and setup camera
        try:
            self.model = tf.keras.models.load_model('siamesemodelV2.h5', custom_objects={'L1Dist':L1Dist})
        except Exception as e:
            Logger.error(f"Error loading model: {e}")
            self.status_label.text = 'Model loading error'
            return
            
        try:
            # Try different camera backends for Windows compatibility
            self.capture = None
            
            # Try DirectShow backend first (usually more reliable on Windows)
            try:
                self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if self.capture.isOpened():
                    Logger.info("Welcome screen camera opened with DirectShow backend")
                else:
                    self.capture = None
            except:
                pass
            
            # Fallback to default backend
            if self.capture is None:
                self.capture = cv2.VideoCapture(0)
                if self.capture.isOpened():
                    Logger.info("Welcome screen camera opened with default backend")
                else:
                    self.capture = None
            
            # Final check
            if self.capture is None or not self.capture.isOpened():
                Logger.error("Could not open camera with any backend")
                self.status_label.text = 'Camera initialization error'
                return
                
            # Set camera properties for better performance
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 20)  # Lower FPS for recognition
            
        except Exception as e:
            Logger.error(f"Error initializing camera: {e}")
            self.status_label.text = 'Camera error'
            return
        
        # Face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            Logger.error(f"Error loading face cascade: {e}")
            self.status_label.text = 'Face detector error'
            return
        
        # Start camera update
        Clock.schedule_interval(self.update_live_recognition, 1.0/10.0)  # 10 FPS for recognition
    
    def load_face_database(self):
        """Load existing face database or create new one"""
        db_path = os.path.join('application_data', 'face_database.json')
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_face_database(self):
        """Save face database to file"""
        db_path = os.path.join('application_data', 'face_database.json')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, 'w') as f:
            json.dump(self.face_database, f)
    
    def set_training_name(self, *args):
        """Set name for training with better name management"""
        # List of available names (you can modify this list)
        available_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
        
        # Get names already in database
        existing_names = list(self.face_database.keys())
        
        # Find next available name or cycle through
        if len(existing_names) < len(available_names):
            # Use first available name not in database
            for name in available_names:
                if name not in existing_names:
                    self.training_name = name
                    break
        else:
            # Cycle through names for retraining
            current_index = len(existing_names) % len(available_names)
            self.training_name = available_names[current_index]
        
        self.name_input_label.text = f'Training: {self.training_name}'
        self.train_button.disabled = False
        
        # Show status about this person
        if self.training_name in self.face_database:
            existing_count = len(self.face_database[self.training_name]['images'])
            self.status_label.text = f'{self.training_name} has {existing_count} images. Ready to add more.'
        else:
            self.status_label.text = f'Ready to train new person: {self.training_name}'
    
    def toggle_training(self, *args):
        """Start/stop training mode"""
        if not self.training_mode:
            # Start training
            self.training_mode = True
            self.training_count = 0
            self.train_button.text = 'Stop Training'
            self.status_label.text = f'Training {self.training_name}... Look at camera!'
            self.training_progress.text = f'Progress: 0/{self.max_training_images}'
            
            # Reset training counter if retraining existing person
            if self.training_name in self.face_database:
                # Continue from existing count or restart
                existing_count = len(self.face_database[self.training_name]['images'])
                if existing_count >= self.max_training_images:
                    self.training_count = 0  # Restart training
                    self.face_database[self.training_name]['images'] = []  # Clear old images
                else:
                    self.training_count = existing_count  # Continue from where we left off
        else:
            # Stop training
            self.training_mode = False
            self.train_button.text = 'Start Training'
            
            if self.training_count > 0:
                self.status_label.text = f'Training paused for {self.training_name}'
                self.save_face_database()
            else:
                self.status_label.text = 'Training cancelled'
    
    def clear_database(self, *args):
        """Clear the face database"""
        self.face_database = {}
        self.save_face_database()
        self.status_label.text = 'Database cleared'
        
        # Clear training images directory
        training_dir = os.path.join('application_data', 'training_images')
        if os.path.exists(training_dir):
            for file in os.listdir(training_dir):
                try:
                    os.remove(os.path.join(training_dir, file))
                except Exception as e:
                    Logger.error(f"Error removing file {file}: {e}")
    
    def show_database_status(self, *args):
        """Show current database status"""
        if not self.face_database:
            self.status_label.text = 'Database is empty'
            return
        
        status_lines = []
        for name, data in self.face_database.items():
            image_count = len(data['images'])
            completed = data.get('training_completed', False)
            status = "✓" if completed else "..."
            status_lines.append(f"{name}: {image_count} images {status}")
        
        status_text = f"Database ({len(self.face_database)} people):\n" + "\n".join(status_lines)
        self.status_label.text = status_text[:200] + "..." if len(status_text) > 200 else status_text
    
    def preprocess_face(self, face_img):
        """Preprocess face image for recognition"""
        # Resize to model input size
        face_resized = cv2.resize(face_img, (100, 100))
        # Convert to RGB and normalize
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb / 255.0
        return face_normalized
    
    def recognize_face(self, face_img):
        """Recognize a face using the siamese model"""
        if not self.face_database:
            return "Unknown", 0.0
        
        preprocessed_face = self.preprocess_face(face_img)
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, face_data in self.face_database.items():
            similarities = []
            
            for reference_path in face_data['images']:
                if os.path.exists(reference_path):
                    try:
                        # Load and preprocess reference image
                        ref_img = cv2.imread(reference_path)
                        if ref_img is not None:
                            ref_preprocessed = self.preprocess_face(ref_img)
                            
                            # Prepare for siamese model (convert to TensorFlow tensors)
                            input_tensor = tf.convert_to_tensor(np.expand_dims(preprocessed_face, axis=0), dtype=tf.float32)
                            ref_tensor = tf.convert_to_tensor(np.expand_dims(ref_preprocessed, axis=0), dtype=tf.float32)
                            
                            # Get similarity score (closer to 0 means more similar for L1 distance)
                            prediction = self.model.predict([input_tensor, ref_tensor], verbose=0)[0][0]
                            
                            # Convert L1 distance to similarity score (invert and normalize)
                            similarity = max(0, 1.0 - prediction)  # Convert distance to similarity
                            similarities.append(similarity)
                    except Exception as e:
                        Logger.error(f"Error processing reference image {reference_path}: {e}")
                        continue
            
            if similarities:
                # Use average similarity for this person
                avg_similarity = np.mean(similarities)
                
                # Update best match if this person has higher average similarity
                if avg_similarity > best_similarity and avg_similarity > 0.6:  # Adjusted threshold
                    best_similarity = avg_similarity
                    best_match = name
        
        return best_match, best_similarity
    
    def save_training_face(self, face_img):
        """Save face image for training"""
        if not self.training_name:
            return False
        
        # Create directories
        training_dir = os.path.join('application_data', 'training_images')
        os.makedirs(training_dir, exist_ok=True)
        
        # Save image with better quality and consistent naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.training_name}_{self.training_count:03d}_{timestamp}.jpg"
        filepath = os.path.join(training_dir, filename)
        
        # Resize face to consistent size before saving
        face_resized = cv2.resize(face_img, (100, 100))
        success = cv2.imwrite(filepath, face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            Logger.error(f"Failed to save training image: {filepath}")
            return False
        
        # Update database
        if self.training_name not in self.face_database:
            self.face_database[self.training_name] = {
                'images': [],
                'created_date': datetime.now().isoformat(),
                'training_completed': False
            }
        
        self.face_database[self.training_name]['images'].append(filepath)
        
        self.training_count += 1
        self.training_progress.text = f'Progress: {self.training_count}/{self.max_training_images}'
        
        # Stop training when enough images collected
        if self.training_count >= self.max_training_images:
            self.face_database[self.training_name]['training_completed'] = True
            self.face_database[self.training_name]['completion_date'] = datetime.now().isoformat()
            self.toggle_training()
            
            # Validate training data
            self.validate_training_data(self.training_name)
        
        return True
    
    def validate_training_data(self, person_name):
        """Validate that training data is good quality"""
        if person_name not in self.face_database:
            return False
        
        images = self.face_database[person_name]['images']
        valid_images = []
        
        for img_path in images:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None and img.shape[0] >= 50 and img.shape[1] >= 50:
                    valid_images.append(img_path)
        
        # Update database with only valid images
        self.face_database[person_name]['images'] = valid_images
        self.face_database[person_name]['valid_image_count'] = len(valid_images)
        
        Logger.info(f"Training validation for {person_name}: {len(valid_images)} valid images")
        
        if len(valid_images) >= 5:  # Minimum required for good recognition
            self.status_label.text = f'Training completed: {person_name} ({len(valid_images)} images)'
            return True
        else:
            self.status_label.text = f'Training incomplete: Need more images for {person_name}'
            return False
    
    def update_live_recognition(self, *args):
        """Update camera feed with live face recognition"""
        # Check if capture is available
        if not hasattr(self, 'capture') or self.capture is None:
            return
            
        try:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                return
            
            # Check frame dimensions to avoid errors
            if frame.shape[0] < 50 or frame.shape[1] < 50:
                return
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            recognized_faces = []
            
            for (x, y, w, h) in faces:
                # Extract face region with bounds checking
                face_roi = frame[y:y+h, x:x+w]
                
                # Check if face region is valid and large enough
                if face_roi.size == 0 or face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    continue
                
                # Only use well-sized faces (not too small or too large)
                if w < 80 or h < 80 or w > 300 or h > 300:
                    continue
                
                if self.training_mode:
                    # Save face for training with some delay to avoid duplicate saves
                    try:
                        # Add a small delay between saves (save every 5th frame approximately)
                        import time
                        current_time = time.time()
                        if not hasattr(self, 'last_save_time'):
                            self.last_save_time = 0
                        
                        if current_time - self.last_save_time > 0.5:  # Save every 0.5 seconds
                            success = self.save_training_face(face_roi)
                            if success:
                                self.last_save_time = current_time
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)  # Yellow for training
                        cv2.putText(frame, f'Training: {self.training_name}', (x, y-15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, f'{self.training_count}/{self.max_training_images}', (x, y+h+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    except Exception as e:
                        Logger.error(f"Error saving training face: {e}")
                else:
                    # Recognize face
                    try:
                        name, confidence = self.recognize_face(face_roi)
                        recognized_faces.append(f"{name} ({confidence:.2f})")
                        
                        # Draw rectangle and name
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f'{name} {confidence:.2f}', (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        Logger.error(f"Error recognizing face: {e}")
                        # Draw red rectangle for error cases
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Error', (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Update recognition results
            if recognized_faces:
                self.results_label.text = '\n'.join(recognized_faces)
            else:
                self.results_label.text = 'No faces detected' if not self.training_mode else 'Training mode active'
            
            # Convert frame to texture and display
            buf = cv2.flip(frame, 0).tostring()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.live_cam.texture = img_texture
            
        except Exception as e:
            Logger.warning(f"Live recognition camera error: {e}")
            # Continue running even if frame capture fails
            pass
    
    def go_back_to_camera(self, *args):
        # Stop the camera when leaving
        try:
            if hasattr(self, 'capture') and self.capture is not None:
                self.capture.release()
                self.capture = None
        except Exception as e:
            Logger.error(f"Error releasing camera: {e}")
        
        self.manager.current = 'camera'
    
    def reinitialize_camera(self):
        """Reinitialize camera if it fails"""
        try:
            if hasattr(self, 'capture') and self.capture is not None:
                self.capture.release()
            
            # Try DirectShow backend first
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                # Fallback to default
                self.capture = cv2.VideoCapture(0)
            
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv2.CAP_PROP_FPS, 20)
                return True
            else:
                return False
        except Exception as e:
            Logger.error(f"Camera reinitialization failed: {e}")
            return False

#Build app with screen management
class CamApp(App):
    def build(self):
        # Create screen manager
        sm = ScreenManager()
        
        # Create screens
        camera_screen = CameraScreen(name='camera')
        welcome_screen = WelcomeScreen(name='welcome')
        
        # Add screens to manager
        sm.add_widget(camera_screen)
        sm.add_widget(welcome_screen)
        
        # Set initial screen
        sm.current = 'camera'
        
        return sm

if __name__ == '__main__':
    CamApp().run() 