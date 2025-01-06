from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os

# Load the pre-trained model and class labels for emotion detection
emotion_classifier = load_model("C:\\Users\\ssava\\OneDrive\\Pulpit\\3 kursas\\1 pusmetis\\Phyton programavimo kalba\\Praktines\\Emotion_Detection.h5")
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the pre-trained model for age estimation
age_classifier = load_model("C:\\Users\\ssava\\OneDrive\\Pulpit\\3 kursas\\1 pusmetis\\Phyton programavimo kalba\\Praktines\\Age_model.h5")

# Create a directory to store images with emotions and ages
output_directory = "emotions_and_ages_images"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

face_classifier = cv2.CascadeClassifier("C:\\Users\\ssava\\OneDrive\\Pulpit\\3 kursas\\1 pusmetis\\Phyton programavimo kalba\\Praktines\\haarcascade_frontalface_default.xml")

class_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

# Function to process an image
def process_image(image_path):
    frame = cv2.imread(image_path)
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the ROI to (256, 256, 3)
        roi_gray_resized = cv2.resize(roi_gray, (256, 256), interpolation=cv2.INTER_AREA)
        roi_gray_resized = cv2.cvtColor(roi_gray_resized, cv2.COLOR_GRAY2BGR)

        if np.sum([roi_gray_resized]) != 0:
            roi = roi_gray_resized.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction on the ROI, then lookup the class
            preds = age_classifier.predict(roi)[0]
            age_label = class_labels[np.argmax(preds)]

            # Make an emotion prediction
            roi_gray_emotion = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_emotion = roi_gray_emotion.astype('float') / 255.0
            roi_emotion = img_to_array(roi_emotion)
            roi_emotion = np.expand_dims(roi_emotion, axis=0)
            emotion_preds = emotion_classifier.predict(roi_emotion)[0]
            emotion_label = emotion_labels[emotion_preds.argmax()]

            label_position = (x, y)
            cv2.putText(frame, f"Emotion: {emotion_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Save the image with age and emotion
            file_name = os.path.join(output_directory, f"emotion_{emotion_label}.jpg")
            cv2.imwrite(file_name, frame)

    cv2.imshow('Emotion Detector', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process video from camera
def process_video_from_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]

            # Resize the ROI to (256, 256, 3)
            roi_gray_resized = cv2.resize(roi_gray, (256, 256), interpolation=cv2.INTER_AREA)
            roi_gray_resized = cv2.cvtColor(roi_gray_resized, cv2.COLOR_GRAY2BGR)

            if np.sum([roi_gray_resized]) != 0:
                roi = roi_gray_resized.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction on the ROI, then lookup the class
                preds = age_classifier.predict(roi)[0]
                age_label = class_labels[np.argmax(preds)]

                # Make an emotion prediction
                roi_gray_emotion = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi_emotion = roi_gray_emotion.astype('float') / 255.0
                roi_emotion = img_to_array(roi_emotion)
                roi_emotion = np.expand_dims(roi_emotion, axis=0)
                emotion_preds = emotion_classifier.predict(roi_emotion)[0]
                emotion_label = emotion_labels[emotion_preds.argmax()]

                label_position = (x, y)
                cv2.putText(frame, f"Emotion: {emotion_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Save the image with age and emotion
                file_name = os.path.join(output_directory, f"emotion_{emotion_label}.jpg")
                cv2.imwrite(file_name, frame)

        cv2.imshow('Emotion 0Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main program
print("Choose an option:")
print("1. Process video from camera")
print("2. Process image from file")
option = input("Enter the option (1 or 2): ")

if option == '1':
    process_video_from_camera()
elif option == '2':
    image_file = input("Enter the path to the image file: ")
    process_image(image_file)
else:
    print("Invalid option")