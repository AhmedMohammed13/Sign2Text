# Import necessary modules
import os  # Operating system interface for file and directory management
import pickle  # Module for serializing and deserializing Python objects
import mediapipe as mp  # MediaPipe for hand detection and landmark processing
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # Matplotlib for plotting (not used in this script)

mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles for landmarks

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data'


data = []  
labels = []  


for dir_ in os.listdir(DATA_DIR):
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Auxiliary list to store normalized landmark coordinates for the current image
        x_ = []  
        y_ = []  

        # Read the image using OpenCV and convert it to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                print("Length of x_:", len(x_))
                print("Length of y_:", len(y_))

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data1.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
