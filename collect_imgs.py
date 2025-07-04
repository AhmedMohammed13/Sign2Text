from flask import Blueprint  # Flask Blueprint for creating modular code
import os  
import cv2  

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 42
dataset_size = 250
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print('Collecting data for class {}'.format(j))
    while True:
        ret, frame = cap.read()  # Capture a frame from the video feed
        # Display instructions on the video feed
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  

        # Break the loop when the user presses 'q'
        if cv2.waitKey(25) == ord('q'):
            break
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  
        print("Captured frame shape:", frame.shape)  
        cv2.imshow('frame', frame)  
        cv2.waitKey(25)  
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1  
cap.release()
cv2.destroyAllWindows()
