import cv2
import cv2;
import dlib;
import numpy as np;
import face_recognition;
import matplotlib.pyplot as plt;

IMAGE_NAME = 'image.jpg';
IMAGE_RATIO = '16:9';

detector_hog = dlib.get_frontal_face_detector();

data=np.zeros((15, 128));

for i in range(1, 16) :
    image = cv2.imread('./images/' + str(i) + '.jpeg');
    # image = cv2.resize(image, (640, 480));
    locations = face_recognition.face_locations(image);
    encodings = face_recognition.face_encodings(image, locations);

    if len(encodings) > 0:
        data[i -1] = encodings[0]
    else:
        print(i, 'no face')

print(data)
 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    locations = face_recognition.face_locations(frame)
   
    if locations :
        encodings = face_recognition.face_encodings(frame, locations)
        if encodings :
            # results = face_recognition.compare_faces(data, encodings[0], tolerance=0.5)
            results = np.linalg.norm(encodings[0] - data, axis=1);
            print(results);

            for i in range(0, len(results)) :
                if results[i] < 0.4 :
                    print(i, ': same');
            
        top, right, bottom, left = locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2, lineType=cv2.LINE_AA)


    # Display the frame
    cv2.imshow('Camera', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Destroy all windows
cv2.destroyAllWindows()