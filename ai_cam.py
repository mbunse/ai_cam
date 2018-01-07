import numpy as np
import cv2
from facenet.src.align import detect_face
import tensorflow as tf
import os.path
from tensorflow.python.platform import gfile

#Supress warning about tensorflow not compiled for current CPU
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

THRESHOLD = [0.7, 0.8, 0.8]
MINSIZE=50
FACTOR = 0.709
cap = cv2.VideoCapture(0)


sess = tf.Session()

pnet_fun, rnet_fun, onet_fun = detect_face.create_mtcnn(sess, model_path=None)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    total_boxes, points = detect_face.detect_face(rgb_frame,
        minsize=MINSIZE, pnet=pnet_fun, rnet=rnet_fun, onet=onet_fun,
        threshold=THRESHOLD, factor=FACTOR)
    #face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (right, top, left, bottom, prob) in total_boxes:
        # See if the face is a match for the known face(s)
        #match = face_recognition.compare_faces([obama_face_encoding], face_encoding)

        # Draw a box around the face
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()