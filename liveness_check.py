import os, sys

import cv2
import dlib
import torch
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_embedding(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    face_tensor = mtcnn(face_pil)

    if face_tensor is None:
        return None

    emb = resnet(face_tensor.unsqueeze(0).to(device))
    return emb.detach().cpu().numpy()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./Models/shape_predictor_68_face_landmarks.dat")

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3
blink_counter = 0
liveness_verified = False
image_captured = False

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                print("Blink detected!")
                liveness_verified = True
            blink_counter = 0

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y = max(0, x), max(0, y)
        face_crop = frame[y:y+h, x:x+w]
        
        if liveness_verified and not image_captured:
            if face_crop.size != 0:
                embedding = get_embedding(face_crop)

                if embedding is not None:
                    print("Embedding shape:", embedding.shape)
                    print("Embedding generated successfully!")
                    image_captured = True
                else:
                    print("Face alignment failed")

        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        status = "LIVE" if liveness_verified else "NOT VERIFIED"
        color = (0, 255, 0) if liveness_verified else (0, 0, 255)

        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Liveness + Embedding", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()