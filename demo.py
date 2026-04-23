import gc
import os, sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image

sys.path.append(os.path.abspath("AdaFace"))


import models.adaface.net as net
from face_alignment import align

import cv2

import dlib
from scipy.spatial import distance
from imutils import face_utils

import time

''' 
Run the code by python demo.py
It needs adaface_ir50_ms1mv2.ckpt shape_predictor_68_face_landmarks.dat files to run
Press space to add a new embedding
Press Enter to verify
Press Esc to quit the program
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_input_tensor(img_tensor):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    img_tensor = F.interpolate(
        img_tensor,
        size=(112, 112),
        mode='bilinear',
        align_corners=False
    )

    img_tensor = img_tensor[:, [2, 1, 0], :, :]

    img_tensor = (img_tensor - 0.5) / 0.5

    img_tensor = img_tensor.to(device)

    return img_tensor

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def reset_gpu():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

def scalar(x):
    if isinstance(x, (tuple, list)):
        return float(x[0])
    return float(x)

def tensor_to_pil(img_tensor):
    img_tensor = img_tensor.detach().cpu()
    img_tensor = (img_tensor + 1) / 2
    img_tensor = img_tensor.clamp(0, 1)
    return TF.to_pil_image(img_tensor)

def image_path_to_tensor(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = TF.pil_to_tensor(img).float() / 255.0
    tensor = tensor * 2 - 1
    return tensor.unsqueeze(0).to(device)

def image_to_tensor(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    tensor = TF.pil_to_tensor(img).float() / 255.0
    tensor = tensor * 2 - 1
    return tensor.unsqueeze(0).to(device)

def load_fr_model():
    model = net.build_model("ir_50")
    ckpt = torch.load(
        "./Models/adaface_ir50_ms1mv2.ckpt")["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in ckpt.items() if k.startswith("model.")})
    model.to(device).eval()
    freeze(model)
    return model

def perspective_crop_tensor(img_tensor, coords, out_size=112):
    img_tensor = img_tensor.squeeze()

    src_pts = torch.tensor(coords, dtype=torch.float32, device=device)

    dst_pts = torch.tensor([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=torch.float32, device=device)

    def get_perspective_transform(src, dst):
        A = []
        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A.append(torch.tensor([x, y, 1, 0, 0, 0, -u*x, -u*y], device=device))
            A.append(torch.tensor([0, 0, 0, x, y, 1, -v*x, -v*y], device=device))
        A = torch.stack(A)

        b = dst.reshape(-1)

        h = torch.linalg.lstsq(A, b).solution
        H = torch.cat([h, torch.tensor([1.0], device=device)]).reshape(3, 3)
        return H

    H = get_perspective_transform(src_pts, dst_pts)

    C, H_img, W_img = img_tensor.shape

    ys, xs = torch.meshgrid(
        torch.linspace(0, out_size - 1, out_size, device=device),
        torch.linspace(0, out_size - 1, out_size, device=device),
        indexing='ij'
    )

    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=-1)

    H_inv = torch.inverse(H)

    src_grid = grid @ H_inv.T
    src_grid = src_grid[..., :2] / src_grid[..., 2:].clamp(min=1e-8)

    x = src_grid[..., 0]
    y = src_grid[..., 1]

    x = 2 * (x / (W_img - 1)) - 1
    y = 2 * (y / (H_img - 1)) - 1

    sampling_grid = torch.stack([x, y], dim=-1)
    img_tensor = img_tensor.unsqueeze(0)
    sampling_grid = sampling_grid.unsqueeze(0)

    cropped = F.grid_sample(
        img_tensor,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return cropped.squeeze(0)

def compute_fr_features_tensor_input(fr_model, face):
    bgr_tensor_input = to_input_tensor(face)
    features, _ = fr_model(bgr_tensor_input)
    del bgr_tensor_input, face
    return features 

def is_fr_success(fr_model, ref_img_path, adv_img_path, threshold = 0.3):
    ref_img = Image.open(ref_img_path)
    adv_img = Image.open(adv_img_path)

    if ref_img.mode != 'RGB':
        ref_img = ref_img.convert('RGB')
    if adv_img.mode != 'RGB':
        adv_img = adv_img.convert('RGB')

    _, _, ref_coords = align.get_aligned_face(None, ref_img)
    _, _, adv_coords = align.get_aligned_face(None, adv_img)
    
    ref_img = image_path_to_tensor(ref_img_path)
    adv_img = image_path_to_tensor(adv_img_path)

    ref_features = compute_fr_features_tensor_input(fr_model, ref_img, ref_coords)
    adv_features = compute_fr_features_tensor_input(fr_model, adv_img, adv_coords)

    similarity = torch.abs(F.cosine_similarity(adv_features, ref_features, dim=1))

    return similarity < threshold

def compute_similarity_matrix(features):
    feature_matrix = torch.cat(features)
    similarity = feature_matrix @ feature_matrix.T
    return similarity.cpu().numpy()


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./Models/shape_predictor_68_face_landmarks.dat")

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3
blink_counter = 0
liveness_verified = False
image_captured = False
verified = False
car_unlocked = False
unlock_time = None
UNLOCK_DURATION = 5  # seconds

fr_model = load_fr_model()
faces_list = []

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

status = "NOT VERIFIED"

face_features = None

while True:
    if status == "VERIFIED":
        elapsed = time.time() - unlock_time
        if elapsed < UNLOCK_DURATION:
            status = "VERIFIED"
        else:
            print("Locking car again")
            car_unlocked = False
            liveness_verified = False
            image_captured = False
            status = "NOT VERIFIED"

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
            if blink_counter >= CONSEC_FRAMES and not liveness_verified:
                print("Blink detected")
                liveness_verified = True
                status = "LIVE"
            blink_counter = 0

        if liveness_verified and not image_captured:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            aligned_face, _, _ = align.get_aligned_face(None, frame_pil)
            face_tensor = image_to_tensor(aligned_face)
            face_features = compute_fr_features_tensor_input(fr_model, face_tensor)

            if face_features is not None:
                print("Embedding generated successfully!")
                image_captured = True
            else:
                print("Face alignment failed")

        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        for (ex, ey) in leftEye:
            cv2.circle(frame, (ex, ey), 1, (0, 255, 0), -1)
        for (ex, ey) in rightEye:
            cv2.circle(frame, (ex, ey), 1, (0, 255, 0), -1)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    key = cv2.waitKey(30) & 0xFF

    if liveness_verified and face_features is not None:
        if key == 32: #Space
            faces_list.append(face_features)
            print(f"Added embedding. Total: {len(faces_list)}")
            image_captured = False

        if key == 13 and len(faces_list) > 0: # Enter

            query = np.squeeze(face_features / np.linalg.norm(face_features))
            gallery = np.array([ f / np.linalg.norm(f) for f in faces_list])
            sim_scores = np.dot(gallery, query)

            best_index = np.argmax(sim_scores)
            best_match = np.max(sim_scores)

            print("Similarity scores:", sim_scores)
            print("Best match:", best_match, "Index:", best_index)

            if best_match > 0.8:
                status = "VERIFIED"
                unlock_time = time.time()
                car_unlocked = True
                print("Car unlocked")
            else:
                print("Verification failed")
                liveness_verified = False
                image_captured = False

    color = (0, 255, 0) if liveness_verified else (0, 0, 255)

    cv2.putText(frame, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Liveness + Embedding", frame)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
