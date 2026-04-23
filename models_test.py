import os,sys
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from deepface import DeepFace
from insightface.app import FaceAnalysis

''' The code consists of 4 models to test

First model : InceptionResnetV1(pretrained='vggface2')
Second model : InceptionResnetV1(pretrained='casia-webface') - To use uncomment this model
Thrid model : FaceNet - To use uncomment line 42-51
Fourth model : ArcFace - To use uncomment line "embedding = DeepFace.represent(...)" line and comment the line above
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

def get_embedding(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    face_tensor = mtcnn(img_pil)

    if face_tensor is None:
        print(f"No face detected in {image_path}")
        return None

    emb = resnet(face_tensor.unsqueeze(0).to(device))
    return emb.detach().cpu().numpy().squeeze()

# app = FaceAnalysis()
# app.prepare(ctx_id=0)

# def get_embedding(image_path):
#     img = cv2.imread(image_path)
#     faces = app.get(img)

#     if len(faces) == 0:
#         return None

#     return faces[0].embedding

mtcnn = MTCNN(image_size=160, margin=0, device=device)

def compute_similarity_matrix(features):
    return np.dot(features, features.T)

folder_path = "./Faces"

features = []
image_names = []

for file_name in sorted(os.listdir(folder_path)):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, file_name)

        # embedding = get_embedding(img_path)
        embedding = DeepFace.represent(img_path=img_path, model_name="OpenFace")[0]["embedding"]

        if embedding is not None:
            features.append(embedding)
            image_names.append(file_name)
            print(f"{file_name} → embedding generated")

features = np.array(features)

features = features / np.linalg.norm(features, axis=1, keepdims=True)

sim_matrix = compute_similarity_matrix(features)

print("\nImage Order:")
for i, name in enumerate(image_names):
    print(f"{i}: {name}")

print("\nSimilarity Matrix:\n")
print(sim_matrix)