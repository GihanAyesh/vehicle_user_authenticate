import gc
import os, sys
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image

sys.path.append(os.path.abspath("AdaFace"))

import models.adaface.net as net
from face_alignment import align

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
    ckpt = torch.load("./Models/adaface_ir50_ms1mv2.ckpt")["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in ckpt.items() if k.startswith("model.")})
    model.to(device).eval()
    freeze(model)
    return model

def perspective_crop_tensor(img_tensor, coords, out_size=112):
    img_tensor = img_tensor.squeeze()

    src_pts = torch.tensor(coords, dtype=torch.float32, device=device)  # (4,2)

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

def add_image(ax, img, xy, zoom=0.55):
    img = img.convert("RGB")
    img = img.resize((155, 155))

    imagebox = OffsetImage(np.array(img), zoom=zoom)

    ab = AnnotationBbox(
        imagebox,
        xy,
        frameon=False,
        pad=0.0,
        box_alignment=(0.5, 0.5)
    )

    ax.add_artist(ab)


def plot_similarity_heatmap(similarity_matrix, images):
    n = len(images)

    fig, ax = plt.subplots(figsize=(12, 12))

    sim_display = similarity_matrix.copy()
    im = ax.imshow(sim_display, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    offset = 1.0

    for i, img in enumerate(images):

        add_image(ax, img, (i, -offset), zoom=0.42)
        add_image(ax, img, (-offset, i), zoom=0.42)

    ax.set_xlim(-1.5, n - 0.5)
    ax.set_ylim(n - 0.5, -1.5)
    ax.set_title("Face Similarity Matrix", fontsize=18, pad=30)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)

    plt.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.08)
    plt.savefig("sim_matrix_adaface.png")


folder_path = "./Faces/"

fr_model = load_fr_model()
features = []
cropped_faces = []

for identity_path in sorted(os.listdir(folder_path)):
    face_path = os.path.join(folder_path, identity_path)
    
    face, _, _ = align.get_aligned_face(face_path, None)
    face_tensor = image_to_tensor(face)
    face_features = compute_fr_features_tensor_input(fr_model, face_tensor)

    features.append(face_features)
    cropped_faces.append(face)

sim_matrix = compute_similarity_matrix(features)

print (sim_matrix)

plot_similarity_heatmap(
    sim_matrix,
    cropped_faces
)