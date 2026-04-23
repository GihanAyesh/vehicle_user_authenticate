import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces, aligned_coords = mtcnn_model.align_multi(img, limit=None)  # no limit to get all faces
        if len(faces) == 0:
            raise ValueError("No faces found")
        areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        largest_idx = areas.index(max(areas))
        face = faces[largest_idx]
        bbox = bboxes[largest_idx]
        coords = aligned_coords[largest_idx]
        # print ("bbox", bbox)
        # print(img.size, face.size)
        # bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        # face = faces[0]
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None
        bbox = None
        coords = None

    return face, bbox, coords

