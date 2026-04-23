# Face Recognition System

This repository contains a face recognition system that integrates face detection, liveness detection, and face recognition for unlocking vehicles.

---

# Usage

Run the main script:

```bash
python demo.py
```

Operating Keys
- Press Space to add a new embedding
- Press Enter to verify
- Press Esc to quit the program

Depending on the implementation, you may need to:
- Download pretrained models into the `Models/` directory -  [Google Drive Link](https://drive.google.com/drive/folders/1-5GONFaO9dotWc6Ar83OsQcf3OD138W5?usp=sharing)

---

# Requirements

Install dependencies including:

```text
opencv-python
torch
torchvision
numpy
scikit-learn
Pillow
facenet-pytorch
deepface
tqdm
insight-face
```

---

## Repository Structure

```text
.
├── Adaface/
├── Faces/
├── Models/
├── adaface.py
├── demo.py
├── liveness_check.py
├── models_test.py
└── README.md
```

### Folder Descriptions

#### `Adaface/`
Contains the support code and utilities required for the AdaFace face recognition model.

#### `Models/`
Stores pretrained models used in the system, including:
- Face detection models
- Liveness detection models

#### `Faces/`
Contains example face images that can be used for face recognition and matching.


---

## Python Scripts

| Script | Description |
|---|---|
| `adaface.py` | Code to run AdaFace and generate similarity matrix for examples in folder Faces |
| `demo.py` | Main code which runs the workflow for unlocking cars using face recognition |
| `liveness_check.py` | Code for checking liveness |
| `models_test.py` | Code for running different face recognition model for comparision |


---

# Features

- Face detection
- Face recognition using AdaFace
- Liveness detection
- Support for stored face galleries
- Real-time or image-based recognition pipeline
- Multiple model comparision - FaceNet, OpenFace, ArcFace, AdaFace 



