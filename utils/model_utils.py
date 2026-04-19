import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from byol_pytorch import BYOL
import seaborn as sns
import mlflow
from dowhy import CausalModel
import mlflow.sklearn
import mlflow.pytorch
import clip as openai_clip
import copy
import random
import transformers
import requests
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.utils.data import Dataset

from pyngrok import ngrok
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import *
from sklearn.preprocessing import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
import umap

import joblib
from warnings import filterwarnings; filterwarnings("ignore")

#Helper Functions
def random_resized_crop(img, size=224, scale=(0.08, 1.0)):
    img = np.array(img)

    h, w = img.shape[:2]
    area = h * w

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(3/4, 4/3)

        new_w = int(round(np.sqrt(target_area * aspect_ratio)))
        new_h = int(round(np.sqrt(target_area / aspect_ratio)))

        if new_w <= w and new_h <= h:
            x = random.randint(0, w - new_w)
            y = random.randint(0, h - new_h)
            crop = img[y:y+new_h, x:x+new_w]
            return cv2.resize(crop, (size, size))

    # fallback
    return cv2.resize(img, (size, size))


def random_horizontal_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, 1)
    return img


def color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4):
    img = img.astype(np.float32) / 255.0

    # brightness
    if brightness > 0:
        factor = 1 + random.uniform(-brightness, brightness)
        img *= factor

    # contrast
    if contrast > 0:
        mean = img.mean(axis=(0, 1), keepdims=True)
        factor = 1 + random.uniform(-contrast, contrast)
        img = (img - mean) * factor + mean

    # saturation (convert to HSV)
    if saturation > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        factor = 1 + random.uniform(-saturation, saturation)
        hsv[..., 1] *= factor
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(img, 0, 1)


def random_grayscale(img, p=0.2):
    if random.random() < p:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack([gray, gray, gray], axis=-1)
    return img


def gaussian_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def normalize_img(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img


def to_tensor(img):
    return np.transpose(img, (2, 0, 1))


def transform(img):
    img = random_resized_crop(img, 224)
    img = random_horizontal_flip(img)
    img = color_jitter(img, 0.4, 0.4, 0.4)
    img = random_grayscale(img, 0.2)
    img = gaussian_blur(img, 3)

    img = img.astype(np.float32) / 255.0
    img = normalize_img(img,
                    mean=np.array([0.485, 0.456, 0.406]),
                    std=np.array([0.229, 0.224, 0.225]))
    img = to_tensor(img)

    return img
def preprocess_inference(img, size=224):
    """
    Deterministic preprocessing for inference (NO augmentation).
    """

    img = cv2.resize(img, (size, size))

    # Ensure float
    img = img.astype(np.float32) / 255.0

    # Normalize (same as training)
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def get_embedding(img, byol_model, device):
    """
    Extract deterministic BYOL embeddings for inference.
    """

    byol_model.eval()

    img = preprocess_inference(img)

    img_tensor = torch.from_numpy(img).float().to(device)

    with torch.no_grad():
        emb = byol_model.online_encoder(img_tensor)

    # Handle BYOL output formats
    if isinstance(emb, (tuple, list)):
        emb = emb[0]

    # If feature map → pool to vector
    if emb.ndim == 4:
        emb = F.adaptive_avg_pool2d(emb, (1, 1))

    emb = emb.view(emb.size(0), -1)

    return emb.cpu().numpy()


