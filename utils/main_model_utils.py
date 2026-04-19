from sklearn.metrics.pairwise import cosine_similarity
from byol_pytorch import BYOL
from sklearn.base impor BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
from sklearn.preprocessing import *
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
import umap





class BYOLWrapper:
    def __init__(self, byol_model):
        self.model = byol_model

    @torch.no_grad()
    def encode(self, x):
        x = self.model.online_encoder(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        return x


def extract_byol_embeddings(byol_model, x):
    with torch.no_grad():
        if hasattr(byol_model.online_encoder, "net"):
            emb = byol_model.online_encoder.net(x)
        else:
            emb = byol_model.online_encoder(x)

        if isinstance(emb, (tuple, list)):
            emb = emb[0]

        return emb
#For a Frozen Encoder only
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class BaselinePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.95, alpha=0.7):
        self.n_components = n_components
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Add device
        backbone = SimpleCNN(out_dim=512)

        self.byol = BYOL(
          net=backbone,
          image_size=224,
          hidden_layer='fc'
        ).to(self.device) # Move BYOL model to device

        self.scaler = StandardScaler()

        self.pca = PCA(
            n_components=self.n_components,
            whiten=True,
            svd_solver='full',
            random_state=42
        )

        self.dbscan = DBSCAN(
            eps=0.7,
            metric="cosine",
            min_samples=10,
            n_jobs=-1
        )

    def fit(self, X, y=None):
        X_tensor = torch.from_numpy(X).float().to(self.device) # Convert to tensor and move to device
        with torch.no_grad():
            X_encoded = self.byol.online_encoder(X_tensor) # Use X_tensor

        if isinstance(X_encoded, tuple):
            X_encoded = X_encoded[1]

        if X_encoded.ndim == 4:
            import torch.nn.functional as F
            X_encoded = F.adaptive_avg_pool2d(X_encoded, (1, 1))

        X_processed = X_encoded.view(X_encoded.size(0), -1).cpu().detach().numpy() # Move back to CPU for sklearn

        X_scaled = self.scaler.fit_transform(X_processed)
        X_pca = self.pca.fit_transform(X_scaled)
        X_norm = normalize(X_pca)

        self.labels_ = self.dbscan.fit_predict(X_norm)
        self.embeddings_ = X_norm

        return self

    def recommend(self, query_embedding, top_k=5):
        # query_embedding is expected to be a single image (batch size 1) already preprocessed as numpy
        q_embedding_tensor = torch.from_numpy(query_embedding).float().to(self.device) # Convert to tensor and move to device
        with torch.no_grad():
            q_embedding_encoded = self.byol.online_encoder(q_embedding_tensor) # Use q_embedding_tensor

        if isinstance(q_embedding_encoded, tuple):
            q_embedding_encoded = q_embedding_encoded[1]

        if q_embedding_encoded.ndim == 4:
            import torch.nn.functional as F
            q_embedding_encoded = F.adaptive_avg_pool2d(q_embedding_encoded, (1, 1))

        query_embedding_processed = q_embedding_encoded.view(q_embedding_encoded.size(0), -1).cpu().detach().numpy() # Move back to CPU for sklearn

        q_scaled = self.scaler.transform(query_embedding_processed.reshape(1, -1))
        q_pca = self.pca.transform(q_scaled)
        q_norm = normalize(q_pca)

        sim = cosine_similarity(q_norm, self.embeddings_)[0]

        q_neighbors = cosine_similarity(q_norm, self.embeddings_)[0]
        nearest_idx = np.argmax(q_neighbors)
        q_cluster = self.labels_[nearest_idx]

        cluster_scores = np.zeros_like(sim)

        for i, label in enumerate(self.labels_):
            if label == -1:
                cluster_scores[i] = -0.2
            elif label == q_cluster:
                cluster_scores[i] = 1.0
            else:
                cluster_scores[i] = 0.0

        final_score = self.alpha * sim + (1 - self.alpha) * cluster_scores

        top_idx = np.argsort(final_score)[::-1][:top_k]

        return top_idx, final_score[top_idx]
