import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator
import joblib
import faiss

def preprocess(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0

    return img.flatten()

#X = np.array([preprocess(img) for img in images])

class BaselinePCA(BaseEstimator):
    def __init__(self, n_components=0.95, image_size=(64, 64)):
        self.n_components = n_components
        self.image_size = image_size

        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components=self.n_components,
            whiten=True,
            svd_solver="full",
            random_state=42
        )

        self.index = None
        self.paths_ = None
        self.embeddings_ = None

    def _preprocess(self, img):
        img = cv2.resize(img, self.image_size)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        img = img.astype(np.float32) / 255.0
        return img.flatten()

    def fit(self, X, file_ids=None):
        self.file_ids_ = file_ids

        X_proc = np.array([self._preprocess(img) for img in X])

        X_scaled = self.scaler.fit_transform(X_proc)
        X_pca = self.pca.fit_transform(X_scaled).astype(np.float32)

        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        X_norm = X_pca / (norms + 1e-8)

        self.embeddings_ = X_norm.astype(np.float32)

        dim = self.embeddings_.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings_)

        return self

    def recommend(self, query_img, top_k=5):
        q = self._preprocess(query_img).reshape(1, -1)

        q_scaled = self.scaler.transform(q)
        q_pca = self.pca.transform(q_scaled).astype(np.float32)

        q_norm = q_pca / (np.linalg.norm(q_pca, axis=1, keepdims=True) + 1e-8)

        scores, idxs = self.index.search(q_norm, top_k)

        idxs = idxs[0]
        scores = scores[0]

        if self.file_ids_ is not None:
            return [self.file_ids_[i] for i in idxs], scores

        return idxs, scores