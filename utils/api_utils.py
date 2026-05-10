import joblib
import numpy as np
from flask import Flask, request, jsonify
from logging_utils import async_log
from drive_utils import download_image_as_array
from model_utils import BaselinePCA
import os
import cv2
from logging_utils import init_wandb

init_wandb(os.getenv("WANDB_API_KEY"))

model_path = os.path.expanduser("../model/baseline_pca.pkl")
model = joblib.load(model_path)

print(model)
print(model.scaler.mean_.shape)

app = Flask(__name__)


def preprocess(img):
    # force RGB (defensive)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0

    # sanity check
    assert img.shape == (64, 64, 3)

    return img.flatten()



@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json

        if not data or "file_id" not in data:
            return jsonify({"error": "file_id is required"}), 400

        file_id = data["file_id"]
        top_k = data.get("top_k", 5)

        img = download_image_as_array(file_id)
        query = preprocess(img)

        print("INPUT SHAPE:", query.shape)
        print("EXPECTED:", model.scaler.n_features_in_)
        print("Model Path", model_path)
        print("QUERY SHAPE:", query.shape)
        print("EMBEDDINGS SHAPE:", model.embeddings_.shape)
        recs, scores = model.recommend(query, top_k=top_k)

        response = {
            "file_id": file_id,
            "recommendations": list(recs),
            "scores": [float(s) for s in scores]
        }

        async_log(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
