import joblib
import numpy as np
from flask import Flask, request, jsonify
from logging_utils import async_log
from drive_utils import download_image_as_array
from model_utils import get_embedding  

app = Flask(__name__)

model = joblib.load("../model/baseline_pca_model.joblib")


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        file_id = data.get("file_id")
        top_k = data.get("top_k", 5)

        img = download_image_as_array(file_id)
        embedding = get_embedding(img, byol_model, device)

        recs, scores = model.recommend(embedding, top_k=top_k)

        response = {
            "file_id": file_id,
            "recommendations": recs.tolist(),
            "scores": scores.tolist()
        }

        async_log(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
