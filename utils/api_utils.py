import joblib
import numpy as np
from flask import Flask, request, jsonify
from logging_utils import async_log, get_db
from drive_utils import download_image_as_array
from model_utils import BaselinePCA
import os
import cv2
from logging_utils import init_wandb
from werkzeug.security import generate_password_hash
from flask import send_file
from flask_cors import CORS
import sqlite3
from werkzeug.security import check_password_hash

init_wandb(os.getenv("WANDB_API_KEY"))

model_path = os.path.expanduser("./model/baseline_pca.pkl")
model = joblib.load(model_path)

print(model)
print(model.scaler.mean_.shape)

app = Flask(__name__)
CORS(app)


def path_to_drive_url(file_id):
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"

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


IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Image_Data")
)

@app.route("/api/image/<filename>")
def get_image(filename):
    path = os.path.join(IMAGE_DIR, filename)
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/recommend", methods=["POST"])
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


        recs = [str(r) for r in recs]  # filenames safe
        scores = [float(s) for s in scores]  # convert np.float32 → float

        response = {
            "file_id": file_id,
            "recommendations": [
            {
                "file_id": rec,
                "image_url": f"http://127.0.0.1:5000/api/image/{rec}",
                "score": score
            }
                for rec, score in zip(recs, scores)
            ]
        }


        log_payload = {
            "file_id": file_id,
            "recommendations": recs,
            "scores": scores
        }

        async_log(log_payload)



        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def register_user(username, email, password):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO users (username, email, password)
        VALUES (?, ?, ?)
    """, (username, email, password))

    conn.commit()
    conn.close()

@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.json

        register_user(
            data["username"],
            data["email"],
            generate_password_hash(data["password"])
        )

        return jsonify({
            "message": "User registered"
        }), 201

    except sqlite3.IntegrityError:
        return jsonify({
            "error": "Email already exists"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

def authenticate_user(email, password):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM users
        WHERE email = ?
    """, (email,))

    user = cursor.fetchone()

    conn.close()

    if not user:
        return None

    if not check_password_hash(user["password"], password):
        return None

    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"]
    }

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json

        user = authenticate_user(
            data["email"],
            data["password"]
        )

        if not user:
            return jsonify({
                "error": "Invalid credentials"
            }), 401

        return jsonify({
            "message": "Login successful",
            "user": user
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)

