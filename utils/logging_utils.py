import sqlite3
import json
from datetime import datetime
import threading
import numpy as np
import wandb
import os
from werkzeug.security import generate_password_hash, check_password_hash


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "artmatch.db")

_wandb_initialized = False

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def register_user(username, email, password):
    conn = get_db()
    cursor = conn.cursor()

    hashed = generate_password_hash(password)

    cursor.execute("""
        INSERT INTO users (username, email, password)
        VALUES (?, ?, ?)
    """, (username, email, hashed))

    conn.commit()
    conn.close()


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


def init_wandb(api_key):
    global _wandb_initialized

    print("[WANDB] initialized")

    wandb.login(key=api_key)
    wandb.init(
        project="artmatch-recommendation",
        name="inference-logging",
        reinit=True
    )

    _wandb_initialized = True


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # USERS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'Viewer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ARTWORKS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artworks (
            artwork_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            image_url TEXT,
            artist_name TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # USER INTERACTIONS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            artwork_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(user_id)
                ON DELETE CASCADE,

            FOREIGN KEY (artwork_id)
                REFERENCES artworks(artwork_id)
                ON DELETE CASCADE
        )
    """)

    # LOGS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            file_id TEXT,
            recommendations TEXT,
            scores TEXT
        )
    """)

    # EMBEDDINGS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            file_id TEXT PRIMARY KEY,
            embedding TEXT
        )
    """)

    # FAVORITES
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            user_id INTEGER,
            artwork_id INTEGER,

            PRIMARY KEY (user_id, artwork_id),

            FOREIGN KEY (user_id)
                REFERENCES users(user_id)
                ON DELETE CASCADE,

            FOREIGN KEY (artwork_id)
                REFERENCES artworks(artwork_id)
                ON DELETE CASCADE
        )
    """)

    # INDEXES
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user
        ON user_interactions(user_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_artwork
        ON user_interactions(artwork_id)
    """)

    conn.commit()
    conn.close()



def log_to_db_only(payload):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (timestamp, file_id, recommendations, scores)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        payload["file_id"],
        json.dumps(payload["recommendations"]),
        json.dumps(payload["scores"])
    ))

    if "embedding" in payload:
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (file_id, embedding)
            VALUES (?, ?)
        """, (
            payload["file_id"],
            json.dumps(payload["embedding"])
        ))

    conn.commit()
    conn.close()

def log_wandb(payload):
    global _wandb_initialized

    if not _wandb_initialized:
        print("[WANDB WARNING] init not called, skipping log")
        return

    try:
        scores = payload.get("scores", [])

        wandb.log({
            "file_id": payload.get("file_id", "unknown"),
            "avg_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "max_score": float(np.max(scores)) if len(scores) > 0 else 0.0,
            "min_score": float(np.min(scores)) if len(scores) > 0 else 0.0
        })

    except Exception as e:
        print(f"[WANDB ERROR] {e}")

def async_log(payload):
    def worker():
        try:
            log_to_db_only(payload)
            log_wandb(payload)
        except Exception as e:
            print(f"[ASYNC LOG ERROR] {e}")

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
