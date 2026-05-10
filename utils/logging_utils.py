import sqlite3
import json
from datetime import datetime
import threading
import numpy as np
import wandb
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "artmatch.db")

_wandb_initialized = False

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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            file_id TEXT,
            recommendations TEXT,
            scores TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            file_id TEXT PRIMARY KEY,
            embedding TEXT
        )
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
        wandb.log({
            "file_id": payload["file_id"],
            "avg_score": float(np.mean(payload["scores"])) if payload["scores"] else 0.0,
            "max_score": float(np.max(payload["scores"])) if payload["scores"] else 0.0,
            "min_score": float(np.min(payload["scores"])) if payload["scores"] else 0.0
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
