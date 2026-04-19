import sqlite3
import json
from datetime import datetime
import threading
from threading import Thread


DB_PATH = "../sql/recommendations.db"

def log_to_db(payload):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
            INSERT INTO logs (timestamp, file_id, recommendations, scores)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            payload["file_id"],
            json.dumps(payload["recommendations"]),
            json.dumps(payload["scores"])
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"[LOGGING ERROR] {e}")


def async_log(payload):
    thread = Thread(target=log_to_db, args=(payload,))
    thread.daemon = True  # dies with main process
    thread.start()



