import sqlite3
import os
import numpy as np

DB_PATH = "data/attendance.db"


def connect():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def add_user(student_id, name, embedding):
    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (
                student_id,
                name,
                embedding
            )
            VALUES (?, ?, ?)
            """,
            (
                student_id,
                name,
                embedding.astype(np.float32).tobytes()
            )
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


if __name__ == "__main__":
    create_tables()
    print("Database created successfully!")


def get_all_users():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, student_id, name, embedding
        FROM users
    """)

    rows = cursor.fetchall()
    conn.close()

    users = []

    for user_id, student_id, name, embedding_blob in rows:
        embedding = np.frombuffer(
            embedding_blob,
            dtype=np.float32
        )

        users.append({
            "id": user_id,
            "student_id": student_id,
            "name": name,
            "embedding": embedding
        })

    return users


def has_attended_today(user_id):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE user_id = ?
        AND DATE(timestamp, 'localtime')
            = DATE('now', 'localtime')
    """, (user_id,))

    count = cursor.fetchone()[0]

    conn.close()

    return count > 0


def mark_attendance(user_id):
    if has_attended_today(user_id):
        return False

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO attendance (user_id)
        VALUES (?)
    """, (user_id,))

    conn.commit()
    conn.close()

    return True