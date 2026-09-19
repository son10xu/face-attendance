import cv2
import numpy as np

from face_engine import face_engine
from database import (
    create_tables,
    get_all_users,
    mark_attendance
)


# ==============================
# CONFIG
# ==============================

# Đây là giá trị khởi đầu để test.
# Sau này cần hiệu chỉnh trên dữ liệu thực tế.
SIMILARITY_THRESHOLD = 0.45

# Chỉ chạy recognition mỗi 3 frame
PROCESS_EVERY_N_FRAMES = 3


def find_best_match(query_embedding, users):
    """
    So sánh khuôn mặt hiện tại với tất cả user.
    """

    best_user = None
    best_score = -1.0

    for user in users:

        stored_embedding = user["embedding"]

        score = face_engine.cosine_similarity(
            query_embedding,
            stored_embedding
        )

        if score > best_score:
            best_score = score
            best_user = user

    return best_user, best_score


def run_attendance():

    create_tables()

    # ==============================
    # LOAD USERS
    # ==============================

    users = get_all_users()

    if len(users) == 0:
        print()
        print("No registered users.")
        print("Please register a user first.")
        return

    print()
    print("=" * 50)
    print("FACE RECOGNITION ATTENDANCE")
    print("=" * 50)

    print(
        f"Loaded {len(users)} registered user(s)."
    )

    print("Press Q to exit.")
    print()

    # ==============================
    # CAMERA
    # ==============================

    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    frame_count = 0

    # Lưu kết quả recognition gần nhất
    display_results = []

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera frame.")
            break

        frame_count += 1

        # ==========================
        # FACE RECOGNITION
        # ==========================

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:

            faces = face_engine.detect_faces(
                frame
            )

            display_results = []

            for face in faces:

                # --------------------------
                # EMBEDDING
                # --------------------------

                embedding = (
                    face.embedding
                    .astype(np.float32)
                )

                norm = np.linalg.norm(
                    embedding
                )

                if norm == 0:
                    continue

                embedding = (
                    embedding / norm
                )

                # --------------------------
                # MATCH
                # --------------------------

                best_user, best_score = (
                    find_best_match(
                        embedding,
                        users
                    )
                )

                # --------------------------
                # BOUNDING BOX
                # --------------------------

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                # --------------------------
                # KNOWN / UNKNOWN
                # --------------------------

                if (
                    best_user is not None
                    and
                    best_score
                    >= SIMILARITY_THRESHOLD
                ):

                    name = best_user["name"]

                    student_id = (
                        best_user["student_id"]
                    )

                    # Tự động điểm danh
                    added = mark_attendance(
                        best_user["id"]
                    )

                    if added:
                        print(
                            f"[ATTENDANCE] "
                            f"{student_id} - "
                            f"{name} - "
                            f"{best_score:.3f}"
                        )

                        status = "Attendance OK"

                    else:
                        status = "Already attended"

                    recognized = True

                else:

                    name = "Unknown"
                    student_id = ""

                    status = "Not recognized"

                    recognized = False

                display_results.append({
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "name": name,
                    "student_id": student_id,
                    "score": best_score,
                    "status": status,
                    "recognized": recognized
                })

        # ==========================
        # DRAW RESULT
        # ==========================

        for result in display_results:

            x1, y1, x2, y2 = (
                result["bbox"]
            )

            if result["recognized"]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Name
            label = (
                f"{result['name']} "
                f"{result['score']:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 30, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Student ID
            if result["student_id"]:

                cv2.putText(
                    frame,
                    result["student_id"],
                    (x1, max(y1 - 8, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # ==========================
        # INFORMATION
        # ==========================

        cv2.putText(
            frame,
            f"Users: {len(users)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Threshold: {SIMILARITY_THRESHOLD}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Press Q to exit",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # ==========================
        # SHOW
        # ==========================

        cv2.imshow(
            "Face Attendance",
            frame
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance()