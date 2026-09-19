import cv2
import numpy as np

from face_engine import face_engine
from database import create_tables, add_user


# ==============================
# CONFIG
# ==============================

SAMPLES_REQUIRED = 10

# Chỉ lấy embedding mỗi N frame
CAPTURE_INTERVAL = 3


def register_user():

    create_tables()

    print("=" * 40)
    print("FACE REGISTRATION")
    print("=" * 40)

    student_id = input(
        "Student ID: "
    ).strip()

    name = input(
        "Name: "
    ).strip()

    if not student_id or not name:
        print("Student ID and name cannot be empty.")
        return

    # ==============================
    # OPEN CAMERA
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

    print()
    print("Camera started.")
    print("Look directly at the camera.")
    print("Move your head slightly.")
    print("Press Q to cancel.")
    print()

    embeddings = []

    frame_count = 0

    # ==============================
    # CAPTURE
    # ==============================

    while len(embeddings) < SAMPLES_REQUIRED:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera frame.")
            break

        frame_count += 1

        # Chỉ chạy AI mỗi vài frame
        if frame_count % CAPTURE_INTERVAL == 0:

            faces = face_engine.detect_faces(
                frame
            )

            # Chỉ đăng ký khi đúng 1 khuôn mặt
            if len(faces) == 1:

                face = faces[0]

                # ==========================
                # GET EMBEDDING
                # ==========================

                embedding = (
                    face.embedding
                    .astype(np.float32)
                )

                # Normalize
                norm = np.linalg.norm(
                    embedding
                )

                if norm > 0:
                    embedding = (
                        embedding / norm
                    )

                    embeddings.append(
                        embedding
                    )

                # ==========================
                # DRAW FACE
                # ==========================

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            elif len(faces) > 1:

                cv2.putText(
                    frame,
                    "Only one face allowed!",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # ==============================
        # DISPLAY STATUS
        # ==============================

        cv2.putText(
            frame,
            f"Register: {name}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"Samples: "
                f"{len(embeddings)}/"
                f"{SAMPLES_REQUIRED}"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to cancel",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Face Registration",
            frame
        )

        # ==============================
        # EXIT
        # ==============================

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            print("Registration cancelled.")

            cap.release()
            cv2.destroyAllWindows()

            return

    # ==============================
    # CLOSE CAMERA
    # ==============================

    cap.release()
    cv2.destroyAllWindows()

    # Không đủ samples
    if len(embeddings) < SAMPLES_REQUIRED:
        print("Not enough face samples.")
        return

    # ==============================
    # MEAN EMBEDDING
    # ==============================

    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    # Normalize lần cuối
    norm = np.linalg.norm(
        mean_embedding
    )

    if norm > 0:
        mean_embedding = (
            mean_embedding / norm
        )

    # ==============================
    # SAVE DATABASE
    # ==============================

    success = add_user(
        student_id,
        name,
        mean_embedding
    )

    if success:
        print()
        print("=" * 40)
        print("Registration successful!")
        print(
            f"Student ID: {student_id}"
        )
        print(
            f"Name: {name}"
        )
        print(
            f"Samples: {len(embeddings)}"
        )
        print("=" * 40)

    else:
        print()
        print(
            "Student ID already exists!"
        )


if __name__ == "__main__":
    register_user()