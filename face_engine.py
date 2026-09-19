import cv2
import numpy as np
import threading
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self):
        print("Loading face recognition model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        # 320x320 nhanh hơn 640x640 khi chạy CPU
        self.app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )

        print("Face recognition model loaded!")

    def detect_faces(self, frame):
        """
        Phát hiện tất cả khuôn mặt trong frame.
        """
        return self.app.get(frame)

    def get_largest_face(self, frame):
        """
        Lấy khuôn mặt lớn nhất trong frame.
        """
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return None

        largest_face = max(
            faces,
            key=lambda face:
            (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1])
        )

        return largest_face

    def get_embedding(self, frame):
        """
        Lấy embedding của khuôn mặt lớn nhất.
        """

        face = self.get_largest_face(frame)

        if face is None:
            return None, None

        embedding = face.embedding.astype(np.float32)

        # Normalize embedding
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        return embedding, face

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Tính cosine similarity giữa 2 embedding.
        """

        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(
            np.dot(embedding1, embedding2)
            / (norm1 * norm2)
        )


# Tạo FaceEngine dùng chung
face_engine = FaceEngine()


# ==============================
# TEST CAMERA
# ==============================

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    # Giữ độ trễ thấp: bỏ qua frame cũ nếu nhận diện chưa kịp xử lý.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Giảm resolution webcam
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
        exit()

    print("Camera started.")
    print("Press Q to exit.")

    frame_lock = threading.Lock()
    result_lock = threading.Lock()
    latest_frame = None
    faces = []
    stop_detection = False

    def detect_in_background():
        nonlocal_frame = None
        while True:
            with frame_lock:
                if latest_frame is not None:
                    nonlocal_frame = latest_frame.copy()

            if nonlocal_frame is None:
                if stop_detection:
                    break
                continue

            detected_faces = face_engine.detect_faces(nonlocal_frame)
            with result_lock:
                faces[:] = detected_faces
            nonlocal_frame = None

            if stop_detection:
                break

    detection_thread = threading.Thread(
        target=detect_in_background,
        daemon=True
    )
    detection_thread.start()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera frame.")
            break

        with frame_lock:
            latest_frame = frame.copy()

        with result_lock:
            faces_to_draw = list(faces)

        # ==========================
        # DRAW RESULT
        # ==========================

        for face in faces_to_draw:

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Detection confidence
            score = float(
                face.det_score
            )

            cv2.putText(
                frame,
                f"Face: {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Hiển thị số khuôn mặt
        cv2.putText(
            frame,
            f"Faces: {len(faces_to_draw)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ==========================
        # SHOW CAMERA
        # ==========================

        cv2.imshow(
            "Face Detection",
            frame
        )

        # Q = exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==============================
    # CLEAN UP
    # ==============================

    cap.release()
    stop_detection = True
    detection_thread.join(timeout=1.0)
    cv2.destroyAllWindows()