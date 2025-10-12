import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_hand_quaternion(landmarks, image_width, image_height):
    # 主要なランドマークを取得
    wrist = np.array(
        [landmarks[0].x * image_width, landmarks[0].y * image_height, landmarks[0].z]
    )
    index_mcp = np.array(
        [landmarks[5].x * image_width, landmarks[5].y * image_height, landmarks[5].z]
    )
    pinky_mcp = np.array(
        [landmarks[17].x * image_width, landmarks[17].y * image_height, landmarks[17].z]
    )
    middle_mcp = np.array(
        [landmarks[9].x * image_width, landmarks[9].y * image_height, landmarks[9].z]
    )

    # 手のローカル座標系を定義
    x_axis = pinky_mcp - index_mcp
    x_axis /= np.linalg.norm(x_axis)

    y_axis = middle_mcp - wrist
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # 再直交化 (Gram-Schmidt)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 回転行列
    R_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    # クオータニオンに変換
    quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)
    return quat


def main():
    cap = cv2.VideoCapture(0)

    # MediaPipe Hands のインスタンスを作成
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # RGB に変換して検出
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # 手のランドマークを描画
        eulers = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                quat = get_hand_quaternion(
                    hand_landmarks.landmark, frame.shape[1], frame.shape[0]
                )
                euler = R.from_quat(quat).as_euler("xyz", degrees=True)
                eulers.append(euler)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

        # print Euler angles for each detected hand
        for i, euler in enumerate(eulers):
            euler_t = f"{int(euler[0])} {int(euler[1])} {int(euler[2])}"
            cv2.putText(
                frame,
                f"Hand {i + 1} Euler: {euler_t}",
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
