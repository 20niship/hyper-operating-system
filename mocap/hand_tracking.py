import matplotlib

matplotlib.use("TkAgg")  # Macの場合は "MacOSX" に変更してください

from matplotlib import pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap.stereo_tri import StereoCalibrator

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore

# MediaPipe Hands のインスタンスを作成
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


class Hand3DTracker:
    def __init__(self, cam_l_idx=0, cam_r_idx=1):
        self.cap1 = cv2.VideoCapture(cam_l_idx)
        self.cap2 = cv2.VideoCapture(cam_r_idx)

        assert self.cap1.isOpened(), "Cannot open left camera"
        assert self.cap2.isOpened(), "Cannot open right camera"
        self.cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
        self.cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore

        self.frame1 = None
        self.frame2 = None

        self.tri = StereoCalibrator()
        if not self.tri.load("stereo_calib.json"):
            raise ValueError("Failed to load stereo calibration data")

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        NUM_HAND_LANDMARKS = 21 * 2  # left, right
        self.plt_x, self.plt_y, self.plt_z = (
            [0] * NUM_HAND_LANDMARKS,
            [0] * NUM_HAND_LANDMARKS,
            [0] * NUM_HAND_LANDMARKS,
        )
        (self.plt_point,) = self.ax.plot([self.plt_x], [self.plt_y], [self.plt_z], "ro")

    def track_hands(self, frame):
        # Process the frame and return 3D hand landmarks
        # This is a placeholder implementation
        landmarks = []  # Replace with actual landmark detection logic
        return landmarks

    def _capture(self):
        ret1, self.frame1 = self.cap1.read()
        ret2, self.frame2 = self.cap2.read()
        if not ret1 or not ret2:
            print("warning: failed to grab frame")

    def _render(self, results, results2):
        assert self.frame1 is not None and self.frame2 is not None

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    self.frame1,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )
        if results2.multi_hand_landmarks:
            for landmarks in results2.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    self.frame2,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

        combined = np.hstack((self.frame1, self.frame2))
        if combined.shape[1] > 2000:
            combined = cv2.resize(combined, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Stereo Cameras", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            return False

    def update(self) -> bool:
        # RGB に変換して検出
        if self.frame1 is None or self.frame2 is None:
            return False
        rgb_f1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        rgb_f2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_f1)
        results2 = hands.process(rgb_f2)
        self._render(results, results2)

        if results.multi_hand_landmarks and results2.multi_hand_landmarks:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results2.multi_hand_landmarks[0]
            pts1 = np.array(
                [
                    [lm.x * self.frame1.shape[1], lm.y * self.frame1.shape[0]]
                    for lm in hand1.landmark
                ],
                dtype=np.float32,
            )
            pts2 = np.array(
                [
                    [lm.x * self.frame2.shape[1], lm.y * self.frame2.shape[0]]
                    for lm in hand2.landmark
                ],
                dtype=np.float32,
            )
            points_3d = [self.tri.triangulate(p1, p2) for p1, p2 in zip(pts1, pts2)]
            for i, point in enumerate(points_3d):
                print(f"Landmark {i}: {point}")  # 3D座標を表示
                self.plt_x[i] = point[0]
                self.plt_y[i] = point[1]
                self.plt_z[i] = point[2]

        self.plt_point.set_data(self.plt_x, self.plt_y)
        self.plt_point.set_3d_properties(self.plt_z)  # type: ignore
        plt.draw()
        plt.pause(0.001)

        return True

    def close(self):
        self.cap1.release()
        self.cap2.release()
        hands.close()
        cv2.destroyAllWindows()


def _get_hand_quat(landmarks, image_width, image_height):
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


if __name__ == "__main__":
    tracker = Hand3DTracker(cam_l_idx=0, cam_r_idx=2)
    while True:
        tracker._capture()
        if not tracker.update():
            break
    tracker.close()
