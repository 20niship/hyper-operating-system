import matplotlib
import sys
from pathlib import Path

path_ = Path(__file__).parent.parent
sys.path.append(str(path_))

matplotlib.use("TkAgg")  # Macの場合は "MacOSX" に変更してください

from matplotlib import pyplot as plt
from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap.stereo_tri import StereoCalibrator
from mocap.ar_coordinate_transformer import calib_marker_coordinates, get_coord_uv_pos

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore

MARKER_ID = 0  # 原点として使用するマーカーID

# MediaPipe Hands のインスタンスを作成
hands_l = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
hands_r = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def _landmarks_to_pos_rot(
    ld_x: list[float], ld_y: list[float], ld_z: list[float], offset_idx=0
) -> tuple[np.ndarray, np.ndarray]:
    wrist = np.array([ld_x[offset_idx], ld_y[offset_idx], ld_z[offset_idx]])
    index_base = np.array(
        [ld_x[5 + offset_idx], ld_y[5 + offset_idx], ld_z[5 + offset_idx]]
    )
    middle_base = np.array(
        [ld_x[9 + offset_idx], ld_y[9 + offset_idx], ld_z[9 + offset_idx]]
    )
    ring_base = np.array(
        [ld_x[13 + offset_idx], ld_y[13 + offset_idx], ld_z[13 + offset_idx]]
    )

    # ベクトルを計算
    v1 = index_base - wrist
    v2 = middle_base - wrist
    v3 = ring_base - wrist

    # 正規化
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)

    # 回転行列の計算（簡易的な方法）
    z_axis = (v1 + v2 + v3) / 3.0
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(v1, v2)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    rotation = R.from_matrix(rot_matrix).as_quat()  # クォータニオンに変換

    return wrist, rotation  # 位置と回転を返す


@dataclass
class HandTrans:
    pos: np.ndarray | list[float]
    rot: np.ndarray | list[float]


class Hand3DTracker:
    NUM_HAND_LANDMARKS = 21  # left, right

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
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)
        self.plt_x, self.plt_y, self.plt_z = (
            [0.0] * self.NUM_HAND_LANDMARKS * 2,
            [0.0] * self.NUM_HAND_LANDMARKS * 2,
            [0.0] * self.NUM_HAND_LANDMARKS * 2,
        )
        (self.plt_point,) = self.ax.plot([self.plt_x], [self.plt_y], [self.plt_z], "ro")

        self.render_3d_hands = False

        self._l_hand_3d_: HandTrans = HandTrans(np.zeros(3), np.zeros(4))
        self._r_hand_3d_: HandTrans = HandTrans(np.zeros(3), np.zeros(4))

        # ARマーカー関連の初期化
        self.trans_ = np.eye(4)
        self._coord_captured = False
        # 座標系を設定
        self.set_coordinate_system()

    def set_coordinate_system(self):
        if self._coord_captured:
            return  # すでに設定されている場合はスキップ

        if self.frame1 is None or self.frame2 is None:
            print("Frames are not captured yet.")
            return
        res = calib_marker_coordinates(
            self.frame1,
            self.frame2,
            self.tri,
            MARKER_ID,
        )
        if res is None:
            return
        self.trans_ = np.linalg.inv(res)
        self._coord_captured = True

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

        self.set_coordinate_system()

        results = hands_l.process(rgb_f1)
        results2 = hands_r.process(rgb_f2)

        found_l = results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0
        found_r = (
            results2.multi_hand_landmarks and len(results2.multi_hand_landmarks) > 0
        )
        if (
            not found_l
            or not found_r
            or len(results.multi_hand_landmarks) != len(results2.multi_hand_landmarks)
        ):
            print("....")
            plt.draw()
            plt.pause(0.001)
            return True

        num_detected = min(
            len(results.multi_hand_landmarks), len(results2.multi_hand_landmarks)
        )
        if num_detected == 2:
            # sort by x coordinate of wrist
            wrist_0 = results.multi_hand_landmarks[0].landmark[0]
            wrist_1 = results.multi_hand_landmarks[1].landmark[0]
            if wrist_0.x > wrist_1.x:
                results.multi_hand_landmarks[0], results.multi_hand_landmarks[1] = (
                    results.multi_hand_landmarks[1],
                    results.multi_hand_landmarks[0],
                )
            wrist_0 = results2.multi_hand_landmarks[0].landmark[0]
            wrist_1 = results2.multi_hand_landmarks[1].landmark[0]
            if wrist_0.x > wrist_1.x:
                results2.multi_hand_landmarks[0], results2.multi_hand_landmarks[1] = (
                    results2.multi_hand_landmarks[1],
                    results2.multi_hand_landmarks[0],
                )

        for i in range(num_detected):
            hand1 = results.multi_hand_landmarks[i]
            hand2 = results2.multi_hand_landmarks[i]
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
            for k, point in enumerate(points_3d):
                self.plt_x[k + i * self.NUM_HAND_LANDMARKS] = point[0]
                self.plt_y[k + i * self.NUM_HAND_LANDMARKS] = point[1]
                self.plt_z[k + i * self.NUM_HAND_LANDMARKS] = point[2]

            wrist, rotation = _landmarks_to_pos_rot(
                self.plt_x,
                self.plt_y,
                self.plt_z,
                offset_idx=i * self.NUM_HAND_LANDMARKS,
            )
            rotation = [0, 0, 0, 1]  # 仮の回転
            wrist_ = self.trans_ @ np.append(wrist, 1.0)  # 同次座標に変換
            wrist = wrist_[:3]  # 元の座標に戻す

            rotation_ = R.from_quat(rotation).as_matrix()
            rotation_ = self.trans_[:3, :3] @ rotation_  # 回転行列を変換
            rotation = R.from_matrix(rotation_).as_quat()

            if i == 0:
                self._l_hand_3d_ = HandTrans(wrist, rotation)
            else:
                self._r_hand_3d_ = HandTrans(wrist, rotation)

            print(f"Hand {i}: Pos {wrist}, Rot {rotation} (quat x,y,z,w)")
        self.plt_point.set_data(
            [self._l_hand_3d_.pos[0], self._r_hand_3d_.pos[0]],
            [self._l_hand_3d_.pos[1], self._r_hand_3d_.pos[1]],
        )
        self.plt_point.set_3d_properties(  # type: ignore
            [self._l_hand_3d_.pos[2], self._r_hand_3d_.pos[2]]
        )

        plt.draw()
        plt.pause(0.001)
        self._render(results, results2)

        return True

    def close(self):
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = Hand3DTracker(cam_l_idx=0, cam_r_idx=2)
    while True:
        tracker._capture()
        if not tracker.update():
            break
    tracker.close()
