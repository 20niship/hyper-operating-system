import cv2
import time
import numpy as np
import glob
import json
from pathlib import Path


def capture_images(video0=0, video1=1):
    if Path("images").exists() and len(list(Path("images").iterdir())) > 40:
        print(
            "images フォルダが既に存在し、画像が含まれています。キャプチャをスキップします。"
        )
        return

    if not Path("images").exists():
        print(
            "images フォルダが存在しません。キャリブレーション画像を保存するために作成します。"
        )
        Path("images").mkdir()

    cap1 = cv2.VideoCapture(video0)  # 左カメラ
    cap2 = cv2.VideoCapture(video1)  # 右カメラ
    print("cap.isOpened()", cap1.isOpened())
    print("cap2.isOpened()", cap2.isOpened())
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore

    Path("images").mkdir(exist_ok=True)
    img_count = 0
    key = -1
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("カメラからの映像取得に失敗しました。")
            continue

        grayL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if key == ord("c"):
            img_count += 1
            cv2.imwrite(f"images/left{img_count:02d}.jpg", frame1)
            cv2.imwrite(f"images/right{img_count:02d}.jpg", frame2)
            time.sleep(1)  # 連続撮影を防ぐために1秒待機
            print(f"Captured pair {img_count}")

        combined = np.hstack((frame1, frame2))
        if combined.shape[1] > 2000:
            combined = cv2.resize(combined, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Stereo Cameras (Press 'c' to capture, 'q' to quit)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if img_count >= 40:
            print("十分な画像がキャプチャされました。終了します。")
            break


# ステレオカメラキャリブレーションクラス
class StereoCalibrator:
    def __init__(self, chessboard_size=(8, 5), square_size=0.025):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.mtx1 = None
        self.dist1 = None
        self.mtx2 = None
        self.dist2 = None
        self.R = None
        self.T = None
        self.P1 = None
        self.P2 = None

    def calibrate(self, left_images, right_images):
        print("Starting calibration...")
        # 両カメラキャリブレーション
        self.mtx1, self.dist1 = self._calibrate_single(left_images)
        self.mtx2, self.dist2 = self._calibrate_single(right_images)
        self._stereo_calibrate()
        print("Calibration completed.")

    def _calibrate_single(self, images_names):
        # 単眼キャリブレーション
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rows, columns = self.chessboard_size[1], self.chessboard_size[0]
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp *= self.square_size
        images = [cv2.imread(imname, 1) for imname in images_names]
        imgpoints, objpoints = [], []
        for frame in images:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
        width, height = images[0].shape[1], images[0].shape[0]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            (width, height),
            None,  # type: ignore
            None,  # type: ignore
        )
        return mtx, dist

    def _stereo_calibrate(self):
        # ステレオキャリブレーション
        left_images = sorted(glob.glob("images/left*.jpg"))
        right_images = sorted(glob.glob("images/right*.jpg"))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        rows, columns = self.chessboard_size[1], self.chessboard_size[0]
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp *= self.square_size
        c1_images = [cv2.imread(im, 1) for im in left_images]
        c2_images = [cv2.imread(im, 1) for im in right_images]
        imgpoints_left, imgpoints_right, objpoints = [], [], []
        for frame1, frame2 in zip(c1_images, c2_images):
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
            c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)
            if c_ret1 and c_ret2:
                corners1 = cv2.cornerSubPix(
                    gray1, corners1, (11, 11), (-1, -1), criteria
                )
                corners2 = cv2.cornerSubPix(
                    gray2, corners2, (11, 11), (-1, -1), criteria
                )
                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)
        width, height = c1_images[0].shape[1], c1_images[0].shape[0]
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            self.mtx1,  # type: ignore
            self.dist1,  # type: ignore
            self.mtx2,  # type: ignore
            self.dist2,  # type: ignore
            (width, height),
            criteria=criteria,
            flags=flags,  # type: ignore
        )
        self.R = R
        self.T = T

        # 射影行列計算
        RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        self.P1 = self.mtx1 @ RT1
        RT2 = np.concatenate([self.R, self.T], axis=-1)  # type: ignore
        self.P2 = self.mtx2 @ RT2

    def save(self, filepath):
        # キャリブレーション結果保存
        data = {
            "mtx1": self.mtx1.tolist() if self.mtx1 is not None else None,
            "dist1": self.dist1.tolist() if self.dist1 is not None else None,
            "mtx2": self.mtx2.tolist() if self.mtx2 is not None else None,
            "dist2": self.dist2.tolist() if self.dist2 is not None else None,
            "R": self.R.tolist() if self.R is not None else None,
            "T": self.T.tolist() if self.T is not None else None,
            "P1": self.P1.tolist() if self.P1 is not None else None,
            "P2": self.P2.tolist() if self.P2 is not None else None,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath) -> bool:
        if not Path(filepath).exists():
            print(f"File {filepath} does not exist.")
            return False

        # キャリブレーション結果読み込み
        with open(filepath, "r") as f:
            data = json.load(f)
        self.mtx1 = np.array(data["mtx1"]) if data["mtx1"] is not None else None
        self.dist1 = np.array(data["dist1"]) if data["dist1"] is not None else None
        self.mtx2 = np.array(data["mtx2"]) if data["mtx2"] is not None else None
        self.dist2 = np.array(data["dist2"]) if data["dist2"] is not None else None
        self.R = np.array(data["R"]) if data["R"] is not None else None
        self.T = np.array(data["T"]) if data["T"] is not None else None
        self.P1 = np.array(data["P1"]) if data["P1"] is not None else None
        self.P2 = np.array(data["P2"]) if data["P2"] is not None else None
        return True

    def get_projection_matrices(self):
        return self.P1, self.P2

    def triangulate(self, point1: list[float], point2: list[float]) -> np.ndarray:
        # 三角測量
        if self.P1 is None or self.P2 is None:
            raise ValueError("Projection matrices are not available.")
        return self._DLT(self.P1, self.P2, point1, point2)

    def _DLT(self, P1, P2, point1, point2):
        # DLT三角測量
        A = [
            point1[1] * P1[2, :] - P1[1, :],
            P1[0, :] - point1[0] * P1[2, :],
            point2[1] * P2[2, :] - P2[1, :],
            P2[0, :] - point2[0] * P2[2, :],
        ]
        A = np.array(A).reshape((4, 4))
        from scipy import linalg

        U, s, Vh = linalg.svd(A.T @ A, full_matrices=False)
        return Vh[3, 0:3] / Vh[3, 3]


if __name__ == "__main__":
    # キャリブレーション実行例
    left_images = sorted(glob.glob("images/left*.jpg"))
    right_images = sorted(glob.glob("images/right*.jpg"))
    calibrator = StereoCalibrator(chessboard_size=(8, 5), square_size=0.025)
    calibrator.calibrate(left_images, right_images)
    calibrator.save("stereo_calibration.json")

    # キャリブレーション結果読み込み例
    calibrator2 = StereoCalibrator()
    calibrator2.load("stereo_calibration.json")
    P1, P2 = calibrator2.get_projection_matrices()

    # 三角測量例
    point1 = (150, 120)  # 左画像の対応点
    point2 = (130, 115)  # 右画像の対応点
    point_3d = calibrator2.DLT(P1, P2, point1, point2)
    print("3D Point:", point_3d)
