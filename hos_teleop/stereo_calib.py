import matplotlib

matplotlib.use("TkAgg")  # Macの場合は "MacOSX" に変更してください

from matplotlib import pyplot as plt
import glob
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from mocap.stereo_tri import StereoCalibrator, capture_images

CAP_IDX1 = 0
CAP_IDX2 = 1

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
TARGET_ID = 0  # トラッキングしたいマーカーID（必要に応じて変更）


def calib():
    capture_images(CAP_IDX1, CAP_IDX2)

    calib = StereoCalibrator((8, 5), 0.025)
    if not Path("stereo_calib.json").exists():
        imags_1 = sorted(glob.glob("images/left*.jpg"))
        imags_2 = sorted(glob.glob("images/right*.jpg"))
        calib.calibrate(imags_1, imags_2)
        calib.save("stereo_calib.json")

    calib.load("stereo_calib.json")

    capL = cv2.VideoCapture(CAP_IDX1)
    capR = cv2.VideoCapture(CAP_IDX2)

    print("capL.isOpened()", capL.isOpened())
    print("capR.isOpened()", capR.isOpened())

    #     capL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
    #     capR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("カメラが開けませ。デバイス番号を確認してください。")

    print("start capturing...")


def detect_aruco_marker(frame, target_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == target_id:
                # マーカーの中心座標を計算
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                return np.array([cx, cy], dtype=np.float32)
    return None


def stereo_calib():
    capL = cv2.VideoCapture(CAP_IDX1)
    capR = cv2.VideoCapture(CAP_IDX2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    x, y, z = 0, 0, 0
    (point,) = ax.plot([x], [y], [z], "ro")

    Path("./images2").mkdir(exist_ok=True)
    I = 0

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            break

        # save images
        cv2.imwrite(f"./images2/left_{I:02d}.jpg", frameL)
        cv2.imwrite(f"./images2/right_{I:02d}.jpg", frameR)
        I += 1

        ptL = detect_aruco_marker(frameL, TARGET_ID)
        ptR = detect_aruco_marker(frameR, TARGET_ID)
        # 検出したマーカーを枠で描画
        if ptL is not None:
            cv2.circle(frameL, (int(ptL[0]), int(ptL[1])), 10, (0, 255, 0), 5)
        if ptR is not None:
            cv2.circle(frameR, (int(ptR[0]), int(ptR[1])), 10, (0, 255, 0), 5)

        combined = np.hstack((frameL, frameR))
        if combined.shape[1] > 2000:
            combined = cv2.resize(combined, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Stereo Cameras (Press 'ESC' to quit)", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

        continue
        if ptL is None or ptR is None:
            print("Marker not detected", frameL.shape, frameR.shape)
            continue

        pts = calib.triangulate([ptL[0], ptL[1]], [ptR[0], ptR[1]])
        x, y, z = pts[0], pts[1], pts[2]
        print(ptL, ptR, "->", x, y, z)

        point.set_data([x], [y])
        point.set_3d_properties([z])  # type: ignore
        plt.draw()
        plt.pause(0.01)

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


# calib()
stereo_calib()
