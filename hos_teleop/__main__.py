import matplotlib

matplotlib.use("TkAgg")  # Macの場合は "MacOSX" に変更してください

import glob
from pathlib import Path
import argparse
import cv2
from .mocap.stereo_tri import StereoCalibrator, capture_images
from hos_teleop.mocap.hand_tracking import Hand3DTracker
from hos_core.topic import Publisher


CAP_IDX1 = 0
CAP_IDX2 = 1

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
TARGET_ID = 0  # トラッキングしたいマーカーID（必要に応じて変更）


def _calib_stereo_cam_main(cap_idx1=CAP_IDX1, cap_idx2=CAP_IDX2):
    """
    uv run hos_teleop --calib_stereo_cam
    """
    capture_images(cap_idx1, cap_idx2)

    calib = StereoCalibrator((8, 5), 0.025)
    if not Path("stereo_calib.json").exists():
        imags_1 = sorted(glob.glob("images/left*.jpg"))
        imags_2 = sorted(glob.glob("images/right*.jpg"))
        calib.calibrate(imags_1, imags_2)
        calib.save("stereo_calib.json")
    calib.load("stereo_calib.json")
    print("DONE")


def _pub_hand_pos_main(cap_idx1=CAP_IDX1, cap_idx2=CAP_IDX2):
    tracker = Hand3DTracker(cap_idx1, cap_idx2)
    pub_l = Publisher("/hand/left", list[float])
    pub_r = Publisher("/hand/right", list[float])

    while True:
        tracker._capture()
        if not tracker.update():
            break
        t_l = tracker._l_hand_3d_
        t_r = tracker._r_hand_3d_
        if t_l:
            l_list = [
                t_l.pos[0],
                t_l.pos[1],
                t_l.pos[2],
                t_l.rot[0],
                t_l.rot[1],
                t_l.rot[2],
                t_l.rot[3],
            ]
            pub_l.publish(l_list)
        if t_r:
            r_list = [
                t_r.pos[0],
                t_r.pos[1],
                t_r.pos[2],
                t_r.rot[0],
                t_r.rot[1],
                t_r.rot[2],
                t_r.rot[3],
            ]
            pub_r.publish(r_list)
    tracker.close()


def _pub_ar_marker_pos_main():
    from hos_teleop.mocap.hand_tracking import Hand3DTracker

    tracker = Hand3DTracker(cam_l_idx=0, cam_r_idx=2)

    while True:
        tracker._capture()
        if not tracker.update():
            break

        t_l = tracker._l_hand_3d_
        t_r = tracker._r_hand_3d_

        print("Left Hand Position:", t_l.pos)
        print("Right Hand Position:", t_r.pos)

    tracker.close()


if __name__ == "__main__":
    # parse args with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib_stereo_cam", action="store_true", help="Calibrate stereo camera"
    )
    parser.add_argument(
        "--hand_tracking", action="store_true", help="Test calibration result"
    )
    parser.add_argument(
        "--ar_marker_tracking", action="store_true", help="Test AR marker tracking"
    )
    parser.add_argument(
        "--cap1", type=int, default=CAP_IDX1, help="Camera index for left camera"
    )
    parser.add_argument(
        "--cap2", type=int, default=CAP_IDX2, help="Camera index for right camera"
    )
    args = parser.parse_args()

    cap1 = args.cap1
    cap2 = args.cap2

    if args.calib_stereo_cam:
        _calib_stereo_cam_main(cap1, cap2)
    elif args.hand_tracking:
        _pub_hand_pos_main()
    elif args.ar_marker_tracking:
        _pub_ar_marker_pos_main()
