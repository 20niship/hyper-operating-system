import cv2
import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from stereo_calib import StereoCalibrator


@dataclass
class _ARMarkerUV:
    center: list[float]
    x_axis: list[float]
    y_axis: list[float]


def _get_coord_uv_pos(frame: np.ndarray, target_id: int) -> Optional[_ARMarkerUV]:
    aruco_dict_type = cv2.aruco.DICT_4X4_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    idx_ = filter(lambda x: x == target_id, ids.flatten())
    if idx_ is None:
        return None
    idx = list(idx_)[0]
    corner = corners[idx][0]
    # x, Y座標を取得
    print(corner)
    center = np.mean(corner, axis=0)
    x_, y_ = int(center[0]), int(center[1])
    return _ARMarkerUV(
        center=[x_, y_],
        x_axis=[
            int((corner[0][0] + corner[1][0]) / 2),
            int((corner[0][1] + corner[1][1]) / 2),
        ],
        y_axis=[
            int((corner[1][0] + corner[2][0]) / 2),
            int((corner[1][1] + corner[2][1]) / 2),
        ],
    )


def calib_marker_coordinates(
    frame1: np.ndarray,
    frame2: np.ndarray,
    calib: StereoCalibrator,
    marker_id: int = 0,
) -> np.ndarray:
    uv_1 = _get_coord_uv_pos(frame1, marker_id)
    uv_2 = _get_coord_uv_pos(frame2, marker_id)
    assert uv_1 is not None and uv_2 is not None

    center = calib.triangulate(uv_1.center, uv_2.center)
    x_axis = calib.triangulate(uv_1.x_axis, uv_2.x_axis)
    y_axis = calib.triangulate(uv_1.y_axis, uv_2.y_axis)
    z_axis = np.cross(x_axis - center, y_axis - center)

    # x, y,z軸から変換行列を計算
    rot_matrix = np.column_stack(
        (
            (x_axis - center) / np.linalg.norm(x_axis - center),
            (y_axis - center) / np.linalg.norm(y_axis - center),
            z_axis / np.linalg.norm(z_axis),
        )
    )
    rot = R.from_matrix(rot_matrix)
    return np.column_stack((rot.as_quat(), center))


