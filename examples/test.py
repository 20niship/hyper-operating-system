import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

urdf_path = "./SO101/so101_new_calib.urdf"
robot = p.loadURDF(urdf_path, useFixedBase=True)
num_joints = p.getNumJoints(robot)
chain = Chain.from_urdf_file(urdf_path)
end_effector_link = 6

t = 0.0
while True:
    # 目標位置（円運動）
    target_pos = [0.3, 0.1 * math.cos(t), 0.1 * math.sin(t) + 0.2]
    target_quat = p.getQuaternionFromEuler(
        [0.6 * math.sin(t * 2), 0.5 * math.cos(t * 2), np.pi / 2]
    )  # (roll, pitch, yaw) → quat

    # 逆運動学（PyBullet内部ソルバ）
    joint_angles = p.calculateInverseKinematics(
        robot,
        end_effector_link,
        target_pos,
        target_quat,
        maxNumIterations=50,
        residualThreshold=1e-4,
    )

    for i in range(p.getNumJoints(robot)):
        if i >= len(joint_angles):
            continue
        p.setJointMotorControl2(
            robot, i, p.POSITION_CONTROL, targetPosition=joint_angles[i], force=500
        )

    # デバッグ: エンドエフェクタ位置にマーカーを出す
    p.addUserDebugLine([0, 0, 0], target_pos, [1, 0, 0], 3, lifeTime=0.1)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    t += 0.01
