import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation as R

p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


class URDFInverseKinematics:
    def __init__(self):
        urdf_path = "./SO101/so101_new_calib.urdf"
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        self.end_effector_link = 6

        # 目標位置（円運動）
        self.target_pos = [0.3, 0.1, 0.3]
        self.target_quat = p.getQuaternionFromEuler([0, 0, 0])  # (roll, pitch, yaw)

    def compute_ik(
        self,
        target_pos: list[float] | np.ndarray,
        target_quat: list[float] | np.ndarray,
    ):
        assert len(target_pos) == 3
        assert len(target_quat) == 4
        self.target_pos = target_pos
        # self.target_quat = target_quat

        joint_angles = p.calculateInverseKinematics(
            self.robot,
            self.end_effector_link,
            self.target_pos,
            self.target_quat,
            maxNumIterations=50,
            residualThreshold=1e-4,
        )

        for i in range(p.getNumJoints(self.robot)):
            if i >= len(joint_angles):
                continue
            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=5000,
            )

        p.addUserDebugLine([0, 0, 0], self.target_pos, [1, 0, 0], 3, lifeTime=0.1)
        p.stepSimulation()
        return joint_angles
