from pathlib import Path
import mujoco

MODEL = Path(__file__).parent.parent / "SO101/dual_so101_scene.xml"


class SO101MultiEnv:
    def __init__(self, n_substeps=20):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))  # type: ignore
        self.sim = mujoco.MjData(self.model)  # type: ignore
        self.viewer = None

    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.sim.reset()
        return self._get_obs()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.sim)  # type: ignore
        self.viewer.render()

    def _get_obs(self):
        return self.sim.data.qpos.flatten()

    def _compute_reward(self):
        # Placeholder for reward computation logic
        return 0.0

    def _check_done(self):
        # Placeholder for termination condition
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer = None


if __name__ == "__main__":
    env = SO101MultiEnv()
    obs = env.reset()
    for _ in range(1000):
        action = env.sim.data.ctrl + 0.01  # Dummy action
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

#         self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
#         self.data = mujoco.MjData(self.model)

#         # ビューアを初期化（ヘルパー関数が全て処理）
#         if self.viewer is None:
#             self.viewer = create_mujoco_viewer(self.model, self.data, self.is_gui)

#         # Set gravity
#         self.model.opt.gravity[:] = [0, 0, -9.81]
#         self.model.opt.timestep = 0.01

#         # Initial step to stabilize
#         mujoco.mj_step(self.model, self.data)

#     def _setup_indices(self):
#         """Setup indices for joints, body, and sensors"""
#         # Joint indices
#         joint_names = [f"leg{i}" for i in range(1, 7)]
#         self.joint_indices = []
#         for joint_name in joint_names:
#             joint_id = mujoco.mj_name2id(
#                 self.model,
#                 mujoco.mjtObj.mjOBJ_JOINT,
#                 joint_name,
#             )
#             if joint_id >= 0:
#                 self.joint_indices.append(joint_id)

#         # Body index
#         self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main")

#         # Sensor indices
#         self.gyro_sensor_id = mujoco.mj_name2id(
#             self.model,
#             mujoco.mjtObj.mjOBJ_SENSOR,
#             "gyro_sensor",
#         )
#         self.acc_sensor_id = mujoco.mj_name2id(
#             self.model,
#             mujoco.mjtObj.mjOBJ_SENSOR,
#             "acc_sensor",
#         )

#         # Step obstacle body and geom indices
#         self.step_body_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_BODY, "step_obstacle"
#         )
#         self.step_geom_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_GEOM, "step_geom"
#         )

#         # MJCFモデルからgoal位置を取得
#         goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
#         self.target_pos = self.model.body_pos[goal_body_id].copy()

#     def _update_step_height(self, difficulty):
#         """Update step height based on difficulty"""
#         # Height ranges from 0.05m to 0.17m based on difficulty
#         z = (1 - difficulty) * 0.01 + difficulty * 0.06

#         assert self.step_body_id >= 0, "Step body not found"
#         self.model.body_pos[self.step_body_id][2] = z

#         # Update step geom size (half-height)
#         assert self.step_geom_id >= 0, "Step geom not found"
#         self.model.geom_size[self.step_geom_id][2] = z

#     def _update_obs(self):
#         """Update observation from simulation state"""
#         # Position and orientation
#         assert self.body_id >= 0, "Main body not found"
#         self.pos = self.data.xpos[self.body_id].copy()
#         quat = self.data.xquat[self.body_id].copy()
#         self.rot = quat_to_euler_scipy(quat)
#         # Correct for initial orientation (y-axis is reversed in mujoco)
#         self.rot[2] = _to_2pi(self.rot[2] + np.pi)

#         # Joint states
#         assert len(self.joint_indices) == 6, "Expected 6 joints"
#         for i, joint_id in enumerate(self.joint_indices):
#             # Joint position (accounting for free joint)
#             joint_qpos_idx = joint_id
#             joint_qvel_idx = joint_id - 1

#             if joint_qpos_idx < len(self.data.qpos):
#                 self.joint_pos[i] = _to_2pi(self.data.qpos[joint_qpos_idx])

#             if joint_qvel_idx < len(self.data.qvel):
#                 self.joint_vel[i] = self.data.qvel[joint_qvel_idx]

#             if joint_qvel_idx < len(self.data.qvel) and i < 6:
#                 self.joint_acc[i] = self.data.qacc[joint_qvel_idx]

#         # Sensor data
#         assert self.gyro_sensor_id >= 0, "Gyro sensor not found"
#         assert self.acc_sensor_id >= 0, "Acc sensor not found"
#         assert self.gyro_sensor_id + 2 < len(self.data.sensordata), "Gyro incomplete"
#         assert self.acc_sensor_id + 2 < len(self.data.sensordata), "Acc incomplete"

#         self.gyro = self.data.sensordata[
#             self.gyro_sensor_id : self.gyro_sensor_id + 3
#         ].copy()

#         self.acc = self.data.sensordata[
#             self.acc_sensor_id : self.acc_sensor_id + 3
#         ].copy()

#         # ローカル座標系から見たゴール方向をcmd_velsにセット
#         dir_vec = self.target_pos[0:2] - self.pos[0:2]
#         yaw = self.rot[2]
#         dir_norm = np.linalg.norm(dir_vec)
#         if dir_norm > 1e-4:
#             dir_vec /= dir_norm
#         cos_yaw = np.cos(-yaw)
#         sin_yaw = np.sin(-yaw)
#         self.cmd_vels[0] = dir_vec[0] * cos_yaw - dir_vec[1] * sin_yaw
#         self.cmd_vels[1] = dir_vec[0] * sin_yaw + dir_vec[1] * cos_yaw

#         # Construct observation vector
#         self.obs = np.concatenate(
#             [
#                 self.rot[0:2],  # roll, pitch (2)
#                 self.gyro,  # 3
#                 self.acc,  # 3
#                 self.joint_pos,  # 6
#                 self.joint_vel,  # 6
#                 self.last_actions,  # 6
#                 self.cmd_vels,  # 2 (x, y、target_posへの方向)
#             ]
#         )

#     def _get_observation(self):
#         """Get current observation"""
#         return self.obs.astype(np.float32)

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
#         """
#         Execute one step in the environment

#         Args:
#             action: Action array of shape (6,)

#         Returns:
#             observation, reward, terminated, truncated, info
#         """
#         action = np.array(action).flatten()
#         assert action.shape == (6,), f"Expected action shape (6,), got {action.shape}"

#         # Execute action
#         reward = self._calc_reward(action)
#         terminated, truncated, add_reward = self._judge_termination()
#         reward += add_reward

#         if self.is_eval:
#             terminated = False

#         self.n_steps += 1
#         self.n_steps_total += 1
#         self.episode_reward += reward

#         obs = self._get_observation()

#         if self.is_gui:
#             self.render()

#         return obs, reward, terminated, truncated, {}

#     def _judge_termination(self) -> Tuple[bool, bool, float]:
#         """Judge if episode should terminate"""
#         terminated = False
#         truncated = False
#         add_reward = 0.0
#         pos = self.pos
#         # rot = self.rot
#         # if abs(pos[1]) > 0.5:
#         #     terminated = True
#         #     add_reward -= 5000
#         # # Check if orientation is too tilted (roll and pitch only)
#         # # Note: rot contains Euler angles where rot[0] is roll, rot[1] is pitch, rot[2] is yaw
#         # if abs(rot[0]) > np.pi / 2 or abs(rot[1]) > np.pi / 2:
#         #     terminated = True
#         #     add_reward -= 5000

#         if np.linalg.norm(pos[0:2] - self.target_pos[0:2]) < 0.3:
#             terminated = True
#             add_reward += 5000

#         if self.n_steps >= self.episode_length:
#             truncated = True

#         return terminated, truncated, add_reward

#     def _calc_reward(self, action: np.ndarray) -> float:
#         """Calculate reward for current step"""
#         action = np.clip(action, -1.0, 1.0)
#         self._step_sim(action)

#         pos = self.pos

#         pos_dist = np.linalg.norm(pos[0:2] - self.target_pos[0:2])
#         pos_rew = self.last_pos_dist - pos_dist  # 前回からの距離変化
#         if self.last_pos_dist < 0:
#             pos_rew = 0.0
#         self.last_pos_dist = pos_dist

#         torque_reward = -1 * np.sum(np.square(action))
#         smoothness_reward = -1 * np.sum(np.square(self.gyro))

#         # Weights
#         position_weight = 20
#         torque_weight = 0.003
#         smoothness_weight = 0.01

#         reward = (
#             position_weight * pos_rew
#             + torque_weight * torque_reward
#             + smoothness_weight * smoothness_reward
#         )

#         if self.log_wandb and self.n_steps_total % 500 == 0:
#             wandb.log(
#                 {
#                     "env/position_reward": position_weight * pos_rew,
#                     "env/torque_reward": torque_weight * torque_reward,
#                     "env/smoothness_reward": smoothness_weight * smoothness_reward,
#                     "env/total_reward": reward,
#                     "env/difficulty": self.difficulty,
#                 }
#             )

#         return reward

#     def _step_sim(self, action: np.ndarray):
#         """Execute action in simulation"""
#         # Invert action (to match convention)
#         action = action * -1.0

#         # Set control inputs
#         for joint_idx in range(6):
#             if joint_idx < len(self.data.ctrl):
#                 self.data.ctrl[joint_idx] = np.clip(action[joint_idx], -10.0, 10.0)

#         # Calculate number of simulation steps
#         simulation_steps = int(self.sim_step_sec / self.model.opt.timestep)
#         if simulation_steps < 1:
#             simulation_steps = 1

#         # Store last action and step simulation
#         self.last_actions = action.copy()
#         mujoco.mj_step(self.model, self.data, nstep=simulation_steps)
#         self._update_obs()

#     def reset(
#         self, seed: int | None = None, options: Dict[str, Any] | None = None
#     ) -> Tuple[np.ndarray, dict]:
#         """Reset environment"""
#         super().reset(seed=seed)

#         # Randomize step height based on difficulty
#         self._update_step_height(self.difficulty)
#         self.difficulty += 0.0005

#         # Reset simulation
#         mujoco.mj_resetData(self.model, self.data)
#         self.model.opt.gravity[:] = [0, 0, -9.81]

#         # Stabilize physics
#         mujoco.mj_step(self.model, self.data, nstep=5)
#         mujoco.mj_rnePostConstraint(self.model, self.data)

#         # Reset state
#         self.episode_reward = 0.0
#         self.n_steps = 0
#         self.last_actions = np.zeros(6)
#         self.last_pos_dist = -1.0

#         # Update observation
#         self._update_obs()

#         # 足の位置をリセット
#         for i in range(6):
#             joint_id = self.joint_indices[i]
#             joint_qpos_idx = joint_id
#             assert joint_qpos_idx >= 0, f"Joint {i} not found"
#             assert joint_qpos_idx < len(self.data.qpos), f"J{i} index out of range"

#             # 各関節を0~2πの範囲でランダムに設定
#             self.data.qpos[joint_qpos_idx] = random.uniform(0, 2 * np.pi)

#         obs = self._get_observation()
#         return obs, {}

#     def render(self, mode="human"):
#         """Render the environment"""
#         if self.is_gui:
#             return render_mujoco_viewer(self.viewer, "track")
#         return []

#     def close(self):
#         """Close the environment"""
#         # ビューアのクリーンアップ
#         close_mujoco_viewer(self.viewer)
#         self.viewer = None

#         super().close()

#     @property
#     def num_envs(self):
#         return self.cfg.num_envs
