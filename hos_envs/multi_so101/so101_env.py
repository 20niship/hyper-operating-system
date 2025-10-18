from pathlib import Path
import mujoco
import numpy as np

MODEL = Path(__file__).parent.parent / "SO101/dual_so101_scene.xml"


class SO101MultiEnv:
    def __init__(self, n_substeps=20):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        self.viewer = None

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)  # type: ignore
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        return obs, reward, done, info

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)  # type: ignore
        return self._get_obs()

    def render(self, mode="human"):
        # Linux/Windowsの場合: ネイティブビューアを使用
        import mujoco.viewer as viewer

        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def _get_obs(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

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
    action = env.data.ctrl
    print("Action shape:", action.shape)
    for _ in range(1000):
        action = np.random.uniform(-100, 100, size=action.shape)
        obs, reward, done, info = env.step(action)
        print(obs)
        env.render()
    env.close()
