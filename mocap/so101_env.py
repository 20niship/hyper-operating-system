from pathlib import Path
import mujoco_py

MODEL = Path(__file__).parent / "SO101/dual_so101_scene.xml"


class SO101MultiEnv:
    def __init__(self, n_substeps=20):
        self.model = mujoco_py.load_model_from_path(str(MODEL))
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
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
            self.viewer = mujoco_py.MjViewer(self.sim)
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
    env = SO101MultiEnv(model_path=str(MODEL))
    obs = env.reset()
    for _ in range(1000):
        action = env.sim.data.ctrl + 0.01  # Dummy action
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()
