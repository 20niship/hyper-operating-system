from hos_teleop.mocap.hand_tracking import Hand3DTracker
from hos_envs.multi_so101.so101_env import SO101MultiEnv
from hos_teleop.mocap.ik import URDFInverseKinematics


def main():
    tracker = Hand3DTracker(cam_l_idx=0, cam_r_idx=2)
    ik_l = URDFInverseKinematics()
    # ik_r = URDFInverseKinematics()
    # env = SO101MultiEnv()
    # obs = env.reset()
    # action = env.data.ctrl
    action = [0.0] * 12

    while True:
        tracker._capture()
        if not tracker.update():
            break

        t_l = tracker._l_hand_3d_
        t_r = tracker._r_hand_3d_
        j_l = ik_l.compute_ik(t_l.pos, [0, 0, 0, 1])
        # j_r = ik_r.compute_ik(t_r.pos, t_r.rot)
        action[0:6] = j_l

        # action[6:12] = j_r
        # obs, reward, done, info = env.step(action)
        # env.render()

    tracker.close()


if __name__ == "__main__":
    main()
