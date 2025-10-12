from mocap.hand_tracking import Hand3DTracker

tracker = Hand3DTracker(cam_l_idx=0, cam_r_idx=2)
while True:
    tracker._capture()
    if not tracker.update():
        break
tracker.close()
