#!/usr/bin/env python3
"""
Test script for dual SO101 robot environment
This script loads and visualizes the dual robot MJCF scene
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # Load the dual robot model
    try:
        model = mujoco.MjModel.from_xml_path("dual_so101_scene.mjcf")
        data = mujoco.MjData(model)
        print("✓ Successfully loaded dual SO101 robot model")
        print(f"Model has {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
        
        # Print joint names to understand the structure
        print("\nJoint names:")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                print(f"  Joint {i}: {joint_name}")
        
        print("\nActuator names:")
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                print(f"  Actuator {i}: {actuator_name}")
        
        # Initialize the simulation
        mujoco.mj_step(model, data)
        
        # Set some initial joint positions for demo
        if model.nu >= 12:  # Ensure we have both robots
            # Robot 1 (first 6 joints) - slight pose
            data.ctrl[0] = 0.2   # shoulder_pan
            data.ctrl[1] = -0.3  # shoulder_lift
            data.ctrl[2] = 0.5   # elbow_flex
            data.ctrl[3] = -0.2  # wrist_flex
            data.ctrl[4] = 0.1   # wrist_roll
            data.ctrl[5] = 0.0   # gripper
            
            # Robot 2 (next 6 joints) - different pose
            data.ctrl[6] = -0.2  # robot2_shoulder_pan
            data.ctrl[7] = 0.3   # robot2_shoulder_lift
            data.ctrl[8] = -0.5  # robot2_elbow_flex
            data.ctrl[9] = 0.2   # robot2_wrist_flex
            data.ctrl[10] = -0.1 # robot2_wrist_roll
            data.ctrl[11] = 0.5  # robot2_gripper
        
        # Launch the viewer
        print("\nLaunching MuJoCo viewer...")
        print("Controls:")
        print("  - Mouse: rotate view")
        print("  - Scroll: zoom")
        print("  - Double-click: select body")
        print("  - Ctrl+mouse: translate view")
        print("  - Press ESC or close window to exit")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Simulation loop
            start_time = time.time()
            while viewer.is_running():
                step_start = time.time()
                
                # Add some dynamic motion for demonstration
                t = time.time() - start_time
                if model.nu >= 12:
                    # Robot 1 - slow sinusoidal motion
                    data.ctrl[0] = 0.3 * np.sin(0.5 * t)
                    data.ctrl[1] = -0.3 + 0.2 * np.sin(0.3 * t)
                    data.ctrl[2] = 0.5 + 0.3 * np.sin(0.7 * t)
                    
                    # Robot 2 - different motion pattern
                    data.ctrl[6] = -0.3 * np.sin(0.4 * t)
                    data.ctrl[7] = 0.3 + 0.2 * np.sin(0.6 * t)
                    data.ctrl[8] = -0.5 + 0.3 * np.sin(0.8 * t)
                
                # Step the physics simulation
                mujoco.mj_step(model, data)
                
                # Synchronize with real-time
                viewer.sync()
                
                # Control frame rate
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
    except Exception as e:
        print(f"Error loading or running model: {e}")
        print("Make sure you have MuJoCo installed and the MJCF files are properly configured.")
        return False
        
    return True

if __name__ == "__main__":
    main()
