#!/usr/bin/env python3
"""
test_ik_node.py - Test script for IK node
Publishes test end-effector poses and checks the IK node response
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hos_core.topic import Publisher, Subscriber


def test_ik_node():
    """Test the IK node by publishing test poses"""
    
    print("=== IK Node Test ===")
    print("NOTE: Make sure the broker and ik_node are running!")
    print("Start broker with: python start_broker.py")
    print("Start IK node with: python ik_node.py --urdf hos_envs/multi_so101/SO101/so101_new_calib.urdf")
    print("====================\n")
    
    # Create publisher for end-effector pose
    # Use str type to send raw list data
    pose_publisher = Publisher("/hand/left", str)
    
    # Create subscribers for joint angles
    joint_angles = {}
    
    def create_joint_callback(joint_idx):
        def callback(message):
            if isinstance(message, dict):
                angle = message.get("data", message)
            else:
                angle = message
            joint_angles[joint_idx] = float(angle)
        return callback
    
    joint_subscribers = []
    for i in range(5):  # SO-101 has 5 actuated joints
        sub = Subscriber(f"/joint{i+1}", create_joint_callback(i+1), float)
        sub.start()
        joint_subscribers.append(sub)
    
    print("Publishers and subscribers initialized")
    time.sleep(1.0)  # Wait for connections
    
    try:
        # Test 1: Publish a simple pose
        print("\n=== Test 1: Simple forward position ===")
        position = [0.3, 0.0, 0.3]  # 30cm forward, 30cm up
        quaternion = R.from_euler('xyz', [0, 0, 0]).as_quat()  # xyzw format
        
        pose = position + quaternion.tolist()
        print(f"Publishing pose: position={position}, quaternion={quaternion}")
        
        # Publish as string representation of list
        pose_publisher.publish(str(pose))
        time.sleep(1.0)  # Wait for IK computation
        
        if joint_angles:
            print(f"Received joint angles: {joint_angles}")
        else:
            print("No joint angles received")
        
        # Test 2: Publish another pose
        print("\n=== Test 2: Different position ===")
        position = [0.25, 0.1, 0.35]
        quaternion = R.from_euler('xyz', [0, np.pi/4, 0]).as_quat()
        
        pose = position + quaternion.tolist()
        print(f"Publishing pose: position={position}, quaternion={quaternion}")
        
        # Publish as string representation of list
        pose_publisher.publish(str(pose))
        time.sleep(1.0)
        
        if joint_angles:
            print(f"Received joint angles: {joint_angles}")
        else:
            print("No joint angles received")
        
        # Test 3: Rapid updates
        print("\n=== Test 3: Rapid updates (10Hz) ===")
        for i in range(10):
            t = i * 0.1
            position = [0.3 + 0.05 * np.sin(t * 2 * np.pi), 0.0, 0.3 + 0.05 * np.cos(t * 2 * np.pi)]
            quaternion = R.from_euler('xyz', [0, 0, t]).as_quat()
            
            pose = position + quaternion.tolist()
            # Publish as string representation of list
            pose_publisher.publish(str(pose))
            time.sleep(0.1)
        
        print(f"Final joint angles: {joint_angles}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        pose_publisher.close()
        for sub in joint_subscribers:
            sub.close()
        print("Test completed")


if __name__ == "__main__":
    test_ik_node()
