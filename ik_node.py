#!/usr/bin/env python3
"""
ik_node.py - Inverse Kinematics Node
Subscribes to end-effector position and quaternion orientation,
computes inverse kinematics, and publishes joint angles.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List
from scipy.spatial.transform import Rotation as R

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hos_core.topic import Publisher, Subscriber

try:
    import placo
    USE_PLACO = True
except ImportError:
    try:
        from ikpy.chain import Chain
        USE_PLACO = False
    except ImportError:
        raise ImportError("Either placo or ikpy must be installed for IK computation")


class IKNode:
    """Inverse Kinematics Node using placo or ikpy library"""
    
    def __init__(
        self,
        urdf_path: str,
        subscribe_topic: str = "/hand/left",
        publish_prefix: str = "joint",
        publish_suffix: str = "",
        base_orientation: Optional[List[float]] = None,
        active_links_mask: Optional[List[bool]] = None
    ):
        """
        Initialize IK Node
        
        Args:
            urdf_path: Path to URDF file
            subscribe_topic: Topic name to subscribe for end-effector pose
            publish_prefix: Prefix for joint topic names (default: "joint")
            publish_suffix: Suffix for joint topic names (default: "")
            base_orientation: Base link rotation matrix as 9 elements [r11, r12, r13, r21, r22, r23, r31, r32, r33]
            active_links_mask: Mask for active links in IK computation
        """
        self.subscribe_topic = subscribe_topic
        self.publish_prefix = publish_prefix
        self.publish_suffix = publish_suffix
        self.active_links_mask = active_links_mask
        self.use_placo = USE_PLACO
        
        # Load URDF and create kinematic solver
        print(f"Loading URDF from: {urdf_path}")
        print(f"Using IK library: {'placo' if self.use_placo else 'ikpy'}")
        
        if self.use_placo:
            # Initialize placo robot model
            self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)
            
            # Apply base orientation if specified
            if base_orientation is not None and len(base_orientation) == 9:
                rot_matrix = np.array(base_orientation).reshape(3, 3)
                # Create transformation matrix
                T_base = np.eye(4)
                T_base[:3, :3] = rot_matrix
                self.robot.set_T_world_frame(T_base)
            
            # Get joint names and count
            self.joint_names = [name for name in self.robot.joint_names() if not name.startswith("universe")]
            self.num_joints = len(self.joint_names)
            
            # Initialize solver
            self.solver = placo.KinematicsSolver(self.robot)
            
            # Store initial joint state for warm starting
            self.last_joint_angles = None
            
            print(f"Loaded robot with {len(self.robot.joint_names())} joints")
            print(f"Number of actuated joints: {self.num_joints}")
        else:
            # Use ikpy as fallback
            self.chain = Chain.from_urdf_file(urdf_path)
            
            # Apply base orientation if specified
            if base_orientation is not None and len(base_orientation) == 9:
                rot_matrix = np.array(base_orientation).reshape(3, 3)
                # Update base link orientation in the chain
                if len(self.chain.links) > 0:
                    self.chain.links[0].orientation = rot_matrix
            
            # Set active links mask if not provided
            # Automatically exclude fixed links for better performance
            if self.active_links_mask is None:
                self.active_links_mask = [link.joint_type in ["revolute", "prismatic"] for link in self.chain.links]
            
            # Get number of actuated joints (excluding fixed joints)
            self.num_joints = sum(self.active_links_mask)
            
            # Store initial joint angles for warm starting
            self.last_joint_angles = None
            
            print(f"Loaded kinematic chain with {len(self.chain.links)} links")
            print(f"Number of actuated joints: {self.num_joints}")
        
        # Create publishers for each joint
        self.joint_publishers = []
        for i in range(self.num_joints):
            topic_name = f"/{self.publish_prefix}{i+1}{self.publish_suffix}"
            pub = Publisher(topic_name, float)
            self.joint_publishers.append(pub)
            print(f"Created publisher: {topic_name}")
        
        # Create subscriber for end-effector pose
        self.subscriber = Subscriber(
            self.subscribe_topic,
            self._pose_callback,
            list
        )
        
        # Store last received pose
        self.last_pose = None
        self.processing = False
        
        print(f"Created subscriber: {self.subscribe_topic}")
        print("IK Node initialized successfully")
    
    def _pose_callback(self, message):
        """
        Callback for end-effector pose messages
        Expected format: [x, y, z, qw, qx, qy, qz] or [x, y, z, qx, qy, qz, qw]
        """
        if self.processing:
            return
        
        try:
            if isinstance(message, dict):
                data = message.get("data", message)
            else:
                data = message
            
            # Handle string-encoded lists
            if isinstance(data, str):
                import ast
                data = ast.literal_eval(data)
            
            if not isinstance(data, (list, tuple)) or len(data) < 7:
                print(f"Invalid pose message format: {message}")
                return
            
            # Extract position and quaternion
            position = np.array(data[0:3], dtype=float)
            
            # Try both quaternion formats (xyzw and wxyz)
            # Assume xyzw format (qx, qy, qz, qw)
            quat_xyzw = np.array(data[3:7], dtype=float)
            
            # Normalize quaternion
            quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
            
            self.last_pose = (position, quat_xyzw)
            self._compute_and_publish_ik()
            
        except Exception as e:
            print(f"Error in pose callback: {e}")
    
    def _compute_and_publish_ik(self):
        """Compute inverse kinematics and publish joint angles"""
        if self.last_pose is None:
            return
        
        self.processing = True
        start_time = time.time()
        
        try:
            position, quat_xyzw = self.last_pose
            
            # Convert quaternion to rotation matrix
            # scipy uses xyzw format (qx, qy, qz, qw)
            rotation = R.from_quat(quat_xyzw)
            target_orientation = rotation.as_matrix()
            
            if self.use_placo:
                # Use placo for IK
                # Create target transformation matrix
                T_target = np.eye(4)
                T_target[:3, :3] = target_orientation
                T_target[:3, 3] = position
                
                # Set initial joint positions if available (warm starting)
                if self.last_joint_angles is not None:
                    for i, name in enumerate(self.joint_names):
                        if i < len(self.last_joint_angles):
                            self.robot.set_joint(name, self.last_joint_angles[i])
                
                # Get end-effector frame name (typically the last link)
                # For placo, we need to find the appropriate frame
                frame_name = self.robot.get_frames()[-1] if self.robot.get_frames() else "tool"
                
                # Create IK task
                task = self.solver.add_frame_task(frame_name, T_target)
                task.configure(frame_name, "soft", 1.0, 1.0)
                
                # Solve IK
                self.solver.solve(True)  # warm_start=True for faster convergence
                
                # Get joint angles
                joint_angles = []
                for name in self.joint_names:
                    joint_angles.append(self.robot.get_joint(name))
                
                # Store for next iteration
                self.last_joint_angles = joint_angles
                
                # Publish joint angles
                for i, angle in enumerate(joint_angles):
                    if i < len(self.joint_publishers):
                        self.joint_publishers[i].publish(float(angle))
                
            else:
                # Use ikpy as fallback
                # Compute inverse kinematics
                # Use previous joint angles as initial guess for faster convergence (warm starting)
                initial_position = self.last_joint_angles
                
                joint_angles = self.chain.inverse_kinematics(
                    target_position=position,
                    target_orientation=target_orientation,
                    orientation_mode="all",  # Match both position and orientation
                    initial_position=initial_position
                )
                
                # Store for next iteration (warm starting for better performance)
                self.last_joint_angles = joint_angles
                
                # Extract only actuated joint angles based on active_links_mask
                actuated_joint_angles = []
                for i, (link, is_active) in enumerate(zip(self.chain.links, self.active_links_mask)):
                    if is_active and i < len(joint_angles):
                        actuated_joint_angles.append(joint_angles[i])
                
                # Publish joint angles
                for i, angle in enumerate(actuated_joint_angles):
                    if i < len(self.joint_publishers):
                        self.joint_publishers[i].publish(float(angle))
            
            # Performance logging
            elapsed = time.time() - start_time
            if elapsed > 0.01:  # Log if slower than 10ms
                print(f"IK computation time: {elapsed*1000:.2f}ms")
                
        except Exception as e:
            print(f"Error computing IK: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False
    
    def start(self):
        """Start the IK node"""
        print("Starting IK Node...")
        self.subscriber.start()
        
    def stop(self):
        """Stop the IK node"""
        print("Stopping IK Node...")
        self.subscriber.close()
        for pub in self.joint_publishers:
            pub.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inverse Kinematics Node - Computes joint angles from end-effector pose"
    )
    
    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to URDF file"
    )
    
    parser.add_argument(
        "--subscribe-topic",
        type=str,
        default="/hand/left",
        help="Topic to subscribe for end-effector pose (default: /hand/left)"
    )
    
    parser.add_argument(
        "--publish-prefix",
        type=str,
        default="joint",
        help="Prefix for joint topic names (default: joint)"
    )
    
    parser.add_argument(
        "--publish-suffix",
        type=str,
        default="",
        help="Suffix for joint topic names (default: empty)"
    )
    
    parser.add_argument(
        "--base-orientation",
        type=float,
        nargs=9,
        metavar=("r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"),
        help="Base link rotation matrix as 9 elements (row-major order)"
    )
    
    args = parser.parse_args()
    
    # Validate URDF path
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        sys.exit(1)
    
    print("=== HOS Inverse Kinematics Node ===")
    print("NOTE: Make sure the broker is running!")
    print("Start broker with: python start_broker.py")
    print("Or: python -m hos_core")
    print("====================================\n")
    
    # Create and start IK node
    ik_node = IKNode(
        urdf_path=str(urdf_path),
        subscribe_topic=args.subscribe_topic,
        publish_prefix=args.publish_prefix,
        publish_suffix=args.publish_suffix,
        base_orientation=args.base_orientation
    )
    
    try:
        ik_node.start()
        
        print("\nIK Node running... (Press Ctrl+C to stop)\n")
        
        # Main loop
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ik_node.stop()
        print("IK Node stopped.")


if __name__ == "__main__":
    main()
