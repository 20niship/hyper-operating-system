# Inverse Kinematics Node Usage

## Overview

The `ik_node.py` script provides real-time inverse kinematics computation for robotic manipulators. It subscribes to end-effector pose topics, computes joint angles using the ikpy library, and publishes the results to individual joint topics.

## Features

- Real-time IK computation with warm starting for improved performance
- Configurable input/output topics
- Support for custom URDF files
- Configurable base link orientation
- Automatic handling of fixed joints
- High-frequency operation suitable for real-time control

## Prerequisites

1. Start the HOS broker:
   ```bash
   python start_broker.py
   ```
   or
   ```bash
   python -m hos_core
   ```

2. Ensure required dependencies are installed:
   ```bash
   pip install ikpy numpy scipy pyzmq
   ```

## Basic Usage

### Minimal Example

```bash
python ik_node.py --urdf hos_envs/multi_so101/SO101/so101_new_calib.urdf
```

This will:
- Subscribe to `/hand/left` for end-effector poses
- Publish joint angles to `/joint1`, `/joint2`, `/joint3`, etc.

### Advanced Usage

#### Custom Topics

```bash
python ik_node.py \
  --urdf path/to/robot.urdf \
  --subscribe-topic /custom/end_effector/pose \
  --publish-prefix arm_joint \
  --publish-suffix _left
```

This will:
- Subscribe to `/custom/end_effector/pose`
- Publish to `/arm_joint1_left`, `/arm_joint2_left`, etc.

#### With Base Orientation

```bash
python ik_node.py \
  --urdf path/to/robot.urdf \
  --base-orientation 1 0 0 0 0 1 0 -1 0
```

The base orientation is specified as a 3x3 rotation matrix in row-major order:
```
[r11, r12, r13, r21, r22, r23, r31, r32, r33]
```

## Message Format

### Input (End-Effector Pose)

The node subscribes to a topic expecting a list/array with 7 elements:

```python
[x, y, z, qx, qy, qz, qw]
```

Where:
- `x, y, z`: Position in meters
- `qx, qy, qz, qw`: Quaternion orientation (xyzw format)

Example:
```python
from hos_core.topic import Publisher
import numpy as np
from scipy.spatial.transform import Rotation as R

pub = Publisher("/hand/left", list)

# Position: 30cm forward, 30cm up
position = [0.3, 0.0, 0.3]

# Orientation: identity (no rotation)
quaternion = R.from_euler('xyz', [0, 0, 0]).as_quat()  # returns [qx, qy, qz, qw]

# Combine and publish
pose = position + quaternion.tolist()
pub.publish(pose)
```

### Output (Joint Angles)

The node publishes individual joint angles as float values to separate topics:
- `/joint1`: First actuated joint angle (radians)
- `/joint2`: Second actuated joint angle (radians)
- etc.

## Performance Optimization

The IK node includes several optimizations for real-time performance:

1. **Warm Starting**: Uses previous solution as initial guess for faster convergence
2. **Active Links Filtering**: Automatically excludes fixed joints from computation
3. **Efficient Publishing**: Direct float values instead of complex message structures
4. **Lock-free Operation**: Minimal thread contention

Typical computation time: 1-5ms per IK solution on modern hardware.

## Testing

A test script is provided to verify the IK node functionality:

```bash
# Terminal 1: Start broker
python start_broker.py

# Terminal 2: Start IK node
python ik_node.py --urdf hos_envs/multi_so101/SO101/so101_new_calib.urdf

# Terminal 3: Run test
python test_ik_node.py
```

The test script will publish various end-effector poses and display the computed joint angles.

## Troubleshooting

### "Timeout connecting to broker"
- Ensure the broker is running: `python start_broker.py`
- Check that the broker is on the correct host/port (default: localhost:5555)

### "URDF file not found"
- Verify the URDF path is correct and the file exists
- Use absolute paths or paths relative to the working directory

### "Invalid pose message format"
- Ensure the input message is a list with 7 elements: `[x, y, z, qx, qy, qz, qw]`
- Check that quaternion values are valid (should be normalized)

### Slow IK computation
- Check CPU usage and system load
- Ensure warm starting is working (should see < 10ms computation time after first iteration)
- Consider reducing the frequency of pose updates if real-time performance is critical

## Integration with Existing Code

The IK node integrates seamlessly with existing HOS teleop code. For example:

```python
from hos_teleop.mocap.hand_tracking import Hand3DTracker
from hos_core.topic import Publisher

# Track hand position
tracker = Hand3DTracker(cap_idx1=0, cap_idx2=1)
pose_pub = Publisher("/hand/left", list)

while True:
    tracker.update()
    t_l = tracker._l_hand_3d_
    if t_l:
        # Publish hand pose (position + quaternion)
        pose = [
            t_l.pos[0], t_l.pos[1], t_l.pos[2],
            t_l.rot[0], t_l.rot[1], t_l.rot[2], t_l.rot[3]
        ]
        pose_pub.publish(pose)
```

Then the IK node will automatically compute and publish joint angles.

## Command-Line Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--urdf` | Yes | - | Path to URDF file |
| `--subscribe-topic` | No | `/hand/left` | Topic to subscribe for end-effector pose |
| `--publish-prefix` | No | `joint` | Prefix for joint topic names |
| `--publish-suffix` | No | `` (empty) | Suffix for joint topic names |
| `--base-orientation` | No | None | Base link rotation matrix (9 floats) |

## Examples

### Example 1: Dual Arm Control

```bash
# Left arm
python ik_node.py \
  --urdf robot.urdf \
  --subscribe-topic /hand/left \
  --publish-suffix _left

# Right arm
python ik_node.py \
  --urdf robot.urdf \
  --subscribe-topic /hand/right \
  --publish-suffix _right
```

### Example 2: Custom Joint Names

```bash
python ik_node.py \
  --urdf robot.urdf \
  --publish-prefix shoulder
```

Output topics: `/shoulder1`, `/shoulder2`, ...
