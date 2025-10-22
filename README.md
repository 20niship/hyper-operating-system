# HOS Hyper Operating System 零式

これはROSの代替としてZeroMQを用いた分散ロボット制御フレームワークです。

また、テレオペで模倣学習のデータセットを用意したり、シミュレーション上で複数のロボットを動かしてデータセットを作成することに特化しています

## Topicによるノード同士の通信システム

パブリッシャーの再起動に対応した堅牢なmany-to-many pub/subシステムです。

## 主な特徴

- **独立したブローカーサービス**: パブリッシャーとサブスクライバーから独立して動作
- **自動再接続**: パブリッシャーやブローカーが再起動しても自動的に再接続
- **Many-to-Many通信**: 複数のパブリッシャーと複数のサブスクライバーが同時に動作可能
- **ZeroMQ XPUB/XSUB**: 高性能なメッセージプロキシパターンを使用

## システム構成

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Publisher 1   │    │                 │    │  Subscriber 1   │
│                 │───▶│  Message Broker │◀───│                 │
└─────────────────┘    │   (XPUB/XSUB)   │    └─────────────────┘
                       │                 │    
┌─────────────────┐    │   Frontend:5556 │    ┌─────────────────┐
│   Publisher 2   │───▶│   Backend :5557 │◀───│  Subscriber 2   │
└─────────────────┘    │   Registry:5555 │    └─────────────────┘
```

トピックを使用するには、brokerを起動する必要があります
```bash
# 方法1: スクリプトから起動
python start_broker.py

# 方法2: モジュールとして起動  
python -m hos_core
```

ブローカーが正常に起動すると以下のような出力が表示されます：

```py
Message Broker initialized:
  Publisher port (XSUB): 5556
  Subscriber port (XPUB): 5557
  Registry port (REP): 5555

=== HOS Message Broker Running ===
Press Ctrl+C to stop
Publishers connect to: tcp://localhost:5556
Subscribers connect to: tcp://localhost:5557
Registry available at: tcp://localhost:5555
==========================================
```

そのあとは、ros(ros2ではない)のROSのようなコマンドを使用可能です

```bash
uv run hos_topic.py echo /chatter
uv run hos_topic.py list 
uv run hos_topic.py info /chatter

# rosbagのようにトピックの中を保存する
uv run hos_bag.py record /chatter
uv run hos_bag.py play sample.bag
```

また、HosLaunchクラスを使用することで、rosのlaunch.pyファイルのような実装を行い複数のノードを作成することが可能です

```py

import subprocess
import time
import signal
import sys
from typing import List

# 起動したいノードのリストを定義
# 形式: (ノード名, 実行コマンドと引数)
NODES_TO_LAUNCH = [
    ("publisher", ["python", "pub_node.py"]),
    ("subscriber", ["python", "sub_node.py"]),
    # "uv run xxxx.py" の形式を使いたい場合は、例えば以下のように記述
    # ("another_node", ["uv", "run", "another_node.py"]), 
]

def launch_nodes(nodes: List[tuple]) -> List[subprocess.Popen]:
    """
    指定されたノードを別々のプロセスとして起動する
    """
    processes = []
    print("--- Starting ZeroMQ Nodes ---")
    
    # ノードを順番に起動
    for name, command in nodes:
        print(f"Launching node: {name} (Command: {' '.join(command)})")
        try:
            # Popenで新しいプロセスを起動。stdoutとstderrは現在のプロセスにパイプする
            process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
            processes.append((name, process))
            # ROS launchのように、ノード起動の間に少し待機時間を設ける（ZeroMQの接続確立のため）
            time.sleep(0.5) 
        except FileNotFoundError:
            print(f"ERROR: Command not found for node {name}. Check if {' '.join(command)} is correct.")
        except Exception as e:
            print(f"ERROR: Failed to launch node {name}: {e}")
            
    print("-----------------------------")
    return processes

def shutdown_nodes(processes: List[tuple]):
    """
    起動された全てのプロセスを終了させる
    """
    print("\n--- Shutting down ZeroMQ Nodes ---")
    for name, process in processes:
        if process.poll() is None: # プロセスがまだ実行中か確認
            print(f"Terminating node: {name} (PID: {process.pid})")
            try:
                # SIGINT (Ctrl+Cに相当) を送って、Gracefulに終了を促す
                process.send_signal(signal.SIGINT)
                process.wait(timeout=2) # 2秒待って終了を確認
                if process.poll() is None:
                    # 終了しない場合は強制終了
                    print(f"Node {name} did not terminate, killing it.")
                    process.kill()
            except Exception as e:
                print(f"Error terminating node {name}: {e}")
        else:
            print(f"Node {name} already terminated (Return Code: {process.returncode})")
            
    print("----------------------------------")

def main():
    processes = []
    try:
        # ノードを起動
        processes = launch_nodes(NODES_TO_LAUNCH)

        # 実行中のノードを監視
        print("All nodes launched. Press Ctrl+C to shut down.")
        while True:
            # 実行中のプロセスが一つでも終了していないかチェック
            for name, process in processes:
                if process.poll() is not None: # 終了している場合
                    print(f"Node {name} has terminated unexpectedly (Return Code: {process.returncode}). Shutting down all nodes.")
                    return # メインループを抜けてシャットダウン処理へ
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        # Ctrl+Cが押されたら終了
        print("\nReceived Ctrl+C. Initiating shutdown.")
    finally:
        # 終了処理
        shutdown_nodes(processes)
        print("Launch finished.")

if __name__ == "__main__":
    main()
```

### 2. サブスクライバーを起動

別のターミナルでサブスクライバーを起動：

```bash
python sub_node.py
```

### 3. パブリッシャーを起動

さらに別のターミナルでパブリッシャーを起動：

```bash
python pub_node.py
```
### パブリッシャーの作成

```python
from hos_core.topic import Publisher

# パブリッシャーを作成（自動的にブローカーに接続）
publisher = Publisher("/my_topic", str)

# メッセージをパブリッシュ（自動再接続付き）
message = {"data": "Hello, World!", "timestamp": time.time()}
success = publisher.publish(message)

# 終了
publisher.close()
```

### サブスクライバーの作成

```python
from hos_core.topic import Subscriber

def message_callback(message):
    print(f"Received: {message}")

# サブスクライバーを作成（自動的にブローカーに接続）
subscriber = Subscriber("/my_topic", message_callback, str)

# 受信開始（バックグラウンドスレッドで動作）
subscriber.start()

# メインループ
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

# 終了
subscriber.close()
```

## ポート使用

- **5555**: レジストリ・管理用
- **5556**: パブリッシャー接続用（XSUB）
- **5557**: サブスクライバー接続用（XPUB）

これらのポートが他のプロセスで使用されていないことを確認してください。



## 逆運動学ノード（Inverse Kinematics Node）

エンドエフェクターの位置と姿勢から関節角度を計算するノードが実装されています。

**サポートするライブラリ:**
- **placo** (推奨): 高性能なIKソルバー、より高速で正確
- **ikpy** (フォールバック): placoが利用できない場合に自動的に使用

### インストール

```bash
# placoを使用する場合（推奨）
pip install placo

# ikpyのみの場合（既にインストール済み）
pip install ikpy
```

ノードは自動的に利用可能なライブラリを検出して使用します。

### 基本的な使用方法

```bash
# ブローカーを起動（別ターミナル）
python start_broker.py

# IKノードを起動
python ik_node.py --urdf hos_envs/multi_so101/SO101/so101_new_calib.urdf
```

デフォルトでは：
- `/hand/left` トピックからエンドエフェクターの位置・姿勢を受信
- `/joint1`, `/joint2`, `/joint3`, ... に関節角度を配信

### カスタマイズオプション

```bash
# 入力トピック名を変更
python ik_node.py --urdf robot.urdf --subscribe-topic /custom/pose

# 出力トピックのプレフィックスとサフィックスを指定
python ik_node.py --urdf robot.urdf --publish-prefix arm --publish-suffix _left
# → /arm1_left, /arm2_left, ... に配信

# ベースリンクの回転行列を指定（3x3行列を9要素で指定）
python ik_node.py --urdf robot.urdf --base-orientation 1 0 0 0 0 1 0 -1 0
```

### 既存のハンドトラッキングとの連携

```python
from hos_teleop.mocap.hand_tracking import Hand3DTracker
from hos_core.topic import Publisher

tracker = Hand3DTracker(cap_idx1=0, cap_idx2=1)
pose_pub = Publisher("/hand/left", str)

while True:
    tracker.update()
    t_l = tracker._l_hand_3d_
    if t_l:
        # 位置とクォータニオンを結合して配信
        pose = [
            t_l.pos[0], t_l.pos[1], t_l.pos[2],
            t_l.rot[0], t_l.rot[1], t_l.rot[2], t_l.rot[3]
        ]
        pose_pub.publish(str(pose))
```

この場合、IKノードが自動的に関節角度を計算して配信します。

詳細は [IKノード使用方法ドキュメント](docs/ik_node_usage.md) を参照してください。

## ステレオカメラのキャリブレーション


## 参考
- https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html


## LeRobot Coreとの連携

- https://note.com/npaka/n/nc5a938a1a598 を参考に
