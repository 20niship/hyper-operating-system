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



## ステレオカメラのキャリブレーション


## 参考
- https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html


## LeRobot Coreとの連携

- https://note.com/npaka/n/nc5a938a1a598 を参考に
