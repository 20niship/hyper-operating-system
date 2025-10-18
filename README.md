# HOS Hyper Operating System 零式

これはROSの代替としてZeroMQを用いた分散ロボット制御フレームワークです。

また、テレオペで模倣学習のデータセットを用意したり、シミュレーション上で複数のロボットを動かしてデータセットを作成することに特化しています

## hos core

ROSのようなコマンドを使用可能です

```bash
uv run hos_topic.py echo /chatter
uv run hos_topic.py list 
uv run hos_topic.py info /chatter


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




## ステレオカメラのキャリブレーション


## 参考
- https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
