import subprocess
import time
import signal
import sys
import threading
from typing import List, Tuple


class Node:
    """起動するノードの情報を保持するデータ構造"""

    def __init__(self, name: str, command: List[str]):
        self.name = name
        self.command = command
        self.process: subprocess.Popen | None = None


class Launcher:
    """ROS launch風に複数の子プロセス（ZeroMQノード）を管理・起動するクラス"""

    def __init__(self):
        self._nodes: List[Node] = []
        self._processes: List[Tuple[str, subprocess.Popen]] = []
        self._shutdown_flag = threading.Event()

        # Ctrl+C (SIGINT) のシグナルハンドラを設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def add_node(self, name: str, command: List[str]):
        """起動するノード（プロセス）を追加する"""
        assert len(command) > 0, "Command must not be empty"
        if command[0].endswith(".py"):
            # Pythonスクリプトの場合、実行可能なPythonインタプリタをコマンドに追加
            command = [sys.executable] + command

        node = Node(name, command)
        self._nodes.append(node)
        print(f"Node added: {name} (Command: {' '.join(command)})")

    def _signal_handler(self, signum, frame):
        """SIGINT/SIGTERMを受け取ったときの処理"""
        print(f"\nReceived signal {signum}. Shutting down all nodes...")
        self._shutdown_flag.set()

    def _output_reader(self, name: str, stream):
        """子プロセスからの出力をリアルタイムで読み込み、メインのStdoutに出力するスレッド関数"""
        while not self._shutdown_flag.is_set():
            try:
                # 1行読み込み (ブロッキング)
                line = stream.readline()
                if not line:
                    break
                # 出力ストリームにプレフィックスを付けて出力
                sys.stdout.write(f"[{name}] {line.decode().strip()}\n")
                sys.stdout.flush()
            except Exception as e:
                # ストリームが閉じられた場合などに終了
                if not self._shutdown_flag.is_set():
                    print(f"[{name}] Output reader error: {e}")
                break

    def _start_nodes(self):
        """登録された全てのノードを起動する"""
        print("\n--- Starting ZeroMQ Nodes ---")
        for node in self._nodes:
            print(f"Launching node: {node.name}")
            try:
                # プロセスを起動し、stdoutとstderrをパイプにリダイレクト
                process = subprocess.Popen(
                    node.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # Pythonノードがバッファリングを最小限にするように設定
                    env={"PYTHONUNBUFFERED": "1", **os.environ},
                )
                node.process = process
                self._processes.append((node.name, process))

                # stdout/stderrを読み込むためのスレッドを起動
                threading.Thread(
                    target=self._output_reader,
                    args=(node.name, process.stdout),
                    daemon=True,
                ).start()
                threading.Thread(
                    target=self._output_reader,
                    args=(node.name + "_ERR", process.stderr),
                    daemon=True,
                ).start()

                # ZeroMQのSlow Joiner問題対策として少し待機
                time.sleep(0.5)
            except FileNotFoundError:
                print(
                    f"ERROR: Command not found for node {node.name}. Check if {' '.join(node.command)} is correct."
                )
                self._shutdown_flag.set()
                break
            except Exception as e:
                print(f"ERROR: Failed to launch node {node.name}: {e}")
                self._shutdown_flag.set()
                break
        print("-----------------------------\n")

    def _shutdown_nodes(self):
        """全ての子プロセスをGracefulに終了させる"""
        print("\n--- Shutting down ZeroMQ Nodes ---")

        # 1. SIGINTを送る（ノードに自発的な終了を促す）
        for name, process in self._processes:
            if process.poll() is None:
                print(f"Terminating node: {name} (PID: {process.pid})")
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass  # 既に終了している場合は無視

        # 2. 終了を待機
        wait_timeout = 3
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            all_terminated = True
            for name, process in self._processes:
                if process.poll() is None:
                    all_terminated = False
            if all_terminated:
                break
            time.sleep(0.1)

        # 3. 強制終了 (まだ実行中のノードがある場合)
        for name, process in self._processes:
            if process.poll() is None:
                print(f"Node {name} did not terminate gracefully, killing it.")
                process.kill()
            else:
                print(f"Node {name} terminated (Return Code: {process.returncode})")

        print("----------------------------------")

    def spin(self):
        """ノードを起動し、Ctrl+Cまたは予期せぬノード終了まで実行を続ける"""

        if not self._nodes:
            print("No nodes registered. Use add_node() first.")
            return

        self._start_nodes()
        if self._shutdown_flag.is_set():
            # 起動中にエラーが発生した場合
            self._shutdown_nodes()
            return

        print("All nodes launched. **Running... Press Ctrl+C to stop.**")

        try:
            while not self._shutdown_flag.is_set():
                # 実行中のノードが予期せず終了していないかチェック
                for name, process in self._processes:
                    if process.poll() is not None:
                        # ノードが終了した場合
                        if not self._shutdown_flag.is_set():
                            print(
                                f"\n--- Node **{name}** terminated unexpectedly (Return Code: {process.returncode}). ---"
                            )
                            self._shutdown_flag.set()  # 全体シャットダウンフラグを立てる
                        break

                if self._shutdown_flag.is_set():
                    break

                time.sleep(0.1)

        finally:
            self._shutdown_nodes()
            print("Launcher finished.")


# --- 使用例 ---
if __name__ == "__main__":
    import os

    launcher = Launcher()

    # ノードを追加
    # uvicorn/uv run を使う場合は、以下のようにコマンドを変更します。
    # launcher.add_node("publisher", ["uv", "run", "pub_node.py"])
    # launcher.add_node("subscriber", ["uv", "run", "sub_node.py"])

    launcher.add_node("pub", ["pub_node.py"])
    launcher.add_node("sub", ["sub_node.py"])

    # 起動と実行
    launcher.spin()
