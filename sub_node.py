#!/usr/bin/env python3
"""
sub_node.py - テスト用サブスクライバーノード
/chatterトピックからメッセージを受信して表示
"""

import time
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hos_core.topic import Subscriber


def message_callback(message):
    """メッセージ受信時のコールバック関数"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Received: {message}")


def main():
    print("Subscribing to /chatter topic... (Press Ctrl+C to stop)")
    
    # サブスクライバーを作成
    subscriber = Subscriber("/chatter", message_callback, "std_msgs/String")
    
    try:
        subscriber.start()
        
        # メインループ
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping subscriber...")
    finally:
        subscriber.close()
        print("Subscriber stopped.")


if __name__ == "__main__":
    main()