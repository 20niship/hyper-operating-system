#!/usr/bin/env python3
"""
sub_node.py - 新ブローカー対応サブスクライバーノード  
/chatterトピックからメッセージを受信して表示
パブリッシャーが再起動しても継続受信可能
"""

import sys
import time
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
    print("=== HOS Subscriber Node ===")
    print("NOTE: Make sure the broker is running!")
    print("Start broker with: python start_broker.py")
    print("Or: python -m hos_core")
    print("=============================\n")
    
    print("Subscribing to /chatter topic... (Press Ctrl+C to stop)")
    print("Subscriber will automatically reconnect if broker restarts.\n")
    
    # サブスクライバーを作成（ブローカーに自動接続・自動再接続）
    subscriber = Subscriber("/chatter", message_callback, "std_msgs/String")
    
    try:
        subscriber.start()
        
        # メインループ - サブスクライバーは別スレッドで動作
        print("Subscriber is running in background...")
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping subscriber...")
    finally:
        subscriber.close()
        print("Subscriber stopped.")


if __name__ == "__main__":
    main()