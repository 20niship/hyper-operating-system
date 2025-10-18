#!/usr/bin/env python3
"""
pub_node.py - テスト用パブリッシャーノード
/chatterトピックに定期的にメッセージを送信
"""

import time
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hos_core.topic import Publisher, TopicManager


def main():
    # レジストリサーバーを起動
    manager = TopicManager()
    manager.start_registry_server()
    time.sleep(0.5)  # 起動待機
    
    # パブリッシャーを作成
    publisher = Publisher("/chatter", "std_msgs/String")
    
    print("Publishing to /chatter topic... (Press Ctrl+C to stop)")
    
    try:
        count = 0
        while True:
            message = {
                "data": f"Hello, World! (message #{count})",
                "seq": count,
                "timestamp": time.time()
            }
            
            publisher.publish(message)
            print(f"Published: {message}")
            
            count += 1
            time.sleep(1.0)  # 1Hz
            
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        publisher.close()
        manager.stop()
        print("Publisher stopped.")


if __name__ == "__main__":
    main()