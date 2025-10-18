#!/usr/bin/env python3
"""
pub_node.py - 新ブローカー対応パブリッシャーノード
/chatterトピックに定期的にメッセージを送信
独立したブローカーサービスに接続する方式
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hos_core.topic import Publisher


def main():
    print("=== HOS Publisher Node ===")
    print("NOTE: Make sure the broker is running!")
    print("Start broker with: python start_broker.py")
    print("Or: python -m hos_core")
    print("=============================\n")
    
    # パブリッシャーを作成（ブローカーに自動接続）
    publisher = Publisher("/chatter", "std_msgs/String")
    
    print("Publishing to /chatter topic... (Press Ctrl+C to stop)")
    print("Publisher will automatically reconnect if broker restarts.\n")
    
    try:
        count = 0
        failed_count = 0
        
        while True:
            message = {
                "data": f"Hello, World! (message #{count})",
                "seq": count,
                "timestamp": time.time()
            }
            
            # パブリッシュ（自動再接続付き）
            success = publisher.publish(message)
            
            if success:
                print(f"Published: {message}")
                failed_count = 0  # 成功時はカウンターリセット
            else:
                failed_count += 1
                print(f"Failed to publish (failed {failed_count} times)")
                
                # 連続失敗時は少し長めに待機
                if failed_count >= 5:
                    print("Multiple failures, waiting longer...")
                    time.sleep(5.0)
                    failed_count = 0
                    continue
            
            count += 1
            time.sleep(1.0)  # 1Hz
            
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        publisher.close()
        print("Publisher stopped.")


if __name__ == "__main__":
    main()