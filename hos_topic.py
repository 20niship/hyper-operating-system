#!/usr/bin/env python3
"""
hos_topic.py - HOSトピックコマンドラインツール
ROSのrostopicコマンドに相当する機能を提供

使用例:
    uv run hos_topic.py echo /chatter
    uv run hos_topic.py list 
    uv run hos_topic.py info /chatter
"""

import argparse
import sys
import time
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from hos_core.topic import TopicClient, TopicManager
except ImportError as e:
    print(f"Error importing HOS modules: {e}")
    print("Please install dependencies: uv add pyzmq")
    sys.exit(1)


def start_registry_if_needed():
    """必要に応じてレジストリサーバーを起動"""
    client = TopicClient()
    
    try:
        # レジストリサーバーが動作中かチェック
        client.list_topics()
        client.close()
        return None  # 既に動作中
    except Exception:
        # レジストリサーバーが動作していない場合、起動
        print("Starting HOS registry server...")
        manager = TopicManager()
        manager.start_registry_server()
        time.sleep(0.5)  # 起動待機
        return manager


def cmd_echo(args):
    """トピックのメッセージをエコー表示"""
    topic_name = args.topic
    count = args.count
    
    registry_manager = start_registry_if_needed()
    
    client = TopicClient()
    
    try:
        client.echo_topic(topic_name, count)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.close()
        if registry_manager:
            registry_manager.stop()


def cmd_list(args):
    """トピック一覧を表示"""
    registry_manager = start_registry_if_needed()
    
    client = TopicClient()
    
    try:
        topics = client.list_topics()
        
        if not topics:
            print("No topics found.")
            return
            
        print("Published topics:")
        print(f"{'Topic Name':<30} {'Type':<20} {'Publishers':<12} {'Subscribers':<12}")
        print("-" * 74)
        
        for topic in topics:
            print(f"{topic.name:<30} {topic.message_type:<20} {topic.publisher_count:<12} {topic.subscriber_count:<12}")
            
    except Exception as e:
        print(f"Error listing topics: {e}")
    finally:
        client.close()
        if registry_manager:
            registry_manager.stop()


def cmd_info(args):
    """トピック情報を表示"""
    topic_name = args.topic
    
    registry_manager = start_registry_if_needed()
    
    client = TopicClient()
    
    try:
        topic_info = client.get_topic_info(topic_name)
        
        if not topic_info:
            print(f"Topic '{topic_name}' not found.")
            return
            
        print(f"Topic: {topic_info.name}")
        print(f"Type: {topic_info.message_type}")
        print(f"Publishers: {topic_info.publisher_count}")
        print(f"Subscribers: {topic_info.subscriber_count}")
        
        if topic_info.last_message_time:
            last_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                    time.localtime(topic_info.last_message_time))
            print(f"Last message time: {last_time}")
        else:
            print("Last message time: No messages yet")
            
    except Exception as e:
        print(f"Error getting topic info: {e}")
    finally:
        client.close()
        if registry_manager:
            registry_manager.stop()


def cmd_pub(args):
    """メッセージをパブリッシュ（テスト用）"""
    from hos_core.topic import Publisher
    import json
    
    topic_name = args.topic
    message = args.message
    rate = args.rate
    count = args.count
    
    registry_manager = start_registry_if_needed()
    
    publisher = None
    try:
        publisher = Publisher(topic_name, str)
        
        print(f"Publishing to '{topic_name}' at {rate} Hz...")
        
        message_count = 0
        
        while True:
            if count is not None and message_count >= count:
                break
                
            # メッセージを送信
            timestamp = time.time()
            if message.startswith('{') and message.endswith('}'):
                # JSONメッセージとして解析
                try:
                    msg_data = json.loads(message)
                    msg_data['seq'] = message_count
                    msg_data['timestamp'] = timestamp
                except json.JSONDecodeError:
                    msg_data = {
                        "data": message,
                        "seq": message_count,
                        "timestamp": timestamp
                    }
            else:
                msg_data = {
                    "data": message,
                    "seq": message_count,
                    "timestamp": timestamp
                }
            
            publisher.publish(msg_data)
            message_count += 1
            
            print(f"Published message {message_count}: {msg_data}")
            
            # レート制御
            if rate > 0:
                time.sleep(1.0 / rate)
                
    except KeyboardInterrupt:
        print("\nStopped publishing.")
    except Exception as e:
        print(f"Error publishing: {e}")
    finally:
        if publisher:
            publisher.close()
        if registry_manager:
            registry_manager.stop()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="HOS Topic Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s echo /chatter          # /chatterトピックをエコー表示
  %(prog)s list                   # トピック一覧表示
  %(prog)s info /chatter          # /chatterトピックの情報表示
  %(prog)s pub /chatter "Hello"   # /chatterトピックにメッセージをパブリッシュ
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # echo コマンド
    echo_parser = subparsers.add_parser('echo', help='Display messages from a topic')
    echo_parser.add_argument('topic', help='Topic name to echo')
    echo_parser.add_argument('-n', '--count', type=int, 
                           help='Number of messages to display (default: unlimited)')
    echo_parser.set_defaults(func=cmd_echo)
    
    # list コマンド
    list_parser = subparsers.add_parser('list', help='List all topics')
    list_parser.set_defaults(func=cmd_list)
    
    # info コマンド
    info_parser = subparsers.add_parser('info', help='Display topic information')
    info_parser.add_argument('topic', help='Topic name to get info')
    info_parser.set_defaults(func=cmd_info)
    
    # pub コマンド（テスト用）
    pub_parser = subparsers.add_parser('pub', help='Publish messages to a topic')
    pub_parser.add_argument('topic', help='Topic name to publish')
    pub_parser.add_argument('message', help='Message to publish')
    pub_parser.add_argument('-r', '--rate', type=float, default=1.0,
                          help='Publishing rate in Hz (default: 1.0)')
    pub_parser.add_argument('-n', '--count', type=int,
                          help='Number of messages to publish (default: unlimited)')
    pub_parser.set_defaults(func=cmd_pub)
    
    # 引数解析
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # コマンド実行
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
