#!/usr/bin/env python3
"""
hos_bag.py - HOSバッグファイル記録・再生ツール
ROSのrosbagに相当する機能を提供

使用例:
    uv run hos_bag.py record /chatter
    uv run hos_bag.py record /chatter /cmd_vel -o mybag.bag
    uv run hos_bag.py play sample.bag
    uv run hos_bag.py info sample.bag
    uv run hos_bag.py topics sample.bag
"""

import argparse
import sys
import time
import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from hos_core.topic import Publisher, Subscriber, TopicClient
except ImportError as e:
    print(f"Error importing HOS modules: {e}")
    print("Please install dependencies: uv add pyzmq")
    sys.exit(1)


class BagMessage:
    """バッグファイルに記録されるメッセージのデータクラス"""
    
    def __init__(self, topic: str, message: Any, timestamp: float, message_type: str = "std_msgs/String"):
        self.topic = topic
        self.message = message
        self.timestamp = timestamp
        self.message_type = message_type
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'message': self.message,
            'timestamp': self.timestamp,
            'message_type': self.message_type
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BagMessage':
        return cls(
            topic=data['topic'],
            message=data['message'],
            timestamp=data['timestamp'],
            message_type=data.get('message_type', 'std_msgs/String')
        )


class BagRecorder:
    """バッグファイル記録クラス"""
    
    def __init__(self, topics: List[str], output_file: Optional[str] = None):
        self.topics = topics
        self.output_file = output_file or f"hos_bag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bag"
        self.messages: List[BagMessage] = []
        self.subscribers: List[Subscriber] = []
        self.recording = False
        self._lock = threading.Lock()
        
        print(f"Recording to: {self.output_file}")
        print(f"Topics to record: {', '.join(topics)}")
        
    def start_recording(self):
        """記録を開始"""
        if self.recording:
            print("Already recording!")
            return
            
        self.recording = True
        self.messages.clear()
        
        # 各トピックのサブスクライバーを作成
        for topic in self.topics:
            def make_callback(topic_name):
                def callback(message):
                    self._record_message(topic_name, message)
                return callback
                
            subscriber = Subscriber(topic, make_callback(topic))
            subscriber.start()
            self.subscribers.append(subscriber)
            
        print("Recording started. Press Ctrl+C to stop and save.")
        
    def _record_message(self, topic: str, message: Any):
        """メッセージを記録"""
        if not self.recording:
            return
            
        timestamp = time.time()
        
        # メッセージをBagMessageとして記録
        bag_msg = BagMessage(topic, message, timestamp)
        
        with self._lock:
            self.messages.append(bag_msg)
            
        # リアルタイム表示
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]
        print(f"[{time_str}] {topic}: {json.dumps(message) if isinstance(message, dict) else str(message)}")
        
    def stop_recording(self):
        """記録を停止してファイルに保存"""
        if not self.recording:
            return
            
        self.recording = False
        
        # サブスクライバーを停止
        for subscriber in self.subscribers:
            subscriber.close()
        self.subscribers.clear()
        
        # ファイルに保存
        self._save_to_file()
        
        print(f"\nRecording stopped. Saved {len(self.messages)} messages to {self.output_file}")
        
    def _save_to_file(self):
        """メッセージをファイルに保存"""
        if not self.messages:
            print("No messages to save.")
            return
            
        # バッグファイル形式で保存
        bag_data = {
            'version': '1.0',
            'created': time.time(),
            'topics': self.topics,
            'message_count': len(self.messages),
            'start_time': self.messages[0].timestamp if self.messages else None,
            'end_time': self.messages[-1].timestamp if self.messages else None,
            'duration': (self.messages[-1].timestamp - self.messages[0].timestamp) if len(self.messages) > 1 else 0,
            'messages': [msg.to_dict() for msg in self.messages]
        }
        
        # gzip圧縮して保存
        with gzip.open(self.output_file, 'wt', encoding='utf-8') as f:
            json.dump(bag_data, f, indent=2, ensure_ascii=False)


class BagPlayer:
    """バッグファイル再生クラス"""
    
    def __init__(self, bag_file: str):
        self.bag_file = bag_file
        self.bag_data: Optional[Dict[str, Any]] = None
        self.publishers: Dict[str, Publisher] = {}
        
    def load_bag(self) -> bool:
        """バッグファイルを読み込み"""
        try:
            if self.bag_file.endswith('.bag'):
                # gzip形式
                with gzip.open(self.bag_file, 'rt', encoding='utf-8') as f:
                    self.bag_data = json.load(f)
            else:
                # プレーンJSON形式
                with open(self.bag_file, 'r', encoding='utf-8') as f:
                    self.bag_data = json.load(f)
                    
            return True
            
        except Exception as e:
            print(f"Error loading bag file: {e}")
            return False
            
    def get_info(self) -> Dict[str, Any]:
        """バッグファイルの情報を取得"""
        if not self.bag_data:
            return {}
            
        info = {
            'file': self.bag_file,
            'version': self.bag_data.get('version', 'unknown'),
            'duration': self.bag_data.get('duration', 0),
            'start_time': self.bag_data.get('start_time'),
            'end_time': self.bag_data.get('end_time'),
            'message_count': self.bag_data.get('message_count', 0),
            'topics': self.bag_data.get('topics', [])
        }
        
        # トピック別メッセージ数を計算
        topic_counts = {}
        messages = self.bag_data.get('messages', [])
        for msg_data in messages:
            topic = msg_data.get('topic')
            if topic:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
        info['topic_counts'] = topic_counts
        return info
        
    def play(self, rate: float = 1.0, loop: bool = False, start_time: Optional[float] = None, duration: Optional[float] = None):
        """バッグファイルを再生"""
        if not self.bag_data:
            print("Bag file not loaded!")
            return
            
        messages = self.bag_data.get('messages', [])
        if not messages:
            print("No messages to play!")
            return
            
        # メッセージをタイムスタンプでソート
        messages.sort(key=lambda x: x.get('timestamp', 0))
        
        # 開始時間と終了時間を決定
        first_timestamp = messages[0]['timestamp']
        
        if start_time is not None:
            start_timestamp = first_timestamp + start_time
            messages = [msg for msg in messages if msg['timestamp'] >= start_timestamp]
            
        if duration is not None and messages:
            end_timestamp = messages[0]['timestamp'] + duration
            messages = [msg for msg in messages if msg['timestamp'] <= end_timestamp]
            
        if not messages:
            print("No messages in specified time range!")
            return
            
        # トピック別パブリッシャーを作成
        topics = set(msg['topic'] for msg in messages)
        for topic in topics:
            self.publishers[topic] = Publisher(topic)
            
        print(f"Playing {len(messages)} messages from {len(topics)} topics...")
        print(f"Rate: {rate}x, Loop: {loop}")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                self._play_messages(messages, rate)
                
                if not loop:
                    break
                    
                print("\nLooping...")
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopped playback.")
        finally:
            # パブリッシャーを閉じる
            for publisher in self.publishers.values():
                publisher.close()
            self.publishers.clear()
            
    def _play_messages(self, messages: List[Dict[str, Any]], rate: float):
        """メッセージを順次再生"""
        if not messages:
            return
            
        start_playback_time = time.time()
        first_msg_timestamp = messages[0]['timestamp']
        
        for i, msg_data in enumerate(messages):
            # 元のタイムスタンプとの差分を計算
            original_elapsed = msg_data['timestamp'] - first_msg_timestamp
            
            # レート調整された再生時間
            target_playback_time = start_playback_time + (original_elapsed / rate)
            
            # 必要に応じて待機
            current_time = time.time()
            if target_playback_time > current_time:
                time.sleep(target_playback_time - current_time)
                
            # メッセージを送信
            topic = msg_data['topic']
            message = msg_data['message']
            
            if topic in self.publishers:
                self.publishers[topic].publish(message)
                
                # 進捗表示
                timestamp_str = datetime.fromtimestamp(msg_data['timestamp']).strftime('%H:%M:%S.%f')[:-3]
                print(f"[{timestamp_str}] {topic}: {json.dumps(message) if isinstance(message, dict) else str(message)}")


def cmd_record(args):
    """記録コマンド"""
    topics = args.topics
    output_file = args.output
    
    # すべてのトピックを記録する場合
    if args.all:
        client = TopicClient()
        try:
            all_topics = client.list_topics()
            topics = [t.name for t in all_topics if t.publisher_count > 0]
            print(f"Recording all active topics: {topics}")
        except Exception as e:
            print(f"Error getting topic list: {e}")
            return 1
        finally:
            client.close()
            
    if not topics:
        print("No topics specified and no active topics found!")
        return 1
        
    recorder = BagRecorder(topics, output_file)
    
    try:
        recorder.start_recording()
        
        # 記録継続
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop_recording()
        
    return 0


def cmd_play(args):
    """再生コマンド"""
    bag_file = args.bag_file
    rate = args.rate
    loop = args.loop
    start_time = args.start
    duration = args.duration
    
    player = BagPlayer(bag_file)
    
    if not player.load_bag():
        return 1
        
    try:
        player.play(rate=rate, loop=loop, start_time=start_time, duration=duration)
    except Exception as e:
        print(f"Error during playback: {e}")
        return 1
        
    return 0


def cmd_info(args):
    """情報表示コマンド"""
    bag_file = args.bag_file
    
    player = BagPlayer(bag_file)
    
    if not player.load_bag():
        return 1
        
    info = player.get_info()
    
    print(f"Bag file: {info['file']}")
    print(f"Version: {info['version']}")
    print(f"Duration: {info['duration']:.3f}s")
    print(f"Messages: {info['message_count']}")
    
    if info['start_time']:
        start_str = datetime.fromtimestamp(info['start_time']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_str = datetime.fromtimestamp(info['end_time']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"Start time: {start_str}")
        print(f"End time: {end_str}")
        
    print(f"Topics: {len(info['topics'])}")
    for topic in info['topics']:
        count = info['topic_counts'].get(topic, 0)
        print(f"  {topic}: {count} messages")
        
    return 0


def cmd_topics(args):
    """トピック一覧表示コマンド"""
    bag_file = args.bag_file
    
    player = BagPlayer(bag_file)
    
    if not player.load_bag():
        return 1
        
    info = player.get_info()
    
    print(f"Topics in {info['file']}:")
    print(f"{'Topic':<30} {'Messages':<12} {'Type':<20}")
    print("-" * 62)
    
    # メッセージからトピック情報を抽出
    topic_info = {}
    if player.bag_data:
        messages = player.bag_data.get('messages', [])
    else:
        messages = []
    for msg_data in messages:
        topic = msg_data.get('topic')
        if topic:
            if topic not in topic_info:
                topic_info[topic] = {
                    'count': 0,
                    'type': msg_data.get('message_type', 'unknown')
                }
            topic_info[topic]['count'] += 1
            
    for topic, data in sorted(topic_info.items()):
        print(f"{topic:<30} {data['count']:<12} {data['type']:<20}")
        
    return 0


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="HOS Bag File Recorder and Player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s record /chatter                    # /chatterトピックを記録
  %(prog)s record /chatter /cmd_vel          # 複数のトピックを記録
  %(prog)s record /chatter -o my_data.bag    # 出力ファイル名を指定
  %(prog)s record --all                      # すべてのアクティブなトピックを記録
  
  %(prog)s play sample.bag                   # バッグファイルを再生
  %(prog)s play sample.bag --rate 2.0        # 2倍速で再生
  %(prog)s play sample.bag --loop            # ループ再生
  %(prog)s play sample.bag --start 10        # 10秒後から再生
  %(prog)s play sample.bag --duration 30     # 30秒間だけ再生
  
  %(prog)s info sample.bag                   # バッグファイルの情報表示
  %(prog)s topics sample.bag                 # バッグファイル内のトピック一覧
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # record コマンド
    record_parser = subparsers.add_parser('record', help='Record topics to bag file')
    record_parser.add_argument('topics', nargs='*', help='Topics to record')
    record_parser.add_argument('-o', '--output', help='Output bag file name')
    record_parser.add_argument('--all', action='store_true', 
                             help='Record all active topics')
    record_parser.set_defaults(func=cmd_record)
    
    # play コマンド
    play_parser = subparsers.add_parser('play', help='Play back bag file')
    play_parser.add_argument('bag_file', help='Bag file to play')
    play_parser.add_argument('--rate', type=float, default=1.0,
                           help='Playback rate multiplier (default: 1.0)')
    play_parser.add_argument('--loop', action='store_true',
                           help='Loop playback')
    play_parser.add_argument('--start', type=float,
                           help='Start playback at this time offset (seconds)')
    play_parser.add_argument('--duration', type=float,
                           help='Play for this duration (seconds)')
    play_parser.set_defaults(func=cmd_play)
    
    # info コマンド
    info_parser = subparsers.add_parser('info', help='Display bag file information')
    info_parser.add_argument('bag_file', help='Bag file to analyze')
    info_parser.set_defaults(func=cmd_info)
    
    # topics コマンド
    topics_parser = subparsers.add_parser('topics', help='List topics in bag file')
    topics_parser.add_argument('bag_file', help='Bag file to analyze')
    topics_parser.set_defaults(func=cmd_topics)
    
    # 引数解析
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # コマンド実行
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
