"""
HOS Message Broker - 独立したメッセージブローカーサービス
ZeroMQのXPUB/XSUBパターンを使用してmany-to-manyのメッセージ配信を実現
パブリッシャーが再起動してもサブスクライバーが継続して受信可能
"""

import zmq
import threading
import time
import signal
import sys
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TopicInfo:
    """トピック情報を格納するデータクラス"""
    name: str
    message_type: str
    publisher_count: int = 0
    subscriber_count: int = 0
    last_message_time: Optional[float] = None
    created_time: Optional[float] = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MessageBroker:
    """
    独立したメッセージブローカー
    XPUB/XSUBパターンでmany-to-many通信を実現
    """
    
    def __init__(self, 
                 frontend_port: int = 5556,  # パブリッシャー接続用
                 backend_port: int = 5557,   # サブスクライバー接続用
                 registry_port: int = 5555): # レジストリ・管理用
        
        self.context = zmq.Context()
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.registry_port = registry_port
        
        # ブローカーのフロントエンド（パブリッシャー接続）
        self.frontend = self.context.socket(zmq.XSUB)
        self.frontend.bind(f"tcp://*:{frontend_port}")
        
        # ブローカーのバックエンド（サブスクライバー接続）  
        self.backend = self.context.socket(zmq.XPUB)
        self.backend.bind(f"tcp://*:{backend_port}")
        
        # レジストリサーバー（管理・監視用）
        self.registry_socket = self.context.socket(zmq.REP)
        self.registry_socket.bind(f"tcp://*:{registry_port}")
        
        # トピック管理
        self.topics: Dict[str, TopicInfo] = {}
        self.active_publishers: Set[str] = set()
        self.active_subscribers: Dict[str, Set[str]] = {}  # topic -> subscriber_ids
        
        self._running = False
        self._broker_thread = None
        self._registry_thread = None
        
        print("Message Broker initialized:")
        print(f"  Publisher port (XSUB): {frontend_port}")
        print(f"  Subscriber port (XPUB): {backend_port}")
        print(f"  Registry port (REP): {registry_port}")
        
    def start(self):
        """ブローカーサービスを開始"""
        if self._running:
            return
            
        self._running = True
        
        # メッセージプロキシスレッド開始
        self._broker_thread = threading.Thread(target=self._run_proxy, daemon=True)
        self._broker_thread.start()
        
        # レジストリサーバースレッド開始
        self._registry_thread = threading.Thread(target=self._run_registry, daemon=True)
        self._registry_thread.start()
        
        print("Message Broker started successfully")
        
    def stop(self):
        """ブローカーサービスを停止"""
        print("Stopping Message Broker...")
        self._running = False
        
        # スレッド終了待機
        time.sleep(0.5)
        
        # ソケット終了
        self.frontend.close()
        self.backend.close()
        self.registry_socket.close()
        self.context.term()
        
        print("Message Broker stopped")
        
    def _run_proxy(self):
        """メッセージプロキシを実行（メインのブローカー機能）"""
        print("Starting message proxy...")
        
        poller = zmq.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        poller.register(self.backend, zmq.POLLIN)
        
        while self._running:
            try:
                sockets = dict(poller.poll(100))  # 100ms timeout
                
                # フロントエンド（パブリッシャー）からのメッセージ
                if self.frontend in sockets:
                    message = self.frontend.recv_multipart(zmq.NOBLOCK)
                    self.backend.send_multipart(message)
                    
                    # トピック統計更新
                    if len(message) >= 2:
                        topic = message[0].decode()
                        self._update_topic_stats(topic)
                
                # バックエンド（サブスクライバー）からの購読情報
                if self.backend in sockets:
                    subscription = self.backend.recv_multipart(zmq.NOBLOCK)
                    self.frontend.send_multipart(subscription)
                    
                    # 購読管理
                    if len(subscription) >= 1:
                        self._handle_subscription(subscription[0])
                        
            except zmq.Again:
                # タイムアウト時は継続
                continue
            except zmq.ZMQError as e:
                if self._running:
                    print(f"Proxy error: {e}")
                    
    def _run_registry(self):
        """レジストリサーバーを実行"""
        print("Starting registry server...")
        
        while self._running:
            try:
                if self.registry_socket.poll(100, zmq.POLLIN):
                    request = self.registry_socket.recv_json()
                    if isinstance(request, dict):
                        response = self._handle_registry_request(request)
                        self.registry_socket.send_json(response)
                    
            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                if self._running:
                    print(f"Registry error: {e}")
                    
    def _update_topic_stats(self, topic_name: str):
        """トピック統計を更新"""
        if topic_name not in self.topics:
            self.topics[topic_name] = TopicInfo(topic_name, "unknown")
            
        self.topics[topic_name].last_message_time = time.time()
        
    def _handle_subscription(self, subscription_frame: bytes):
        """購読情報を処理"""
        # ZMQ購読フレームの最初のバイトは購読(1)/取消(0)フラグ
        if len(subscription_frame) > 0:
            subscribe_flag = subscription_frame[0]
            topic = subscription_frame[1:].decode()
            
            if topic and topic != "":  # 空文字でない場合のみ処理
                if subscribe_flag == 1:  # 購読
                    if topic not in self.topics:
                        self.topics[topic] = TopicInfo(topic, "unknown")
                    self.topics[topic].subscriber_count += 1
                    print(f"New subscription to topic: {topic}")
                    
                elif subscribe_flag == 0:  # 購読取消
                    if topic in self.topics:
                        self.topics[topic].subscriber_count = max(0, 
                            self.topics[topic].subscriber_count - 1)
                        print(f"Unsubscribed from topic: {topic}")
        
    def _handle_registry_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """レジストリリクエストを処理"""
        action = request.get("action")
        
        if action == "register_publisher":
            return self._register_publisher(
                request["topic"], 
                request.get("message_type", "unknown")
            )
            
        elif action == "register_subscriber":
            return self._register_subscriber(
                request["topic"],
                request.get("message_type", "unknown")
            )
            
        elif action == "get_broker_info":
            return {
                "status": "ok",
                "frontend_port": self.frontend_port,
                "backend_port": self.backend_port,
                "registry_port": self.registry_port
            }
            
        elif action == "list_topics":
            return {
                "status": "ok",
                "topics": [info.to_dict() for info in self.topics.values()]
            }
            
        elif action == "get_topic_info":
            topic_name = request["topic"]
            if topic_name in self.topics:
                return {"status": "ok", "info": self.topics[topic_name].to_dict()}
            else:
                return {"status": "error", "message": "Topic not found"}
                
        elif action == "broker_status":
            earliest_time = time.time()
            if self.topics:
                valid_times = [t.created_time for t in self.topics.values() if t.created_time is not None]
                if valid_times:
                    earliest_time = min(valid_times)
            
            return {
                "status": "ok",
                "running": self._running,
                "topic_count": len(self.topics),
                "uptime": time.time() - earliest_time
            }
            
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    def _register_publisher(self, topic: str, message_type: str) -> Dict[str, Any]:
        """パブリッシャーを登録"""
        if topic not in self.topics:
            self.topics[topic] = TopicInfo(topic, message_type)
        
        self.topics[topic].publisher_count += 1
        self.active_publishers.add(topic)
        
        return {
            "status": "ok",
            "frontend_port": self.frontend_port,
            "message": f"Publisher registered for topic: {topic}"
        }
        
    def _register_subscriber(self, topic: str, message_type: str) -> Dict[str, Any]:
        """サブスクライバーを登録"""
        if topic not in self.topics:
            self.topics[topic] = TopicInfo(topic, message_type)
            
        return {
            "status": "ok", 
            "backend_port": self.backend_port,
            "message": f"Subscriber registered for topic: {topic}"
        }


def signal_handler(signum, frame):
    """シグナルハンドラー（Ctrl+C対応）"""
    print("\nReceived interrupt signal")
    sys.exit(0)


def main():
    """メインブローカーサービスを起動"""
    # シグナルハンドラー設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ブローカー作成・起動
    broker = MessageBroker()
    
    try:
        broker.start()
        
        print("\n=== HOS Message Broker Running ===")
        print("Press Ctrl+C to stop")
        print(f"Publishers connect to: tcp://localhost:{broker.frontend_port}")
        print(f"Subscribers connect to: tcp://localhost:{broker.backend_port}")
        print(f"Registry available at: tcp://localhost:{broker.registry_port}")
        print("==========================================\n")
        
        # メインループ（監視・統計表示）
        last_stats_time = 0
        while True:
            time.sleep(1)
            
            # 定期的に統計表示
            current_time = time.time()
            if current_time - last_stats_time >= 30:  # 30秒おきに統計表示
                print(f"\n--- Broker Stats [{datetime.now().strftime('%H:%M:%S')}] ---")
                print(f"Active topics: {len(broker.topics)}")
                for topic_name, info in broker.topics.items():
                    last_msg = "Never" if info.last_message_time is None else \
                             f"{current_time - info.last_message_time:.1f}s ago"
                    print(f"  {topic_name}: {info.publisher_count} pubs, "
                          f"{info.subscriber_count} subs, last msg: {last_msg}")
                print("----------------------------------------\n")
                last_stats_time = current_time
                
    except KeyboardInterrupt:
        pass
    finally:
        broker.stop()


if __name__ == "__main__":
    main()