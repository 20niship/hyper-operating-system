"""
HOSのトピック管理システム
ZeroMQを使用してROSのようなpub/sub機能を提供
"""

import zmq
import json
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class TopicInfo:
    """トピック情報を格納するデータクラス"""
    name: str
    message_type: str
    publisher_count: int = 0
    subscriber_count: int = 0
    last_message_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TopicManager:
    """トピックの管理とメッセージルーティングを行うクラス"""
    
    def __init__(self, registry_port: int = 5555):
        self.context = zmq.Context()
        self.registry_port = registry_port
        self.topics: Dict[str, TopicInfo] = {}
        self.publishers: Dict[str, zmq.Socket] = {}
        self.subscribers: Dict[str, zmq.Socket] = {}
        self._running = False
        
        # レジストリサーバー（トピック情報管理）
        self.registry_socket = self.context.socket(zmq.REP)
        self.registry_socket.bind(f"tcp://*:{registry_port}")
        
    def start_registry_server(self):
        """レジストリサーバーを開始"""
        self._running = True
        
        def registry_worker():
            while self._running:
                try:
                    # レジストリリクエストを受信（タイムアウト付き）
                    if self.registry_socket.poll(100, zmq.POLLIN):
                        message = self.registry_socket.recv_json()
                        response = self._handle_registry_request(message)
                        self.registry_socket.send_json(response)
                except zmq.ZMQError as e:
                    if self._running:  # 正常終了時はエラーを無視
                        print(f"Registry server error: {e}")
                        
        registry_thread = threading.Thread(target=registry_worker, daemon=True)
        registry_thread.start()
        
    def stop(self):
        """サービス停止"""
        self._running = False
        time.sleep(0.2)  # スレッド終了を待機
        
        # ソケットを閉じる
        self.registry_socket.close()
        for socket in self.publishers.values():
            socket.close()
        for socket in self.subscribers.values():
            socket.close()
        self.context.term()
        
    def _handle_registry_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """レジストリリクエストを処理"""
        action = message.get("action")
        
        if action == "register_publisher":
            return self._register_publisher(
                message["topic"], 
                message["message_type"], 
                message["port"]
            )
        elif action == "register_subscriber":
            return self._register_subscriber(
                message["topic"],
                message["message_type"]
            )
        elif action == "get_topic_info":
            topic_name = message["topic"]
            if topic_name in self.topics:
                return {"status": "ok", "info": self.topics[topic_name].to_dict()}
            else:
                return {"status": "error", "message": "Topic not found"}
        elif action == "list_topics":
            return {
                "status": "ok", 
                "topics": [info.to_dict() for info in self.topics.values()]
            }
        elif action == "get_publisher_port":
            topic_name = message["topic"]
            if topic_name in self.topics:
                return {"status": "ok", "port": 5556 + hash(topic_name) % 10000}
            else:
                return {"status": "error", "message": "Topic not found"}
        else:
            return {"status": "error", "message": "Unknown action"}
            
    def _register_publisher(self, topic: str, message_type: str, port: int) -> Dict[str, Any]:
        """パブリッシャーを登録"""
        if topic not in self.topics:
            self.topics[topic] = TopicInfo(topic, message_type)
        
        self.topics[topic].publisher_count += 1
        return {"status": "ok", "port": port}
        
    def _register_subscriber(self, topic: str, message_type: str) -> Dict[str, Any]:
        """サブスクライバーを登録"""
        if topic not in self.topics:
            self.topics[topic] = TopicInfo(topic, message_type)
        
        self.topics[topic].subscriber_count += 1
        
        # パブリッシャーのポートを返す
        publisher_port = 5556 + hash(topic) % 10000
        return {"status": "ok", "publisher_port": publisher_port}


class TopicClient:
    """トピック操作を行うクライアントクラス"""
    
    def __init__(self, registry_host: str = "localhost", registry_port: int = 5555):
        self.context = zmq.Context()
        self.registry_host = registry_host
        self.registry_port = registry_port
        
    def _send_registry_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """レジストリサーバーにリクエストを送信"""
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.registry_host}:{self.registry_port}")
        
        try:
            socket.send_json(request)
            response = socket.recv_json()
            return response
        finally:
            socket.close()
            
    def list_topics(self) -> List[TopicInfo]:
        """トピック一覧を取得"""
        try:
            response = self._send_registry_request({"action": "list_topics"})
            if response["status"] == "ok":
                return [TopicInfo(**info) for info in response["topics"]]
            else:
                return []
        except Exception as e:
            print(f"Error listing topics: {e}")
            return []
            
    def get_topic_info(self, topic_name: str) -> Optional[TopicInfo]:
        """特定のトピック情報を取得"""
        try:
            response = self._send_registry_request({
                "action": "get_topic_info",
                "topic": topic_name
            })
            if response["status"] == "ok":
                return TopicInfo(**response["info"])
            else:
                return None
        except Exception as e:
            print(f"Error getting topic info: {e}")
            return None
            
    def echo_topic(self, topic_name: str, count: Optional[int] = None):
        """トピックのメッセージをエコー（表示）"""
        try:
            # パブリッシャーのポートを取得
            response = self._send_registry_request({
                "action": "get_publisher_port",
                "topic": topic_name
            })
            
            if response["status"] != "ok":
                print(f"Topic '{topic_name}' not found")
                return
                
            publisher_port = response["port"]
            
            # サブスクライバーとして接続
            socket = self.context.socket(zmq.SUB)
            socket.connect(f"tcp://{self.registry_host}:{publisher_port}")
            socket.setsockopt_string(zmq.SUBSCRIBE, topic_name)
            
            print(f"Listening to topic '{topic_name}'... (Press Ctrl+C to stop)")
            
            message_count = 0
            try:
                while True:
                    if count is not None and message_count >= count:
                        break
                        
                    # メッセージを受信（タイムアウト付き）
                    if socket.poll(100, zmq.POLLIN):
                        topic, message = socket.recv_multipart()
                        
                        try:
                            # JSONデコード
                            data = json.loads(message.decode())
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            print(f"[{timestamp}] {topic_name}: {data}")
                            message_count += 1
                        except json.JSONDecodeError:
                            # JSON以外のメッセージの場合
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {topic_name}: {message.decode()}")
                            message_count += 1
                            
            except KeyboardInterrupt:
                print("\nStopped listening.")
            finally:
                socket.close()
                
        except Exception as e:
            print(f"Error echoing topic: {e}")
            
    def close(self):
        """クライアント終了"""
        self.context.term()


class Publisher:
    """メッセージパブリッシャー"""
    
    def __init__(self, topic: str, message_type: str = "std_msgs/String", 
                 registry_host: str = "localhost", registry_port: int = 5555):
        self.context = zmq.Context()
        self.topic = topic
        self.message_type = message_type
        self.registry_host = registry_host
        self.registry_port = registry_port
        
        # ポート番号を計算（トピック名のハッシュから）
        self.port = 5556 + hash(topic) % 10000
        
        # パブリッシャーソケット作成
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        
        # レジストリに登録
        self._register_with_registry()
        
        # ソケット接続の安定化のため少し待機
        time.sleep(0.1)
        
    def _register_with_registry(self):
        """レジストリサーバーに登録"""
        client_socket = self.context.socket(zmq.REQ)
        client_socket.connect(f"tcp://{self.registry_host}:{self.registry_port}")
        
        try:
            request = {
                "action": "register_publisher",
                "topic": self.topic,
                "message_type": self.message_type,
                "port": self.port
            }
            client_socket.send_json(request)
            response = client_socket.recv_json()
        finally:
            client_socket.close()
            
    def publish(self, message: Any):
        """メッセージをパブリッシュ"""
        if isinstance(message, dict):
            message_data = json.dumps(message)
        elif isinstance(message, str):
            message_data = message
        else:
            message_data = json.dumps({"data": str(message)})
            
        self.socket.send_multipart([
            self.topic.encode(),
            message_data.encode()
        ])
        
    def close(self):
        """パブリッシャー終了"""
        self.socket.close()
        self.context.term()


class Subscriber:
    """メッセージサブスクライバー"""
    
    def __init__(self, topic: str, callback, message_type: str = "std_msgs/String",
                 registry_host: str = "localhost", registry_port: int = 5555):
        self.context = zmq.Context()
        self.topic = topic
        self.callback = callback
        self.message_type = message_type
        self.registry_host = registry_host
        self.registry_port = registry_port
        self._running = False
        
        # レジストリからパブリッシャー情報を取得
        self.publisher_port = self._get_publisher_port()
        
        # サブスクライバーソケット作成
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{registry_host}:{self.publisher_port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        
    def _get_publisher_port(self) -> int:
        """レジストリからパブリッシャーのポートを取得"""
        client_socket = self.context.socket(zmq.REQ)
        client_socket.connect(f"tcp://{self.registry_host}:{self.registry_port}")
        
        try:
            # サブスクライバーとして登録
            request = {
                "action": "register_subscriber",
                "topic": self.topic,
                "message_type": self.message_type
            }
            client_socket.send_json(request)
            response = client_socket.recv_json()
            
            return response["publisher_port"]
        finally:
            client_socket.close()
            
    def start(self):
        """サブスクライバー開始"""
        self._running = True
        
        def worker():
            while self._running:
                try:
                    if self.socket.poll(100, zmq.POLLIN):
                        topic, message = self.socket.recv_multipart()
                        
                        try:
                            # JSONデコード
                            data = json.loads(message.decode())
                        except json.JSONDecodeError:
                            # JSON以外の場合は文字列として扱う
                            data = message.decode()
                            
                        # コールバック実行
                        self.callback(data)
                        
                except zmq.ZMQError as e:
                    if self._running:
                        print(f"Subscriber error: {e}")
                        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        
    def stop(self):
        """サブスクライバー停止"""
        self._running = False
        time.sleep(0.2)
        
    def close(self):
        """サブスクライバー終了"""
        self.stop()
        self.socket.close()
        self.context.term()