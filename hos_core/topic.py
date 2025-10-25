"""
HOSのトピック管理システム（改良版）
独立したブローカーサービスを使用してmany-to-many通信を実現
パブリッシャーの再起動に対応した動的再接続機能付き
"""

import zmq
import json
import msgpack
import threading
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict


def _type_to_string(message_type: Any) -> str:
    """
    Pythonのタイプを文字列表現に変換
    """
    if isinstance(message_type, str):
        return message_type
    
    if hasattr(message_type, '__name__'):
        # 基本的なタイプ（str, int, float, etc.）
        if message_type.__name__ in ['str', 'int', 'float', 'bool', 'bytes']:
            return f"std_msgs/{message_type.__name__.capitalize()}"
        elif message_type.__name__ == 'ndarray':
            return "std_msgs/Float32MultiArray"
        elif message_type.__name__ == 'list':
            return "std_msgs/Array"
        else:
            return f"custom/{message_type.__name__}"
    
    # typing.Union などの複合タイプの場合
    if hasattr(message_type, '__origin__'):
        origin = message_type.__origin__
        if origin is Union:
            # Union[str, int] -> "std_msgs/Union[String,Int32]"
            args = message_type.__args__
            type_names = [_type_to_string(arg) for arg in args]
            return f"std_msgs/Union[{','.join(type_names)}]"
        elif origin is list:
            # List[float] -> "std_msgs/Float32Array"
            if message_type.__args__:
                inner_type = _type_to_string(message_type.__args__[0])
                return f"{inner_type}Array"
            return "std_msgs/Array"
    
    # フォールバック
    return str(message_type)


@dataclass
class TopicInfo:
    """トピック情報を格納するデータクラス"""
    name: str
    message_type: str
    publisher_count: int = 0
    subscriber_count: int = 0
    last_message_time: Optional[float] = None
    created_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BrokerClient:
    """ブローカーサービスとの通信クライアント"""
    
    def __init__(self, registry_host: str = "localhost", registry_port: int = 5555):
        self.registry_host = registry_host
        self.registry_port = registry_port
        self.context = zmq.Context()
        
    def _send_registry_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """レジストリサーバーにリクエストを送信"""
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒タイムアウト
        socket.connect(f"tcp://{self.registry_host}:{self.registry_port}")
        
        try:
            socket.send_json(request)
            response = socket.recv_json()
            if isinstance(response, dict):
                return response
            return None
        except zmq.Again:
            print(f"Timeout connecting to broker at {self.registry_host}:{self.registry_port}")
            return None
        except Exception as e:
            print(f"Error communicating with broker: {e}")
            return None
        finally:
            socket.close()
            
    def get_broker_info(self) -> Optional[Dict[str, Any]]:
        """ブローカー情報を取得"""
        return self._send_registry_request({"action": "get_broker_info"})
        
    def list_topics(self) -> List[TopicInfo]:
        """トピック一覧を取得"""
        response = self._send_registry_request({"action": "list_topics"})
        if response and response.get("status") == "ok":
            return [TopicInfo(**info) for info in response["topics"]]
        return []
        
    def get_topic_info(self, topic_name: str) -> Optional[TopicInfo]:
        """特定のトピック情報を取得"""
        response = self._send_registry_request({
            "action": "get_topic_info",
            "topic": topic_name
        })
        if response and response.get("status") == "ok":
            return TopicInfo(**response["info"])
        return None


class TopicClient(BrokerClient):
    """トピック操作を行うクライアントクラス（レガシー互換性）"""
    
    def echo_topic(self, topic_name: str, count: Optional[int] = None):
        """トピックのメッセージをエコー（表示）"""
        print(f"Listening to topic '{topic_name}'... (Press Ctrl+C to stop)")
        
        def callback(message):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {topic_name}: {message}")
            
        subscriber = Subscriber(topic_name, callback, str)
        try:
            subscriber.start()
            
            message_count = 0
            while True:
                if count is not None and message_count >= count:
                    break
                time.sleep(0.1)
                message_count += 1
        except KeyboardInterrupt:
            print("\nStopped listening.")
        finally:
            subscriber.close()
            
    def close(self):
        """クライアント終了"""
        self.context.term()


class Publisher:
    """
    新しいブローカー方式対応パブリッシャー
    自動再接続機能付き
    """
    
    def __init__(self, topic: str, message_type: Any = str, 
                 registry_host: str = "localhost", registry_port: int = 5555):
        self.context = zmq.Context()
        self.topic = topic
        self.message_type = _type_to_string(message_type)
        self.registry_host = registry_host
        self.registry_port = registry_port
        
        self.socket = None
        self.frontend_port = None
        self._connected = False
        self._lock = threading.Lock()
        
        # 初期接続
        self._connect_to_broker()
        
    def _connect_to_broker(self) -> bool:
        """ブローカーに接続"""
        with self._lock:
            try:
                # ブローカー情報を取得
                client = BrokerClient(self.registry_host, self.registry_port)
                broker_info = client.get_broker_info()
                
                if not broker_info or broker_info.get("status") != "ok":
                    print("Failed to get broker info")
                    return False
                    
                self.frontend_port = broker_info["frontend_port"]
                
                # 既存ソケットがあれば閉じる
                if self.socket:
                    self.socket.close()
                
                # 新しいソケット作成
                self.socket = self.context.socket(zmq.PUB)
                self.socket.connect(f"tcp://{self.registry_host}:{self.frontend_port}")
                
                # パブリッシャー登録
                response = client._send_registry_request({
                    "action": "register_publisher",
                    "topic": self.topic,
                    "message_type": self.message_type
                })
                
                if response and response.get("status") == "ok":
                    self._connected = True
                    print(f"Publisher connected to broker: {self.topic}")
                    return True
                else:
                    print(f"Failed to register publisher: {response}")
                    return False
                    
            except Exception as e:
                print(f"Error connecting to broker: {e}")
                self._connected = False
                return False
                
    def _ensure_connection(self) -> bool:
        """接続を確認し、必要に応じて再接続"""
        if not self._connected:
            return self._connect_to_broker()
        return True
        
    def publish(self, message: Any, retry_count: int = 3):
        """メッセージをパブリッシュ（自動再接続付き）"""
        for attempt in range(retry_count):
            if not self._ensure_connection():
                if attempt < retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))  # 指数バックオフ
                    continue
                else:
                    print(f"Failed to publish message after {retry_count} attempts")
                    return False
                    
            try:
                # メッセージをバイナリにエンコード (msgpack使用)
                if isinstance(message, str):
                    # 文字列の場合はdictにラップ
                    message_bytes = msgpack.packb({"data": message}, use_bin_type=True)
                elif isinstance(message, (dict, list, int, float, bool, bytes)):
                    # msgpackで直接シリアライズ可能な型
                    message_bytes = msgpack.packb(message, use_bin_type=True)
                else:
                    # その他の型は文字列化してからdictにラップ
                    message_bytes = msgpack.packb({"data": str(message)}, use_bin_type=True)
                    
                # パブリッシュ
                with self._lock:
                    if self.socket:
                        self.socket.send_multipart([
                            self.topic.encode(),
                            message_bytes
                        ], zmq.NOBLOCK)
                return True
                
            except zmq.Again:
                print(f"Publisher queue full, retrying... (attempt {attempt + 1})")
                self._connected = False
                if attempt < retry_count - 1:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error publishing message: {e}")
                self._connected = False
                if attempt < retry_count - 1:
                    time.sleep(0.5)
                    
        return False
        
    def close(self):
        """パブリッシャー終了"""
        with self._lock:
            if self.socket:
                self.socket.close()
                self.socket = None
            self._connected = False
        self.context.term()


class Subscriber:
    """
    新しいブローカー方式対応サブスクライバー
    パブリッシャーが再起動しても継続受信可能
    """
    
    def __init__(self, topic: str, callback, message_type: Any = str,
                 registry_host: str = "localhost", registry_port: int = 5555):
        self.context = zmq.Context()
        self.topic = topic
        self.callback = callback
        self.message_type = _type_to_string(message_type)
        self.registry_host = registry_host
        self.registry_port = registry_port
        
        self.socket = None
        self.backend_port = None
        self._running = False
        self._connected = False
        self.worker_thread = None
        self._lock = threading.Lock()
        
        # 初期接続
        self._connect_to_broker()
        
    def _connect_to_broker(self) -> bool:
        """ブローカーに接続"""
        with self._lock:
            try:
                # ブローカー情報を取得
                client = BrokerClient(self.registry_host, self.registry_port)
                broker_info = client.get_broker_info()
                
                if not broker_info or broker_info.get("status") != "ok":
                    print("Failed to get broker info")
                    return False
                    
                self.backend_port = broker_info["backend_port"]
                
                # 既存ソケットがあれば閉じる
                if self.socket:
                    self.socket.close()
                
                # 新しいソケット作成
                self.socket = self.context.socket(zmq.SUB)
                self.socket.connect(f"tcp://{self.registry_host}:{self.backend_port}")
                self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
                
                # サブスクライバー登録
                response = client._send_registry_request({
                    "action": "register_subscriber",
                    "topic": self.topic,
                    "message_type": self.message_type
                })
                
                if response and response.get("status") == "ok":
                    self._connected = True
                    print(f"Subscriber connected to broker: {self.topic}")
                    return True
                else:
                    print(f"Failed to register subscriber: {response}")
                    return False
                    
            except Exception as e:
                print(f"Error connecting to broker: {e}")
                self._connected = False
                return False
                
    def _worker(self):
        """メッセージ受信ワーカー（再接続機能付き）"""
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0
        
        while self._running:
            try:
                if not self._connected:
                    print(f"Reconnecting subscriber for topic: {self.topic}")
                    if self._connect_to_broker():
                        reconnect_delay = 1.0  # 成功時はdelayをリセット
                    else:
                        time.sleep(min(reconnect_delay, max_reconnect_delay))
                        reconnect_delay *= 2  # 指数バックオフ
                        continue
                
                # メッセージ受信
                with self._lock:
                    if self.socket and self.socket.poll(1000, zmq.POLLIN):  # 1秒タイムアウト
                        try:
                            topic_bytes, message_bytes = self.socket.recv_multipart(zmq.NOBLOCK)
                            
                            try:
                                # msgpackデコード
                                data = msgpack.unpackb(message_bytes, raw=False)
                            except (msgpack.exceptions.ExtraData, 
                                    msgpack.exceptions.UnpackException,
                                    ValueError):
                                # msgpackデコード失敗時はJSONを試行（後方互換性）
                                try:
                                    data = json.loads(message_bytes.decode())
                                except json.JSONDecodeError:
                                    # JSONでもない場合は文字列として扱う
                                    data = message_bytes.decode()
                                
                            # コールバック実行
                            self.callback(data)
                            
                        except zmq.Again:
                            # NOBLOCKでタイムアウト
                            continue
                            
                    elif not self._running:
                        break
                        
            except zmq.ZMQError as e:
                if self._running:
                    print(f"Subscriber connection error: {e}, reconnecting...")
                    self._connected = False
                    time.sleep(1.0)
            except Exception as e:
                if self._running:
                    print(f"Subscriber error: {e}")
                    time.sleep(0.5)
                    
    def start(self):
        """サブスクライバー開始"""
        if self._running:
            return
            
        self._running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def stop(self):
        """サブスクライバー停止"""
        self._running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
    def close(self):
        """サブスクライバー終了"""
        self.stop()
        with self._lock:
            if self.socket:
                self.socket.close()
                self.socket = None
            self._connected = False
        self.context.term()


# レガシーサポート用のTopicManagerクラス
class TopicManager:
    """レガシー互換性のためのダミークラス"""
    
    def __init__(self, registry_port: int = 5555):
        print("WARNING: TopicManager is deprecated. Use the independent broker service instead.")
        print("Start the broker with: python -m hos_core.broker")
        
    def start_registry_server(self):
        print("TopicManager registry server is deprecated.")
        
    def stop(self):
        pass