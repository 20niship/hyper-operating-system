# HOS Many-to-Many Pub/Sub システム

パブリッシャーの再起動に対応した堅牢なmany-to-many pub/subシステムです。

## 主な特徴

- **独立したブローカーサービス**: パブリッシャーとサブスクライバーから独立して動作
- **自動再接続**: パブリッシャーやブローカーが再起動しても自動的に再接続
- **Many-to-Many通信**: 複数のパブリッシャーと複数のサブスクライバーが同時に動作可能
- **ZeroMQ XPUB/XSUB**: 高性能なメッセージプロキシパターンを使用

## システム構成

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Publisher 1   │    │                 │    │  Subscriber 1   │
│                 │───▶│  Message Broker │◀───│                 │
└─────────────────┘    │   (XPUB/XSUB)   │    └─────────────────┘
                       │                 │    
┌─────────────────┐    │   Frontend:5556 │    ┌─────────────────┐
│   Publisher 2   │───▶│   Backend :5557 │◀───│  Subscriber 2   │
└─────────────────┘    │   Registry:5555 │    └─────────────────┘
```

## 使用方法

### 1. ブローカーサービスを開始

独立したターミナルでブローカーを起動：

```bash
# 方法1: スクリプトから起動
python start_broker.py

# 方法2: モジュールとして起動  
python -m hos_core
```

ブローカーが正常に起動すると以下のような出力が表示されます：

```
Message Broker initialized:
  Publisher port (XSUB): 5556
  Subscriber port (XPUB): 5557
  Registry port (REP): 5555

=== HOS Message Broker Running ===
Press Ctrl+C to stop
Publishers connect to: tcp://localhost:5556
Subscribers connect to: tcp://localhost:5557
Registry available at: tcp://localhost:5555
==========================================
```

### 2. サブスクライバーを起動

別のターミナルでサブスクライバーを起動：

```bash
python sub_node.py
```

### 3. パブリッシャーを起動

さらに別のターミナルでパブリッシャーを起動：

```bash
python pub_node.py
```

## 再起動テスト

### パブリッシャー再起動テスト

1. ブローカーとサブスクライバーを起動
2. パブリッシャーを起動してメッセージが受信されることを確認
3. パブリッシャーを停止（Ctrl+C）
4. パブリッシャーを再起動
5. サブスクライバーが継続してメッセージを受信することを確認

### ブローカー再起動テスト

1. 全てのノードを起動
2. ブローカーを停止（Ctrl+C）
3. ブローカーを再起動
4. パブリッシャーとサブスクライバーが自動的に再接続することを確認

## プログラムでの使用方法

### パブリッシャーの作成

```python
from hos_core.topic import Publisher

# パブリッシャーを作成（自動的にブローカーに接続）
publisher = Publisher("/my_topic", "std_msgs/String")

# メッセージをパブリッシュ（自動再接続付き）
message = {"data": "Hello, World!", "timestamp": time.time()}
success = publisher.publish(message)

# 終了
publisher.close()
```

### サブスクライバーの作成

```python
from hos_core.topic import Subscriber

def message_callback(message):
    print(f"Received: {message}")

# サブスクライバーを作成（自動的にブローカーに接続）
subscriber = Subscriber("/my_topic", message_callback, "std_msgs/String")

# 受信開始（バックグラウンドスレッドで動作）
subscriber.start()

# メインループ
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

# 終了
subscriber.close()
```

## トラブルシューティング

### ブローカーに接続できない

- ブローカーが起動していることを確認
- ファイアウォールでポート5555-5557がブロックされていないか確認

### メッセージが届かない

- ブローカーの統計表示で接続状況を確認
- パブリッシャーとサブスクライバーが同じトピック名を使用していることを確認

### 自動再接続が動作しない

- ネットワークの設定を確認
- ログ出力でエラーメッセージを確認

## ポート使用

- **5555**: レジストリ・管理用
- **5556**: パブリッシャー接続用（XSUB）
- **5557**: サブスクライバー接続用（XPUB）

これらのポートが他のプロセスで使用されていないことを確認してください。
