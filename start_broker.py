#!/usr/bin/env python3
"""
HOS Message Broker - 独立したブローカーサービス起動スクリプト
バックグラウンドで実行可能な完全に独立したメッセージブローカー
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hos_core.broker import main

if __name__ == "__main__":
    main()