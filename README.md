# 概要
OpenAIテスト用

# 環境構築
ルートディレクトリへ.envを追加し、OPENAI_API_KEYにAPIキーを登録

# 参考
## pip index versions
https://zenn.dev/hiroga/articles/pip-index-versions
pipでインストール可能なパッケージのバージョン一覧を表示。このコマンドを使用してバージョンを確認し、requirement.txtに記載してインストールにも使える。実験的コマンドのため削除される可能性もある。

```bash
$ pip index versions langchain-community
WARNING: pip index is currently an experimental command. It may be removed/changed in a future release without prior warning.
langchain-community (0.3.14)
Available versions: 0.3.14, 0.3.13, 0.3.12, 0.3.11, 0.3.10, 0.3.9, 0.3.8, 0.3.7, 0.3.6, 0.3.5, 0.3.4, 0.3.3, 0.3.2, 0.3.1, 0.3.0, 0.2.19, 0.2.18, 0.2.17, 0.2.16, 0.2.15, 0.2.13, 0.2.12, 0.2.11, 0.2.10, 0.2.9, 0.2.7, 0.2.6, 0.2.5, 0.2.4, 0.2.3, 0.2.2, 0.2.1, 0.2.0, 0.0.38, 0.0.37, 0.0.36, 0.0.35, 0.0.34, 0.0.33, 0.0.32, 0.0.31, 0.0.30, 0.0.29, 0.0.28, 0.0.27, 0.0.26, 0.0.25, 0.0.24, 0.0.23, 0.0.22, 0.0.21, 0.0.20, 0.0.19, 0.0.18, 0.0.17, 0.0.16, 0.0.15, 0.0.14, 0.0.13, 0.0.12, 0.0.11, 0.0.10, 0.0.8, 0.0.7, 0.0.6, 0.0.5, 0.0.4, 0.0.3, 0.0.2, 0.0.1
```