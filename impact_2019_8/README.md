# Integrating Data Layout Transformations with the Polyhedral Model

## 要件
- 入力はPythonの構造体を使用する
- 出力はllvm irを出力してjitコンパイルを行う

## python 構造体 ir

1. 計算の構造を表現
2. データレイアウト、スケジュールの最適化のメタデータを含む

## アーキテクチャ

1. フロントエンド: Python構造体を解析し、中間表現(IR)を生成
2. 最適化パス: ポリヘドロンモデルを使用してデータレイアウトとスケジュールを最適化
3. バックエンド: LLVM IRを生成し、JITコンパイルを実行


## 使用ライブラリ
- `llvmlite`: LLVM IRの生成とJITコンパイルのため
- `islpy`: ポリヘドロンモデルの操作のため

## lint
```
ruff check . --fix
ruff format .
```
