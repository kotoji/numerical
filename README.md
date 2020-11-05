# 数値計算ライブラリ
数値計算に関するアルゴリズムの実装をまとめておくところ。  
色々 numpy に依存してやっていたりするが気にしない。これは勉強用なので。

## ディレクトリ構成
Python のライブラリの構成に慣れるため numpy を参考にディレクトリをきる。  
実装は現状、例えば `numerical.linalg` モジュールであれば、`linalg.py` か或いはそれが依存する非公開の実装をまとめた `_linalg.py` にある。  
以下は、ディレクトリ構成のイメージ。

```tree
.
├── numerical
│   ├── linalg
│   │    ├── linalg.py
│   │    └── _linalg.py
│   ├── integrate
│   └── tests
│       ├── linalg
│       │  └── test_linalg.py
│       └── integrate
└── README.md
```

## テスト
現状 pytest による雑ユニットテストのみ。

- すべてのユニットテストを実行
  ```
  $ python -m pytest
  ```
- `./numerical/tests/linalg` 以下のテストを実行
  ```
  $ python -m pytest numerical/tests/linalg
  ```

## デモの動かし方
`demo/lorenz.py` は以下のコマンドで実行できる。
```
$ python -m demo.lorenz
```
