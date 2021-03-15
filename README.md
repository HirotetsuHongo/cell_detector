# cell_detector

## 初期設定
### Pipenv
PipenvはPythonの仮想環境を管理するツールです．
本検出器では処理の大部分は後述のDocker内部で行いますが，
画像の読み込み等のためにOpenCVを必要としています．


#### Pipenvのインストール
公式ページ(https://pipenv-ja.readthedocs.io/ja/translate-ja/install.html#installing-pipenvPipenv)
を参考にPipenvをインストールします．

HomebrewまたはLinuxbrewを用いるインストール方法:

``` bash
brew install pipenv
```

pipを用いるインストール方法:

``` bash
pip install --user pipenv
```


#### Pipenvを用いた開発環境の構築
プロジェクトのルートディレクトリで以下のコマンドを実行してください．

``` bash
pipenv install --dev
```


### Docker
DockerはOSを含む軽量な仮想環境を管理するツールです．
本検出器は特定のGPU環境に依存するため，
GPUを必要とする処理をDocker内で完結させることで環境構築を簡単にしました．

#### イメージのビルド
まずは処理の大部分を担うDockerイメージをビルドします．
ビルドのためのスクリプトは既に用意しています．
検出器のルートディレクトリで以下のコマンドを実行してください．

``` bash
pipenv run python start.py
```


## 起動
本検出器のプログラムはDockerコンテナと
Dockerコンテナ内のサーバーを起動した上で実行します．
Dockerコンテナの起動は `start.py` ，
サーバーの起動は `server.py` を用いて行います．

``` bash
pipenv run python start.py
pipenv run python server.py
```

他のプログラムからサーバーを起動する場合も，
`start.main` と `server.main` という関数を用いて起動することができます．


## 学習
まずは検出器を起動します．
``` bash
pipenv run python start.py
```

次に学習に用いる画像を用意します．
[labelImg](https://github.com/tzutalin/labelImg)などを用いてYOLOv3形式のアノテーションを作成します．
画像は `workspace/dataset/image` ，
アノテーションは `workspace/dataset/train` に置きます．

細かい学習時の設定は後述の「検出器の設定」を参照してください．

データセットの設定が終わったら，サーバーを起動します．

``` bash
pipenv run python server.py
```

サーバーを起動したターミナルは待機状態に入るため，
他のターミナルから学習を行います．

``` bash
pipenv run python train.py
```

学習が終了すると，学習済みパラメータは `workspace/dataset/weight` に保存されます．


## 評価
学習後は得られたパラメータの精度を評価する必要があります．
評価に用いるアノテーションは `workspace/dataset/test` に置きます．

サーバーを起動します．

``` bash
pipenv run python start.py
pipenv run python server.py
```

サーバーを起動したターミナルは待機状態に入るため，
他のターミナルから `workspace/dataset/weight` 内のパラメータの評価を行います．

``` bash
pipenv run python test.py
```

## 検出
学習済みのパラメータを `workspace/weight.pt` に置きます．

サーバーを起動します．

``` bash
pipenv run python start.py
pipenv run python server.py
```

サーバーを起動したターミナルは待機状態に入るため，
他のターミナルから検出を行います．
`detect.py` では具体的に画像と検出結果を出力するパスを指定する必要がありますが，
`detect_all.py` を使えば `images` 内の全ての画像に対して検出を行い，
`results` 内に拡張子のみが異なる同じ名前のテキストファイルを出力します．

`detect.py` による検出:

``` bash
pipenv run python detect.py <画像のパス> <出力先のパス>
```

`detect_all.py` による検出:

``` bash
pipenv run python detect_all.py
```

## 検出器の設定
検出器の詳細な設定は `workspace/config/config.yaml` にあります．
各項目について説明します．

*   `batch_size` :
    学習時に一度にする画像の枚数です．
    これを大きくすると学習時間が短縮される一方，
    多くのGPUのメモリを必要とします．
*   `num_channels` :
    チャンネル数です．
    例えば，単一波長の顕微鏡画像のみを扱う場合は1に設定します．
    複数の顕微鏡画像を異なるチャンネルとして重ねた画像を扱う場合は，
    重ねたチャンネルの数に設定します．
    全ての画像は同じチャンネル数である必要があります．
*   `classes` :
    検出する対象（クラス）を列挙して設定します．
*   `height` , `width` :
    入力する画像のサイズではなく，
    内部的に検出器が扱う画像のサイズを設定します．
    入力画像はこのサイズにリサイズされてから検出器に渡されます．
    サイズが大きいほど高精度に検出できると考えられますが，
    多くのGPUのメモリを必要とします．
*   `anchors` :
    アンカー（anchor）とは検出するバウンディングボックスのテンプレートに相当し，
    3x3個のアンカーを必要とします．
    アンカーは手動で設定することもできますが，
    k-means法で設定するとよりよい精度が得られる場合があります．
    `workspace/k_means.py` は以下のように用いることで
    アノテーションデータに対してk-means法を行って9個のアンカーを返します．
    ``` bash
    pipenv run python workspace/k_means.py 9 <アノテーションデータを含むディレクトリ>
    ```
*   `num_epochs` :
    学習を行う回数（エポック数）を設定する．
    一般に，エポック数が多いほど学習は進むが，
    過学習が起こるため適切なエポック数を探す必要がある．
    本検出器では全ての全てのエポックにおけるパラメータが保存されるため，
    評価して最も良い精度のパラメータを採用すれば良い．
*   `TP_IoU` :
    検出したバウンディングボックスがアノテーションと重なっているかどうかを判定するIoUの閾値．
    0から1までの値で設定する．
    検出器の評価時に使用する．
*   `NMS_IoU` :
    検出した2つのバウンディングボックスが同じ物体を指しているかどうかを判定するIoUの閾値．
    0から1までの値で設定する．
    同じ物体を指すバウンディングボックスが複数個あるとき，最も信頼度が高いもの以外は抑制される．
*   `confidency` :
    検出したバウンディングボックスに物体が含まれているかどうかを判定する信頼度の閾値．
    0から1までの値で設定する．
*   `learning_rate` :
    学習率を設定する．
    学習の効率と検出器の精度に関係する．
*   `weight_decay` :
    重み減衰のパラメータを設定する．
    学習の効率と検出器の精度に関係する．
*   `CUDA` :
    GPUを使用するかどうかのブール値を設定する．
*   `path` :
    各種のパスを設定する．
    `train` は学習に用いるアノテーションを含むディレクトリ，
    `test` は評価に用いるアノテーションを含むディレクトリ，
    `image` は `train` と `test` のアノテーションに対応する全ての画像を含むディレクトリ，
    `weight` は学習したパラメータを保存するディレクトリ，
    `weight_test` は評価に用いるパラメータを含むディレクトリ，
    `initial_weight` は初期パラメータまたはFalse，
    `detect_weight` は検出に用いるパラメータのパスに設定する．


