ダメ絶対音感を深層学習で実現してみました。ダメ相対音感くらいまではできたかなぁと。

# Usage

## 準備

~~~ bash
$ pip3 install --upgrade funcy h5py keras librosa matplotlib pyaudio tensorflow-gpu
~~~

## 訓練

~~~ bash
$ python3 train.py
~~~

遅すぎて待っていられない場合は、train.pyの`return rcompose(wide_residual_net(),`を`return rcompose(squeeze_net(),`にしてみてください。学習速度が少し速くなります。たぶん、少し精度が落ちますけど。

## 訓練結果の確認

~~~ bash
$ python3 check.py
~~~

検証用データでの、`{正解の声優}:{正解のキャラ}\t{推測した声優}:{推測したキャラ}`が出力されます。

キャラあたりの訓練データの数にばらつきがあるので、データ量が少ないキャラは間違えやすいです。無口キャラでデータ量が少ないはずなのに確実に識別される東山奈央の志摩リンすげー。

## ダメ絶対音感の実行

~~~ bash
$ python3 useless_absolute_pitch.py
~~~

多分、マイクだと精度が出ません。私はUbuntu16.04の上で、[このページ](https://ameblo.jp/ninjin-drink/entry-12153085235.html)のやり方を使用して、Chromeで再生中のAmazonプライム・ビデオの音でやりました。

# Notes

xxx
