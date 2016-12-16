# coding: utf-8

from __future__ import print_function
from flask import Flask, request, jsonify
import argparse

import cv2
import numpy as np
import chainer

import googlenetbn                  # 他のアーキテクチャを使いたい場合はここを書き換えてください


WIDTH = 256                         # リサイズ後の幅
HEIGHT = 256                        # リサイズ後の高さ
LIMIT = 3                           # クラス数

model = googlenetbn.GoogLeNetBN()   # 他のアーキテクチャを使いたい場合はここを書き換えてください

app = Flask(__name__)

# JSON 中の日本語を ASCII コードに変換しないようにする (curl コマンドで見やすくするため。ASCII に変換しても特に問題ない)
app.config['JSON_AS_ASCII'] = False


# train_imagenet.py PreprocessedDataset の get_example() を参考
def preproduce(image, crop_size, mean):
    # リサイズ
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # (height, width, channel) -> (channel, height, width) に変換
    image = image.transpose(2, 0, 1)

    _, h, w = image.shape

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size

    image = image[:, top:bottom, left:right]
    image -= mean[:, top:bottom, left:right]
    image /= 255

    return image


@app.route('/')
def hello():
    return 'Hello!'


# 画像分類 API
# http://localhost:8090/predict に画像を投げると JSON で結果が返る
@app.route('/predict', methods=['POST'])
def predict():
    # 画像読み込み
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)

    # 前処理
    image = preproduce(image.astype(np.float32), model.insize, mean)

    # 推定
    p = model.predict(np.array([image]))[0].data
    indexes = np.argsort(p)[::-1][:LIMIT]

    # 結果を JSON にして返す
    return jsonify({
        'result': [[classes[index][1], float(p[index])] for index in indexes]
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initmodel', type=str, default='',
                        help='Initialize the model from given file')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--labelmaster', '-l', type=str, default='label_master.txt',
                        help='Label master file')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    args = parser.parse_args()

    mean = np.load(args.mean)
    chainer.serializers.load_npz(args.initmodel, model)

    with open(args.labelmaster, 'r') as f:
        classes = [line.strip().split(' ') for line in f.readlines()]

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    app.run(host='0.0.0.0', port=8090)