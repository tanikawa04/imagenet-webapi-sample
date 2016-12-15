# coding: utf-8

import os
import shutil
import re
import random

import cv2
import numpy as np


WIDTH = 256                             # リサイズ後の幅
HEIGHT = 256                            # リサイズ後の高さ

SRC_BASE_PATH = './original'            # ダウンロードした画像が格納されているベースディレクトリ
DST_BASE_PATH = './resized'             # リサイズ後の画像を格納するベースディレクトリ

LABEL_MASTER_PATH = 'label_master.txt'  # クラスとラベルの対応をまとめたファイル
TRAIN_LABEL_PATH = 'train_label.txt'    # 学習用のラベルファイル
VAL_LABEL_PATH = 'val_label.txt'        # 検証用のラベルファイル

VAL_RATE = 0.2                          # 検証データの割合


if __name__ == '__main__':
    with open(LABEL_MASTER_PATH, 'r') as f:
        classes = [line.strip().split(' ') for line in f.readlines()]

    # リサイズ後の画像の格納先を初期化
    if os.path.exists(DST_BASE_PATH):
        shutil.rmtree(DST_BASE_PATH)

    os.mkdir(DST_BASE_PATH)

    train_dataset = []
    val_dataset = []

    for c in classes:
        os.mkdir(os.path.join(DST_BASE_PATH, c[0]))

        class_dir_path = os.path.join(SRC_BASE_PATH, c[0])

        # JPEG か PNG 画像のみ取得
        files = [
            file for file in os.listdir(class_dir_path)
            if re.search(r'\.(jpe?g|png)$', file, re.IGNORECASE)
        ]

        # リサイズしてファイル出力
        for file in files:
            src_path = os.path.join(class_dir_path, file)
            image = cv2.imread(src_path)
            resized_image = cv2.resize(image, (WIDTH, HEIGHT))
            cv2.imwrite(os.path.join(DST_BASE_PATH, c[0], file), resized_image)

        # 学習・検証のラベルデータを作成
        bound = int(len(files) * (1 - VAL_RATE))
        random.shuffle(files)
        train_files = files[:bound]
        val_files = files[bound:]

        train_dataset.extend([(os.path.join(c[0], file), c[2]) for file in train_files])
        val_dataset.extend([(os.path.join(c[0], file), c[2]) for file in val_files])

    # 学習用ラベルファイルを出力
    with open(TRAIN_LABEL_PATH, 'w') as f:
        for d in train_dataset:
            f.write(' '.join(d) + '\n')

    # 検証用ラベルファイルを出力
    with open(VAL_LABEL_PATH, 'w') as f:
        for d in val_dataset:
            f.write(' '.join(d) + '\n')
