#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

train_lmdb = "E:/face_attribute_recognition_yqy/train_lmdb"
validation_lmdb = "E:/face_attribute_recognition_yqy/val_lmdb"
image_path = "E:/face_attribute_recognition_yqy/train1/"
label_path="E:/face_attribute_recognition_yqy/celebatrainattr1.txt"

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
#    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
#    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
#    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

#    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tobytes())


if __name__ =="__main__":
    count = 0
    print '\nCreating train_lmdb'
    in_db = lmdb.open(train_lmdb, map_size=int(1.5e9))
    with in_db.begin(write=True) as in_txn:# 创建操作数据库句柄
        with open(label_path) as labels:
            while True:
                count+=1
                line = labels.readline()
                if not line:
                    break
                # if count % 10 == 0:
                #     continue
                line = line.split()
                img = cv2.imread(image_path+line[0])
                img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
                datum = make_datum(img, int(line[1]))
                in_txn.put('{:0>5d}'.format(count), datum.SerializeToString())
                print '{:0>5d}'.format(count)
    in_db.close()
    print '\nFinished processing all images'