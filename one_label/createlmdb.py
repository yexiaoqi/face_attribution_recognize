#!/usr/bin/env python
# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2



def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    # Size of images
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tobytes())

def gen_lmdb(lmdb_file,label_path,image_path):
    count=0
    lmdb_env = lmdb.open(lmdb_file, map_size=int(3e9))#map_size指的是数据库的最大容量，根据需求设置
    #lmdb_txn = lmdb_env.begin(write=True)
    with lmdb_env.begin(write=True) as lmdb_txn:
        with open(label_path) as labels:
            #while True:
            #count+=1
            #imgpath=labels.readline()
            for imgpath in labels.readlines():
                imgpath=imgpath.split('\n')[0]
                imgname=imgpath.split(' ')[0]
                male=imgpath.split(' ')[1]
                male=int(male)
                #print(imgname)
                img=cv2.imread(image_path+imgname)
                #datum = caffe.io.array_to_datum(img, male)
                datum = make_datum(img, male)
                #datum=make_datum(img,male)
                count += 1
                lmdb_txn.put('{:0>8d}'.format(count), datum.SerializeToString())
                print '{:0>8d}'.format(count)
    lmdb_env.close()
    '''
    datum = caffe_pb2.Datum()#caffe中经常采用datum这种数据结构存储数据

    data = cv2.imread(image_name)
    label = image_label#设置图像的label

    datum = caffe.io.array_to_datum(data, label)
    keystr = '{:0>8d}'.format(label)
    lmdb_txn.put(keystr, datum.SerializeToString())

    lmdb_txn.commit()
    '''
def main():
    #label_path = "E:\\face_attribute_recognition_yqy\\attribution.txt"
    #image_path = "E:\\face_attribute_recognition_yqy\\1\\"
    '''
    label_path = "E:\\face_attribute_recognition_yqy\\celebatrainattr1.txt"
    image_path = "E:\\face_attribute_recognition_yqy\\train1\\"
    mdb_file = 'celeba_train_lmdb1'
    '''
    label_path = "E:\\face_attribute_recognition_yqy\\celebatestattr1.txt"
    image_path = "E:\\face_attribute_recognition_yqy\\test1\\"
    lmdb_file = 'celeba_test_lmdb2'
    gen_lmdb(lmdb_file,label_path,image_path)
    print ("done")
#
if __name__ == '__main__':
    main()

