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
    # 生成标签lmdb
    count = 0
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1.5e8))  # map_size指的是数据库的最大容量，根据需求设置
    # lmdb_txn = lmdb_env.begin(write=True)
    with lmdb_env.begin(write=True) as lmdb_txn:
        with open(label_path) as labels:
            for (num,imgpath) in enumerate(labels):
               # print("num=",num,"line=",imgpath)
            #for imgpath in labels.readlines():
                # target_label = np.zeros((3, 1, 1))
                target_label = np.zeros((1, 1, 1))
                imgpath = imgpath.split('\n')[0]
                imgname = imgpath.split(' ')[0]
                testname = eval(imgname.split('.')[0])
                if num>=380000:
                #if testname > 0:
                    '''
                    male=imgpath.split(' ')[1]
                    male=int(male)
                    target_label[0, 0, 0]=male
                    wearglasses=imgpath.split(' ')[18]
                    wearglasses=int(wearglasses)
                    target_label[1, 0, 0]=wearglasses
                    smiling = imgpath.split(' ')[33]
                    smiling = int(smiling)
                    target_label[2, 0, 0] = smiling
                    '''
                    age = imgpath.split(' ')[2]
                    age = age.split('.')[0]
                    age = int(age)
                    target_label[0, 0, 0] = age
                    # print(imgname)
                    # img=cv2.imread(image_path+imgname)
                    # datum = make_datum(img, male)
                    datum = caffe.io.array_to_datum(target_label)
                    # datum = make_datum( male,wearglasses)
                    # datum=make_datum(img,male)
                    count += 1
                    lmdb_txn.put('{:0>8d}'.format(count), datum.SerializeToString())
                    print '{:0>8d}'.format(count)
                # if imgname == "50000.jpg":  # 选择哪些图像
                #     break
    lmdb_env.close()
    print ('train labels are done!')


    # #生成标签lmdb
    # count=0
    # lmdb_env = lmdb.open(lmdb_file, map_size=int(5e7))#map_size指的是数据库的最大容量，根据需求设置
    # #lmdb_txn = lmdb_env.begin(write=True)
    # with lmdb_env.begin(write=True) as lmdb_txn:
    #     with open(label_path) as labels:
    #         for imgpath in labels.readlines():
    #             #target_label = np.zeros((3, 1, 1))
    #             target_label = np.zeros((1, 1, 1))
    #             imgpath=imgpath.split('\n')[0]
    #             imgname=imgpath.split(' ')[0]
    #             testname=eval(imgname.split('.')[0])
    #             if testname >50000:
    #                 '''
    #                 male=imgpath.split(' ')[1]
    #                 male=int(male)
    #                 target_label[0, 0, 0]=male
    #                 wearglasses=imgpath.split(' ')[18]
    #                 wearglasses=int(wearglasses)
    #                 target_label[1, 0, 0]=wearglasses
    #                 smiling = imgpath.split(' ')[33]
    #                 smiling = int(smiling)
    #                 target_label[2, 0, 0] = smiling
    #                 '''
    #                 age = imgpath.split(' ')[2]
    #                 age=age.split('.')[0]
    #                 age = int(age)
    #                 target_label[0, 0, 0] = age
    #                 #print(imgname)
    #                 #img=cv2.imread(image_path+imgname)
    #                 #datum = make_datum(img, male)
    #                 datum = caffe.io.array_to_datum(target_label)
    #                 #datum = make_datum( male,wearglasses)
    #                 #datum=make_datum(img,male)
    #                 count += 1
    #                 lmdb_txn.put('{:0>8d}'.format(count), datum.SerializeToString())
    #                 print '{:0>8d}'.format(count)
    #             if imgname=="60000.jpg":#选择哪些图像
    #                 break
    # lmdb_env.close()
    # print ('train labels are done!')

    #生成图像lmdb
    count2=0
    lmdb_env2 = lmdb.open(lmdb_file, map_size=int(7e10))
    lmdb_txn2=lmdb_env2.begin(write=True)
    #with lmdb_env2.begin(write=True) as lmdb_txn2:
    with open(label_path) as labels:
        for (num, imgpath) in enumerate(labels):
        #for imgpath in labels.readlines():
            imgpath=imgpath.split('\n')[0]
            imgname=imgpath.split(' ')[0]
            testname = eval(imgname.split('.')[0])
            if num<350000:
            #if testname >50000:
                #print(imgname)
                img=cv2.imread(image_path+imgname)
                img=np.array(img)
                img=img[:,:,::-1]#把im的RGB调整为BGR
                img=img.transpose((2,0,1))#把height*width*channel调整为channel*height*width
                datum=caffe.io.array_to_datum(img)
                #datum = make_datum(img)
                count2 += 1
                lmdb_txn2.put('{:0>8d}'.format(count2), datum.SerializeToString())
                print '{:0>8d}'.format(count2)
                if count2%50000==0:
                    lmdb_txn2.commit()
                    lmdb_txn2=lmdb_env2.begin(write=True)
            # if imgname == "60000.jpg":  # 选择哪些图像
            #     break
    lmdb_env2.close()
    print ('train data(images) are done!')



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
    label_path = "E:/img_crop_all/shufwikiandcelera.txt"
    image_path = "E:/img_crop_all/wikiandcelera180822/"
    #lmdb_file = 'lmdb_test_age_lmdb180815'
    lmdb_file = 'lmdb_train_img_lmdb180823'
    gen_lmdb(lmdb_file,label_path,image_path)
    #print ("done")
#
if __name__ == '__main__':
    main()

