#!/usr/bin/env python
# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
from xlwt import *

###wiki和celeba年龄属性图像测测试集数目统计并写入excel###
def attrcollect(label_path):
    countage=np.zeros(shape=(100,1))
    with open(label_path) as labels:
        for (num, attr) in enumerate(labels):
            if num>=350000:
                    attr=attr.split('\n')[0]
                    age = attr.split(' ')[2]
                    age = age.split('.')[0]
                    age = int(age)
                    for i in range(100):
                        if age==i+1:
                            countage[i]+=1
    labels.close()
    print(countage)
    return countage

if __name__ == '__main__':
    label_path="E:/img_crop_all/shufwikiandcelera.txt"
    countage=attrcollect(label_path)
    file=Workbook(encoding='utf-8')
    table=file.add_sheet('attrage')
    for i in range(0, 100):
        table.write(i, 0, str(i+1))
        table.write(i, 1, int(countage[i][0]))
    file.save('E:/face_attribute_recognition_yqy/attr.xls')



