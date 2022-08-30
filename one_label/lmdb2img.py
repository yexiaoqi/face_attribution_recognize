#!/usr/bin/env python
# -*- coding: utf-8 -*-
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2

def read_lmdb(lmdb_file):
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)

        cv2.imshow('cv2', data)
        cv2.waitKey(0)
        print('{},{}'.format(key, label))

def main():
    lmdb_file = 'celeba_test_lmdb1'
    read_lmdb(lmdb_file)

if __name__ == '__main__':
    main()

