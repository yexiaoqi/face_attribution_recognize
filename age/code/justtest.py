#!/usr/bin/env python
# -*- coding: utf-8 -*-

#用训练完的caffemodel计算测试集年龄的mae
import caffe
from caffe import layers as L,params as P
import numpy as np
from pylab import  *
import os

def main():

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=None

    #solver=caffe.SGDSolver('E:/face_attribute_recognition_yqy/face_attr_recog_solver.prototxt')#性别，是否戴眼镜，是否微笑三个属性
    #solver = caffe.SGDSolver('E:/face_attribute_recognition_yqy/agetest/face_age_recog_solver.prototxt')#年龄属性
    #solver = caffe.SGDSolver('D:/caffe-windows-master_pc3/examples/mnist/lenet_auto_solver.prototxt')
    solver = caffe.get_solver('E:/face_attribute_recognition_yqy/age_solver_180817.prototxt')

    #solver.net.copy_from('E:/face_attribute_recognition_yqy/resnet-18.caffemodel')#如果有预训练模型则加上这句
    #solver.net.copy_from('E:/face_attribute_recognition_yqy/mobilenet_v2.caffemodel')
    solver.net.copy_from('E:/face_attribute_recognition_yqy/face_attr_recog/age0817/age0817_iter_500000.caffemodel')
    #solver.restore('E:/face_attribute_recognition_yqy/face_attr_recog/age/age_iter_120000.solverstate')


    niter=500
    test_loss=np.zeros(niter)
    mae_all=0
    count=0
    #solver.test_nets[0].forward(start='data')
    solver.step(1)
    for it in range(niter):
        print 'Iteration',it,'testing'
        solver.test_nets[0].forward()
        test_loss[it] = solver.test_nets[0].blobs['loss'].data
        age=np.zeros(20)#根据batchsize修改
        mae = 0
        for batch in range(0,20):#根据batchsize修改
            for i in range(0,100):
                if solver.test_nets[0].blobs['fc200'].data[batch,2*i]<solver.test_nets[0].blobs['fc200'].data[batch,2*i+1]:
                    age[batch]+=1
            mae += abs(solver.test_nets[0].blobs['labels'].data[batch, 0, 0, 0].T - age[batch])
        mae=mae/20#根据batchsize修改
        print("label=", solver.test_nets[0].blobs['labels'].data[:,0, 0, 0].T)
        print("age=",age)
        mae_all=(mae_all*count+mae)/(count+1)
        print("mae_all=", mae_all)
        count+=1
        age = np.zeros(20)#根据batchsize修改
        mae = 0
        #print("output=", solver.test_nets[0].blobs['fc_output'].data)

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), test_loss)
    #ax2.plot(test_interval * arange(len(mae_all)), mae_all, 'r')
    ax1.set_xlabel('iteration')
    ax2.set_ylabel('train loss')
    #ax2.set_ylabel('test mae')
    #ax2.set_title('Test Accuracy:{:.2f}'.format(mae_all[-1]))
    plt.show()

if __name__=="__main__":
    main()
