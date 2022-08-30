#!/usr/bin/env python
# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L,params as P
import numpy as np
from pylab import  *




def face_attr_recog(lmdb,batch_size):

    n=caffe.NetSpec()#n是获取Caffe的一个Net，我们只需不断的填充这个n，最后面把n输出到文件就会使我们在Caffe学习2里面看到的Net的protobuf的定义
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,transform_param=dict(scale=1./255),ntop=2)
    n.conv1=L.Convolution(n.data,kernel_size=5,num_output=20,weight_filler=dict(type='xavier'))
    n.pool1=L.Pooling(n.conv1,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.conv2=L.Convolution(n.pool1,kernel_size=5,num_output=50,weight_filler=dict(type='xavier'))
    n.pool2=L.Pooling(n.conv2,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.fc1=L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.relu1=L.ReLU(n.fc1,in_place=True)
    n.score=L.InnerProduct(n.relu1,num_output=2,weight_filler=dict(type='xavier'))
    n.loss=L.SoftmaxWithLoss(n.score,n.label)
    return n.to_proto()


def main():
    '''
    with open('E:/face_attribute_recognition_yqy/face_attr_recog_train.prototxt', 'w') as f:
        f.write(str(face_attr_recog('E:/face_attribute_recognition_yqy/celeba_train_lmdb1', 64)))
    with open('E:/face_attribute_recognition_yqy/face_attr_recog_test.prototxt', 'w') as f:
        f.write(str(face_attr_recog('E:/face_attribute_recognition_yqy/celeba_test_lmdb1', 10)))
    '''
    '''

    with open('D:/caffe-windows-master_pc3/examples/mnist/lenet_auto_train.prototxt', 'w') as f:
        f.write(str(face_attr_recog('D:/caffe-windows-master_pc3/examples/mnist/mnist_train_lmdb', 64)))

    with open('D:/caffe-windows-master_pc3/examples/mnist/lenet_auto_test.prototxt', 'w') as f:
        f.write(str(face_attr_recog('D:/caffe-windows-master_pc3/examples/mnist/mnist_test_lmdb', 100)))
    '''

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=None

    solver=caffe.SGDSolver('E:/face_attribute_recognition_yqy/face_attr_recog_solver.prototxt')
    #solver = caffe.SGDSolver('D:/caffe-windows-master_pc3/examples/mnist/lenet_auto_solver.prototxt')
    print([(k, v.data.shape) for k, v in solver.net.blobs.items()])

    solver.net.forward()
    solver.test_nets[0].forward()
    solver.step(1)

    niter=100
    test_interval=25
    train_loss=np.zeros(niter)
    test_acc=np.zeros(int(np.ceil(niter/test_interval)))
    #output=np.zeros((niter,8,2))

    for it in range(niter):
        solver.step(1)
        train_loss[it]=solver.net.blobs['loss'].data
        # solver.test_nets[0].forward(start='conv1')
        # output[it]=solver.test_nets[0].blobs['score'].data[:8]
        if it%test_interval==0:
            #print(train_loss)
            print 'Iteration',it,'testing'
            test_acc[it // test_interval] = solver.test_nets[0].blobs['accuracy'].data
            #print test_acc
            '''
            correct=0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct+=sum(solver.test_nets[0].blobs['score'].data.argmax(1)==solver.test_nets[0].blobs['label'].data)
            test_acc[it//test_interval]=correct/1e3
            '''

    _,ax1=subplots()
    ax2=ax1.twinx()
    ax1.plot(arange(niter),train_loss)
    ax2.plot(test_interval*arange(len(test_acc)),test_acc,'r')
    ax1.set_xlabel('iteration')
    ax2.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Test Accuracy:{:.2f}'.format(test_acc[-1]))
    plt.show()

if __name__=="__main__":
    main()
