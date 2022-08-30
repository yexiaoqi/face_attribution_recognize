#!/usr/bin/env python
# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L,params as P
import numpy as np
from pylab import  *
import os

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

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=None

    #solver=caffe.SGDSolver('E:/face_attribute_recognition_yqy/face_attr_recog_solver.prototxt')#性别，是否戴眼镜，是否微笑三个属性
    solver = caffe.SGDSolver('E:/face_attribute_recognition_yqy/face_age_recog_solver.prototxt')#年龄属性
    #solver = caffe.SGDSolver('D:/caffe-windows-master_pc3/examples/mnist/lenet_auto_solver.prototxt')
    #print([(k, v.data.shape) for k, v in solver.net.blobs.items()])

    #solver.net.copy_from('E:/face_attribute_recognition_yqy/resnet-18.caffemodel')#如果有预训练模型则加上这句
    solver.net.copy_from('E:/face_attribute_recognition_yqy/mobilenet_v2.caffemodel')
    #solver.restore('E:/face_attribute_recognition_yqy/face_attr_recog/age/age_iter_120000.solverstate')
    # solver.net.forward()
    # solver.test_nets[0].forward()
    # solver.step(1)

    niter=1000
    test_interval=10
    train_loss=np.zeros(niter)
    test_acc=np.zeros(int(np.ceil(niter/test_interval)))
    #output=np.zeros((niter,8,2))

    # age = np.zeros(10)
    mae_all=0
    count=0
    for it in range(1,niter+1):
        #print("data=", solver.test_nets[0].blobs['data'].data)
        solver.step(1)
        train_loss[it-1]=solver.net.blobs['loss'].data
        # solver.test_nets[0].forward(start='conv1')
        # output[it]=solver.test_nets[0].blobs['score'].data[:8]

        # print("a",solver.test_nets[0].blobs['label'].data[:,0,0,0].T)
        # #print("y",it%test_interval)
        # for i in range(0, 100):
        #     if solver.test_nets[0].blobs['fc_output'].data[0, 2 * i] < solver.test_nets[0].blobs['fc_output'].data[
        #         0, 2 * i + 1]:
        #         age[it%test_interval-1] += 1
        # for j in range(10):
        #     mae += (solver.test_nets[0].blobs['label'].data[j, 0, 0, 0].T - age[it%test_interval-1])
        # #print("output=", solver.test_nets[0].blobs['fc_output'].data)
        #     print("label=", solver.test_nets[0].blobs['label'].data[j, 0, 0, 0].T)
        # print("age=", age)


        if it % test_interval == 0 :
       # if it%test_interval==0:
            #print(train_loss)
            print 'Iteration',it,'testing'
            #test_acc[it // test_interval] = solver.test_nets[0].blobs['accuracy'].data  #除了年龄以外的属性都可以直接用accuracy

            #print(type(solver.test_nets[0].blobs['fc_output'].data[0, 1]))


            #print solver.test_nets[0].blobs['label'].data  # add

            age=np.zeros(1)
            mae = 0
            for batch in range(0,1):
                for i in range(0,100):
                    if solver.test_nets[0].blobs['fc_output'].data[batch,2*i]<solver.test_nets[0].blobs['fc_output'].data[batch,2*i+1]:
                        age[batch]+=1
                #mae += abs(solver.test_nets[0].blobs['label'].data[batch, 0, 0, 0].T - age[batch])
                mae += abs(solver.test_nets[0].blobs['label'].data[batch, 0] - age[batch])

        #print("label=", solver.test_nets[0].blobs['label'].data)
            #print solver.test_nets[0].blobs['label'].data[:, 0, 0].T#add
            print("label=", solver.test_nets[0].blobs['label'].data[:,0, 0, 0].T)
            print("age=",age)
            mae_all=(mae_all*count+mae)/(count+1)
            print("mae_all=", mae_all)
            count+=1
            age = np.zeros(1)
            mae = 0

           # print("label=",solver.test_nets[0].blobs['label'].data[0,0,0,0])
            # print("label=", solver.test_nets[0].blobs['label'].data)
            #print("output=",solver.test_nets[0].blobs['fc_output'].data[0,1])
            #print("output=", solver.test_nets[0].blobs['fc_output'].data)
            #print test_acc
            '''
            correct=0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct+=sum(solver.test_nets[0].blobs['score'].data.argmax(1)==solver.test_nets[0].blobs['label'].data)
            test_acc[it//test_interval]=correct/1e3
            '''


    solvers = [ ('age0816', solver)]
    #loss,acc,weights=run_solvers(niter,solvers)
    # Save the learned weights from both nets.
    weight_dir = "E:/face_attribute_recognition_yqy"
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), train_loss)
    #ax2.plot(test_interval * arange(len(mae_all)), mae_all, 'r')
    ax1.set_xlabel('iteration')
    ax2.set_ylabel('train loss')
    #ax2.set_ylabel('test mae')
    #ax2.set_title('Test Accuracy:{:.2f}'.format(mae_all[-1]))
    plt.show()
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
    '''
if __name__=="__main__":
    main()
