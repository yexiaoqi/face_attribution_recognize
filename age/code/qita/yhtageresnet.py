#!/usr/bin/env python
# -*- coding: utf-8 -*-
#by yht
from pylab import *
import caffe

caffe_model_path = 'E:/face_attribute_recognition_yqy/mobilenet_v2.caffemodel'
solver_config_path = 'E:/face_attribute_recognition_yqy/age_solver_180817.prototxt'
solver = caffe.get_solver(solver_config_path)

### solve
niter = 1000000  # EDIT HERE increase to train for longer
test_interval = niter / 1000
# test_interval=10
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
caffe.set_device(0)
caffe.set_mode_gpu()
# solver.net = caffe.Net(train_net, caffe_model_path, caffe.TRAIN)
solver.net.copy_from(caffe_model_path)
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    # store the train loss
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        #        print solver.net.blobs['fc1'].data
        #        print solver.net.blobs['labels'].data
        age = np.zeros(18)
        for j in range(18):
            for i in range(100):
                if solver.net.blobs['fc200'].data[j, 2 * i] < solver.net.blobs['fc200'].data[j, 2 * i + 1]:
                    age[j] += 1
        MAE=0
        print ("age=", age)
        print ("label",solver.net.blobs['labels'].data[:,0,0].T)
        for i in range(18):
            MAE+=abs(solver.net.blobs['labels'].data[i,0,0]-age[i])
        MAE = MAE / 18
        print 'train:'
        print ("mae=",MAE)
        #        print solver.net.blobs['fc8_gender'].data
#        print train_loss
        print 'Iteration', it, 'testing...'
        age = np.zeros(1)
        for j in range(1):
            for i in range(100):
                if solver.test_nets[0].blobs['fc200'].data[j, 2 * i] < solver.test_nets[0].blobs['fc200'].data[j, 2 * i + 1]:
                    age[j] += 1
        MAE=0
        print ("age=",age)
        print solver.test_nets[0].blobs['labels'].data[:,0,0].T
        for i in range(1):
            MAE+=abs(solver.test_nets[0].blobs['labels'].data[i,0,0]-age[i])
        MAE = MAE / 1
        print 'test:'
        print ("mae=", MAE)
#        test_acc[it // test_interval] = solver.test_nets[0].blobs['loss'].data
#        print test_acc

#        print solver.net.blobs['fc1000'].data
#        print solver.net.blobs['fc1000'].data.argmax(1)
#        print solver.net.blobs['label'].data

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
show()
