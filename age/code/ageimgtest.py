#!/usr/bin/env python
# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L,params as P
import numpy as np
from pylab import  *
import os
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
#ageimgtest是实时测试人脸年龄的代码

def bbreg(boundingbox, reg):
    reg = reg.T

    # calibrate bouding boxes
    if reg.shape[1] == 1:
        #print("reshape of reg")
        pass  # reshape of reg
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1

    bb0 = boundingbox[:, 0] + reg[:, 0] * w
    bb1 = boundingbox[:, 1] + reg[:, 1] * h
    bb2 = boundingbox[:, 2] + reg[:, 2] * w
    bb3 = boundingbox[:, 3] + reg[:, 3] * h
    #print(reg[:, 0])
    boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
    # print("bb", boundingbox)
    return boundingbox

'''
def pad(boxesA, w, h):
    boxes = boxesA.copy()  # shit, value parameter!!!
    # print('#################')
    # print('boxes', boxes)
    # print('w,h', w, h)

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    # print('tmph', tmph)
    # print('tmpw', tmpw)

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy - 1)
    dx = np.maximum(0, dx - 1)
    y = np.maximum(0, y - 1)
    x = np.maximum(0, x - 1)
    edy = np.maximum(0, edy - 1)
    edx = np.maximum(0, edx - 1)
    ey = np.maximum(0, ey - 1)
    ex = np.maximum(0, ex - 1)

    # print("dy"  ,dy )
    # print("dx"  ,dx )
    # print("y "  ,y )
    # print("x "  ,x )
    # print("edy" ,edy)
    # print("edx" ,edx)
    # print("ey"  ,ey )
    # print("ex"  ,ex )

    # print('boxes', boxes)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
'''

def pad(bboxes, w, h):
    """
        pad the the bboxes, alse restrict the size of it
    Parameters:
    ----------
        bboxes: numpy array, n x 5
            input bboxes
        w: float number
            width of the input image
        h: float number
            height of the input image
    Returns :
    ------s
        dy, dx : numpy array, n x 1
            start point of the bbox in target image
        edy, edx : numpy array, n x 1
            end point of the bbox in target image
        y, x : numpy array, n x 1
            start point of the bbox in original image
        ex, ex : numpy array, n x 1
            end point of the bbox in original image
        tmph, tmpw: numpy array, n x 1
            height and width of the bbox
    """
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    # print('bboxA', bboxA)
    # print('w', w)
    # print('h', h)
    # print('l', l)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())  # read s using I

    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0, :, :].T
    dy1 = reg[1, :, :].T
    dx2 = reg[2, :, :].T
    dy2 = reg[3, :, :].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x

    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet

        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
    # print("dx1.shape", dx1.shape)
    # print('map.shape', map.shape)

    score = map[x, y]
    reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T  # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T  # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    # print('(x,y)',x,y)
    # print('score', score)
    # print('reg', reg)

    return boundingbox_out.T

'''
# change by yqy
def drawBoxes(im, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2
    x4 = x1 - (x3 - x1) / 3
    y4 = y1 - (y3 - y1) / 3
    x5 = x2 + (x2 - x3) / 3
    y5 = y2 + (y2 - y3) / 3
    # cv2.circle(im, (int(boxes[0,2]), int(boxes[0,3])), 1, (0, 0, 255), 2)
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x4[i]), int(y4[i])), (int(x5[i]), int(y5[i])), (0, 255, 0), 1)
        # cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
        # cv2.circle(im, (int(x1[i]), int(y1[i])), 1, (0, 0, 255), 2)
    return im


# change end
'''
# note by yqy

def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    #for i in range(x1.shape[0]):
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im



# note end

# add by yqy
def drawLanmarks(im, points):
    # p=points[0,:]
    for p in points:
        for i in range(5):
            cv2.circle(im, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
    return im


# add end

# from time import time
#
# _tstart_stack = []
#
#
# def tic():
#     _tstart_stack.append(time())
#
#
# def toc(fmt="Elapsed: %s s"):
#     print(fmt % (time() - _tstart_stack.pop()))


# def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):#note by yqy
def detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, fastresize, factor):  # add by yqy

    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0, 9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0 / minsize
    minl = minl * m

    # total_boxes = np.load('total_boxes.npy')
    # total_boxes = np.load('total_boxes_242.npy')
    # total_boxes = np.load('total_boxes_101.npy')

    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))

        if fastresize:
            im_data = (img - 127.5) * 0.0078125  # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws, hs))  # default is bilinear
        else:
            im_data = cv2.resize(img, (ws, hs))  # default is bilinear
            im_data = (im_data - 127.5) * 0.0078125  # [0,255] -> [-1,1]
        # im_data = imResample(img, hs, ws); print("scale:", scale)

        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype=np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()

        boxes = generateBoundingBox(out['prob1'][0, 1, :, :], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            # print(boxes[4:9])
            # print('im_data', im_data[0:5, 0:5, 0], '\n')
            # print('prob1', out['prob1'][0,0,0:3,0:3])

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0:
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)

    # np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    #print("[1]:", total_boxes.shape[0])
    # print(total_boxes)
    # return total_boxes, []

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        #print("[2]:", total_boxes.shape[0])

        # revise and convert to square
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        t5 = total_boxes[:, 4]
        total_boxes = np.array([t1, t2, t3, t4, t5]).T
        # print("[3]:",total_boxes.shape[0])
        # print(regh)
        # print(regw)
        # print('t1',t1)
        # print(total_boxes)

        total_boxes = rerec(total_boxes)  # convert box to square
        #print("[4]:", total_boxes.shape[0])

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
        #print("[4.5]:", total_boxes.shape[0])
        # print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    # print(total_boxes.shape)
    # print(total_boxes)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        # print('tmph', tmph)
        # print('tmpw', tmpw)
        # print("y,ey,x,ex", y, ey, x, ex, )
        # print("edy", edy)

        # tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3))  # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) + 1, int(tmpw[k]) + 1, 3))

            # print("dx[k], edx[k]:", dx[k], edx[k])
            # print("dy[k], edy[k]:", dy[k], edy[k])
            # print("img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape)
            # print("tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape)

            tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = img[int(y[k]):int(ey[k]) + 1,
                                                                          int(x[k]):int(ex[k]) + 1]
            # print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            # print("tmp", tmp.shape)

            tempimg[k, :, :, :] = cv2.resize(tmp, (24, 24))
            # tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            # print('tempimg', tempimg[k,:,:,:].shape)
            # print(tempimg[k,0:5,0:5,0] )
            # print(tempimg[k,0:5,0:5,1] )
            # print(tempimg[k,0:5,0:5,2] )
            # print(k)

        # print(tempimg.shape)
        # print(tempimg[0,0,0,:])
        tempimg = (tempimg - 127.5) * 0.0078125  # done in imResample function wrapped by python

        # np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        # print(tempimg[0,:,0,0])

        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        # print(out['conv5-2'].shape)
        # print(out['prob1'].shape)

        score = out['prob1'][:, 1]
        # print('score', score)
        pass_t = np.where(score > threshold[1])[0]
        # print('pass_t', pass_t)

        score = np.array([score[pass_t]]).T
        total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
        #print("[5]:", total_boxes.shape[0])
        # print(total_boxes)

        # print("1.5:",total_boxes.shape)

        mv = out['conv5-2'][pass_t, :].T
        # print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            # print('pick', pick)
            if len(pick) > 0:
                total_boxes = total_boxes[pick, :]
                #print("[6]:", total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                #print("[7]:", total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                #print("[8]:", total_boxes.shape[0])

        #####
        # 2 #
        #####
        #print("2:", total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage

            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

            # print('tmpw', tmpw)
            # print('tmph', tmph)
            # print('y ', y)
            # print('ey', ey)
            # print('x ', x)
            # print('ex', ex)

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = img[int(y[k]):int(ey[k]) + 1,
                                                                              int(x[k]):int(ex[k]) + 1]
                tempimg[k, :, :, :] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg - 127.5) * 0.0078125  # [0,255] -> [-1,1]

            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()

            score = out['prob1'][:, 1]
            points = out['conv6-3']
            pass_t = np.where(score > threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
            #print("[9]:", total_boxes.shape[0])

            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:, 3] - total_boxes[:, 1] + 1
            h = total_boxes[:, 2] - total_boxes[:, 0] + 1

            points[:, 0:5] = np.tile(w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:, 0], (5, 1)).T - 1
            points[:, 5:10] = np.tile(h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:, 1], (5, 1)).T - 1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:, :])
                #print("[10]:", total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')

                # print(pick)
                if len(pick) > 0:
                    total_boxes = total_boxes[pick, :]
                    #print("[11]:", total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    #print("3:", total_boxes.shape)


    num_box = total_boxes.shape[0]
    if num_box > 0:
        num_box = total_boxes.shape[0]
        tempimg = np.zeros((num_box, 15, 24, 24))
        patchw = np.maximum(total_boxes[:, 2] - total_boxes[:, 0] + 1, total_boxes[:, 3] - total_boxes[:, 1] + 1)
        patchw = np.round(patchw * 0.25)
        # patchw = np.round(patchw * 0.5)
        patchw[np.where(np.mod(patchw, 2) == 1)] += 1  # 这句没有什么影响
        #print(points)
        for i in range(5):
            x, y = points[:, i], points[:, i + 5]
            x, y = np.round(x - 0.5 * patchw), np.round(y - 0.5 * patchw)
            # [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]=pad(np.concatenate((x,y,x+patchw,y+patchw),axis=0),w,h)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(np.concatenate(
                (np.transpose([x]), np.transpose([y]), np.transpose([x + patchw]), np.transpose([y + patchw])),
                axis=1), img.shape[1], img.shape[0])
            # [dy,edy,dx,edx,y,ey,x,ex,tmpw,tmph]=pad([x.tolist(),y.tolist(),(x+patchw).tolist(),(y+patchw).tolist()],w,h)
            for j in range(num_box):
                # tmpim=np.zeros((tmpw[j],tmpw[j],3),dtype=np.float32)
                tmpim = np.zeros((int(tmpw[j]), int(tmpw[j]), 3))
                tmpim[int(dy[j]):int(edy[j]), int(dx[j]):int(edx[j]), :] = img[int(y[j]):int(ey[j]), int(x[j]):int(ex[j]),
                                                                           :]
                # tempimg[ :, :, i * 3+1:i * 3 + 3,j] = cv2.resize(tmpim, (24, 24))
                tempimg[j, i * 3:i * 3 + 3, :, :] = np.swapaxes(cv2.resize(tmpim, (24, 24)), 0, 2)
        LNet.blobs['data'].reshape(num_box, 15, 24, 24)
        tempimg = (tempimg - 127.5) * 0.0078125

        LNet.blobs['data'].data[...] = tempimg
        out = LNet.forward()
        # score = out['fc5_3'][1,:]
        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))
        for k in range(5):
            tmp_index = np.where(np.abs(out['fc5_' + str(k + 1)] - 0.5) > 0.35)
            out['fc5_' + str(k + 1)][tmp_index[0]] = 0.5
            pointx[:, k] = np.round(points[:, k] - 0.5 * patchw) + out['fc5_' + str(k + 1)][:, 0] * patchw
            pointy[:, k] = np.round(points[:, k + 5] - 0.5 * patchw) + out['fc5_' + str(k + 1)][:, 1] * patchw
            #print("yqy")
            #print(out['fc5_' + str(k + 1)][0, :])
            # pointx[:, k] = np.round(points[:, k] - 0.5 * patchw) + out['fc5_'+str(k+1)][:, 0] * patchw
            # pointy[:, k] = np.round(points[:, k + 5] - 0.5 * patchw) + out['fc5_'+str(k+1)][:, 1] * patchw
            # out[k][tmp_index[0]] = 0.5
            # pointx[:,k]=np.round(points[:,k]-0.5*patchw)+out[k][:,0]*patchw
            # pointy[:,k]=np.round(points[:,k+5]-0.5*patchw)+out[k][:,1]*patchw
        for j in range(num_box):
            points = np.hstack([pointx, pointy])
            points = points.astype((np.int32))
        # add end


    return total_boxes, points


def initFaceDetector():
    minsize = 20
    caffe_model_path = "/home/duino/iactive/mtcnn/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)
    LNet = caffe.Net(caffe_model_path + "/det4.prototxt", caffe_model_path + "/det4.caffemodel",
                     caffe.TEST)  # add by yqy
    # return (minsize, PNet, RNet, ONet, threshold, factor)#note by yqy
    return (minsize, PNet, RNet, ONet, LNet, threshold, factor)  # add by yqy


def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    # add by yqy
    LNet = facedetector[4]
    threshold = facedetector[5]
    factor = facedetector[6]
    # add end
    # threshold = facedetector[4]
    # factor = facedetector[5]

    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:, :, 2].copy()
    img_matlab[:, :, 2] = img_matlab[:, :, 0]
    img_matlab[:, :, 0] = tmp

    # tic()
    # boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)#note by yqy
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, LNet, threshold, False,
                                        factor)  # add by yqy
    # toc()
    containFace = (True, False)[boundingboxes.shape[0] == 0]
    return containFace, boundingboxes

#
# # add by yqy   旋转图像使得眼睛在一条水平线上
# def warp_affine(image, points, scale=1.0):
#     eye_center = ((points[0, 0] + points[0, 1]) / 2, (points[0, 5] + points[0, 6]) / 2)
#     dy = points[0, 6] - points[0, 5]
#     dx = points[0, 1] - points[0, 0]
#     # 计算旋转角度
#     angle = cv2.fastAtan2(dy, dx)
#     rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
#     rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))
#     # cv2.imshow(rot_img)
#     return rot_img
#
#
# # add end

# add by yqy   旋转图像使得眼睛在一条水平线上
def warp_affine(image, points, scale=1.0):
    eye_center = ((points[0] + points[1]) / 2, (points[ 5] + points[6]) / 2)
    dy = points[6] - points[5]
    dx = points[1] - points[0]
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
    rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))
    # cv2.imshow(rot_img)
    return rot_img


# add end
if __name__=="__main__":

    minsize = 20
    caffe_model_path = "E:/mtcnn-master1807262/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)
    LNet = caffe.Net(caffe_model_path + "/det4.prototxt", caffe_model_path + "/det4.caffemodel", caffe.TEST)

   #mobilenet
    model_def = 'E:/face_attribute_recognition_yqy/180822_age_mobilenet_v2_deploy.prototxt'
    #model_weights = 'E:/face_attribute_recognition_yqy/face_attr_recog/age0817/mobilenet_v2_iter_500000.caffemodel'
    model_weights = 'E:/face_attribute_recognition_yqy/face_attr_recog/age0817/age0817_iter_500000.caffemodel'
    #resnet
    # model_def = 'E:/face_attribute_recognition_yqy/resnet18_deploy1.prototxt'
    # model_weights = 'E:/face_attribute_recognition_yqy/weights.scratch.caffemodel'
    #model_weights = 'E:/face_attribute_recognition_yqy/yht_net_iter_2000.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    mean_filename = 'E:/face_attribute_recognition_yqy/yht/train_lmdb_mean.binaryproto'
    #mean_filename = 'E:/face_attribute_recognition_yqy/train_mean.binaryproto'
    bin_mean = open(mean_filename, 'rb').read()
    blob = caffe.io.caffe_pb2.BlobProto()
    blob.ParseFromString(bin_mean)
    mean = caffe.io.blobproto_to_array(blob)[0].mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean)
    # transformer.set_raw_scale('data',0.00392156862745)
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(1, 3, 224, 224)
    #cap = cv2.VideoCapture('E:/face_attribute_recognition_yqy/realtime1.gif')#输入视频
    cap = cv2.VideoCapture(0)#输入图片则注释这句
    #
    # while (1):
    #     # get a frame
    #     ret, img = cap.read()#输入图片则注释这句
    #     start = time.clock()
    #     # show a frame
    #     #cv2.imshow("capture", img)
    #     # starttime = datetime.datetime.now()
    #     #img = cv2.imread('E:/face_attribute_recognition_yqy/1.jpg')#输入图片则不注释这句
    #     img_matlab = img.copy()
    #     raw_img = img.copy()
    #     tmp = img_matlab[:, :, 2].copy()
    #     img_matlab[:, :, 2] = img_matlab[:, :, 0]
    #     img_matlab[:, :, 0] = tmp
    #     #starttime = datetime.datetime.now()
    #     boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, LNet, threshold, False, factor)
    #     output = net.forward()
    #     #start = time.clock()
    #
    #     end = time.clock()
    #     raw_boundingboxes = boundingboxes.copy()
    #     for faces in range(boundingboxes.shape[0]):
    #         img = warp_affine(raw_img, points[faces])
    #
    #         bbx_scale = 0.8
    #         width = max(boundingboxes[faces, 2] - boundingboxes[faces, 0],
    #                     boundingboxes[faces, 3] - boundingboxes[faces, 1])
    #         center_x = (boundingboxes[faces, 0] + boundingboxes[faces, 2]) / 2
    #         center_y = (boundingboxes[faces, 1] + boundingboxes[faces, 3]) / 2
    #         boundingboxes[faces, 0] = center_x - width * bbx_scale
    #         boundingboxes[faces, 1] = center_y - width * bbx_scale
    #         boundingboxes[faces, 2] = center_x + width * bbx_scale
    #         boundingboxes[faces, 3] = center_y + width * bbx_scale
    #
    #
    #         # boundingboxes[faces, 0] = boundingboxes[faces, 0] - (boundingboxes[faces, 2] - boundingboxes[faces, 0]) / 6
    #         # boundingboxes[faces, 1] = boundingboxes[faces, 1] - (boundingboxes[faces, 3] - boundingboxes[faces, 1]) / 6
    #         # boundingboxes[faces, 2] = boundingboxes[faces, 2] + (boundingboxes[faces, 2] - boundingboxes[faces, 0]) / 6
    #         # boundingboxes[faces, 3] = boundingboxes[faces, 3] + (boundingboxes[faces, 3] - boundingboxes[faces, 1]) / 6
    #
    #         [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(boundingboxes, img.shape[1], img.shape[0])
    #         #print boundingboxes
    #         boundingboxes[faces, 0] = x[faces]
    #         boundingboxes[faces, 1] = y[faces]
    #         boundingboxes[faces, 2] = ex[faces]
    #         boundingboxes[faces, 3] = ey[faces]
    #         face = img[int(boundingboxes[faces, 1]):int(boundingboxes[faces, 3]),
    #                int(boundingboxes[faces, 0]):int(boundingboxes[faces, 2])]
    #         face = cv2.resize(face, (224, 224))
    #         tranformed_image = transformer.preprocess('data', face)
    #         tranformed_image = tranformed_image * 0.00392156862745
    #         net.blobs['data'].data[...] = tranformed_image
    #         # labels_male = ['female', 'male']
    #         # labels_eyeglasses = ['without-eyeglasses', 'with-eyeglasses']
    #         # labels_smiling = ['not smiling', 'smiling']
    #
    #
    #
    #         #output = net.forward()
    #         end=time.clock()
    #         print("time cost:",end-start)
    #
    #         #print("output=",output['fc_output'][0].data)
    #         # print("output=", output['fc_output'][0,1])
    #         # print("output1=", output['fc_output'][0])
    #         age=0
    #         for i in range(0, 100):
    #             #if output['fc200'][0, 2 * i] < output['fc200'][ 0, 2 * i + 1]:
    #             if net.blobs['fc200'].data[0, 2 * i] < net.blobs['fc200'].data[0, 2 * i + 1]:
    #                 age+=1
    #
    #
    #
    #         # labels_male = ['女', '男']
    #         # labels_eyeglasses = ['不戴眼镜', '戴眼镜']
    #         # labels_smiling = ['不笑', '笑']
    #         # endtime = datetime.datetime.now()
    #         # print ((endtime - starttime).microseconds, "us")
    #
    #
    #         # output_prob_male = output['loss_male'][0]
    #         # output_prob_eyeglasses = output['loss_eyeglasses'][0]
    #         # output_prob_smiling = output['loss_smiling'][0]
    #
    #
    #         # print (output_prob_male, output_prob_eyeglasses, output_prob_smiling)
    #         # print(labels_male[output_prob_male.argmax()], labels_eyeglasses[output_prob_eyeglasses.argmax()],
    #         #       labels_smiling[output_prob_smiling.argmax()])
    #         text=str(age)
    #         #text = labels_male[output_prob_male.argmax()]+":"+str(max(output_prob_male[0],output_prob_male[1]))+'\n'+labels_eyeglasses[output_prob_eyeglasses.argmax()]+":"+str(max(output_prob_eyeglasses[0],output_prob_eyeglasses[1]))+'\n'+labels_smiling[output_prob_smiling.argmax()]+":"+str(max(output_prob_smiling[0],output_prob_smiling[1]))
    #         #text = labels_male[output_prob_male.argmax()]+","+labels_eyeglasses[output_prob_eyeglasses.argmax()]+","+labels_smiling[output_prob_smiling.argmax()]
    #         #cv2.putText(img, text, (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #         #cv2.putText(img, text, (int(boundingboxes[:,0]), int(boundingboxes[:,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #         dy=12
    #         for i,text in enumerate(text.split('\n')):
    #             newy=int(raw_boundingboxes[faces,1]+i*dy)
    #             cv2.putText(raw_img, text, (int(raw_boundingboxes[faces, 0]), newy+5),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 输出英文
    #
    #         #cv2.putText(img, text, (int(raw_boundingboxes[faces,0]),int(raw_boundingboxes[faces,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)#输出英文
    #         # endtime = datetime.datetime.now()
    #         # print ((endtime - starttime).microseconds, "us")
    #
    #         #输出中文
    #         '''
    #         img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         font = ImageFont.truetype(
    #             'D:/anzhuangbao/anaconda/anzhuanghou/Lib/site-packages/noto-cjk-master/NotoSansCJK-Black.ttc', 10)
    #         position = (int(raw_boundingboxes[faces, 0]), int(raw_boundingboxes[faces, 1]))
    #         # position = (100, 100)
    #         fillColor = (0, 0, 255)
    #         if not isinstance(str, unicode):
    #             text = text.decode('utf8')
    #         draw = ImageDraw.Draw(img_PIL)
    #         draw.text(position, text, font=font, fill=fillColor)
    #         img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    #         '''
    #     raw_img = drawLanmarks(raw_img, points)
    #     img = drawBoxes(raw_img, raw_boundingboxes)
    #     cv2.imshow("capture", img)#上面两句应该在循环外，否则几个人画几个框
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    while (1):
        #img = cv2.imread('E:/face_attribute_recognition_yqy/1.jpg')  # 输入图片则不注释这句
        # get a frame
        ret, img = cap.read()
        start = time.clock()
        raw_img = img.copy()
        print img.shape
        img_matlab = img.copy()
        tmp = img_matlab[:, :, 2].copy()
        img_matlab[:, :, 2] = img_matlab[:, :, 0]
        img_matlab[:, :, 0] = tmp

        # check rgb position
        # tic()
        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, LNet, threshold, False, factor)
        output = net.forward()
        end = time.clock()
        raw_boundingboxes = boundingboxes.copy()
        for faces in range(boundingboxes.shape[0]):
            img = warp_affine(raw_img, points[faces])

            bbx_scale = 0.8
            width = max(boundingboxes[faces, 2] - boundingboxes[faces, 0],
                        boundingboxes[faces, 3] - boundingboxes[faces, 1])
            center_x = (boundingboxes[faces, 0] + boundingboxes[faces, 2]) / 2
            center_y = (boundingboxes[faces, 1] + boundingboxes[faces, 3]) / 2
            boundingboxes[faces, 0] = center_x - width * bbx_scale
            boundingboxes[faces, 1] = center_y - width * bbx_scale
            boundingboxes[faces, 2] = center_x + width * bbx_scale
            boundingboxes[faces, 3] = center_y + width * bbx_scale

            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(boundingboxes, img.shape[1], img.shape[0])
            boundingboxes[faces, 0] = x[faces]
            boundingboxes[faces, 1] = y[faces]
            boundingboxes[faces, 2] = ex[faces]
            boundingboxes[faces, 3] = ey[faces]
            face = img[int(boundingboxes[faces, 1]):int(boundingboxes[faces, 3]),
                   int(boundingboxes[faces, 0]):int(boundingboxes[faces, 2])]
            face = cv2.resize(face, (224, 224))

            # Attribute inference#
            transformed_image = transformer.preprocess('data', face)
            transformed_image = transformed_image * 0.00392156862745
            net.blobs['data'].data[...] = transformed_image
            print end - start
            age = 0
            for i in range(100):
                if net.blobs['fc200'].data[0, 2 * i] < net.blobs['fc200'].data[0, 2 * i + 1]:
                    age += 1
            str_age = 'Age:'
            str_age += str(age)

            font = cv2.FONT_HERSHEY_SIMPLEX
            raw_img = cv2.putText(raw_img, str_age,
                                  (int(raw_boundingboxes[faces, 0]), int(raw_boundingboxes[faces, 1])),
                                  font, 0.5, (0, 255, 0), 2)
        raw_img = drawLanmarks(raw_img, points)
        img = drawBoxes(raw_img, raw_boundingboxes)
        cv2.imshow('img', raw_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    #cap.release()#输入图片则注释这句
    cv2.destroyAllWindows()