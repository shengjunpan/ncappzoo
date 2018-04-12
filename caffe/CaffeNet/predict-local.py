import os
import sys
import re
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def debug_array(var, name):
    print('{}: type={}, dtype={}, shape={}'.format(name, type(var), var.dtype, var.shape))

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    global TRANSFORMER
    img = TRANSFORMER.preprocess('data', img)
    
    return img

test_imge_paths = sys.argv[1:]

print('Reading mean image')
mean_blob = caffe_pb2.BlobProto()
with open('model_data/input/mean.binaryproto', 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

print('Reading model architecture and weights')
net = caffe.Net('deploy.prototxt',
                'model_data/snapshots/caffenet_iter_5000.caffemodel',
                caffe.TEST)

print('Define image transformers')
TRANSFORMER = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
TRANSFORMER.set_transpose('data', (2,0,1))
TRANSFORMER.set_mean('data', mean_array)

print('Reading image paths')
for img_path in test_imge_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    net.blobs['data'].data[...] = transform_img(
        img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    out = net.forward()
    pred_probas = out['prob']

    prob = pred_probas.max()
    label = 'cat' if pred_probas.argmax() == 0 else 'dog'
    cv2.rectangle(img, (0,0), (100,20), (255,255,255), cv2.FILLED)
    cv2.putText(img=img,
                text='{}: {:.2f}%'.format(label, prob*100),
                org=(0,15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0,0,255),
                thickness=2)

    cv_window_name = os.path.basename(img_path)
    cv2.imshow(cv_window_name, img)
    print('Hit any key to continue. \033[31;1mDo NOT close window!\033[0m')
    cv2.waitKey(0)
    cv2.destroyWindow(cv_window_name)
