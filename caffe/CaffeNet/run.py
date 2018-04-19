import os
import sys
import re
import glob
import cv2
import caffe
import lmdb
import numpy as np
import argparse
import pandas as pd

from caffe.proto import caffe_pb2
from mvnc import mvncapi as mvnc
from abc import ABC, abstractmethod

caffe.set_mode_gpu() 

_parser = argparse.ArgumentParser()
_parser.add_argument("-d", "--device", default='gpu',
                     help="choose device: gpu or ncs")
_parser.add_argument("-a", "--action", default='predict',
                     help="choose action: predict or validate")

_parser.add_argument("-m", "--mean-proto",
                     default='model_data/input/mean.binaryproto',
                     help="path to mean binary proto file")
_parser.add_argument("-P", "--prediction-path",
                     default=None,
                     help="path to save predicted probabilities")
_parser.add_argument("-c", "--caffemodel",
                     default='model_data/snapshots/caffenet_iter_5000.caffemodel',
                     help="path to caffe model file")
_parser.add_argument("-p", "--prototxt",
                     default='deploy.prototxt',
                     help="path to prototxt file")
_parser.add_argument("-g", "--graph",
                     default='graph',
                     help="path to compiled NCS graph file")
_parser.add_argument("-e", "--equalized",
                     action='store_true',
                     help="whether equalization histogram has been applied to the image")
_parser.add_argument("image", nargs="+",
                     help="image (predict) or lmdb (validate) paths")
ARGS = _parser.parse_args()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

class Predictor(ABC):
    """
    Base class for a predictor using a trained CaffeNet model
    """
    def __init__(self, input_width, input_height, mean_proto):
        self.input_width = input_width
        self.input_height = input_height

        mean_blob = caffe_pb2.BlobProto()
        with open(mean_proto, 'rb') as f:
            mean_blob.ParseFromString(f.read())
        input_shape = (mean_blob.channels, mean_blob.height, mean_blob.width)
        self.mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(input_shape).transpose((1,2,0)) # CHW -> HWC

        super().__init__()

    def transform_img(self, img, equalized):
        if not equalized:
            #Histogram Equalization
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
        #Image Resizing
        img = cv2.resize(img, (self.input_width, self.input_height), interpolation = cv2.INTER_CUBIC)

        img = img.astype(np.float32)
        img -= self.mean_array
        return img

    @abstractmethod
    def predict(self, img):
        pass

    
class LocalPredictor(Predictor):
    """
    Predictor using a trained CaffeNet with GPU
    """
    def __init__(self, input_width, input_height, mean_proto, prototxt, caffemodel):
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        super().__init__(input_width, input_height, mean_proto)
        
    def predict(self, img, equalized=False):
        img = super(LocalPredictor, self).transform_img(img, equalized)
        img = img.transpose((2,0,1)) # HWC -> CHW
        img = img.reshape([1] + list(img.shape))
        self.net.blobs['data'].data[...] = img
        
        out = self.net.forward()
        pred_probas = out['prob']
        prob = pred_probas.max()
        label = pred_probas.argmax()
        return label, prob
    
class DevicePredictor(Predictor):
    """
    Predictor using a trained CaffeNet with NCS
    """
    def __init__(self, input_width, input_height, mean_proto, ncs_graph):
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('No NCS devices found')
            exit(1)
        self.device = mvnc.Device(devices[0])
        self.device.OpenDevice()
        with open(ncs_graph, mode='rb') as f:
            blob = f.read()
        self.graph = self.device.AllocateGraph(blob)

        super().__init__(input_width, input_height, mean_proto)
        
    def predict(self, img, equalized=False):
        img = super(DevicePredictor, self).transform_img(img, equalized)
        self.graph.LoadTensor(img.astype(np.float16), 'user object')

        output, userobj = self.graph.GetResult()
        max_order = output.argmax()
        prob = output[max_order]
        label = max_order
        
        return label, prob

    def __del__(self):
        self.graph.DeallocateGraph()
        self.device.CloseDevice()

def debug_array(var, name):
    print('{}: type={}, dtype={}, shape={}'.format(name, type(var), var.dtype, var.shape))

if ARGS.device == 'gpu':
    predictor = LocalPredictor(IMAGE_WIDTH, IMAGE_HEIGHT, ARGS.mean_proto, ARGS.prototxt, ARGS.caffemodel)
elif ARGS.device == 'ncs':
    predictor = DevicePredictor(IMAGE_WIDTH, IMAGE_HEIGHT, ARGS.mean_proto, ARGS.graph)
else:
    print("device must be 'gpu' or 'ncs'",file=sys.stderr)
    exit(2)

if ARGS.action == 'predict':
    print('Reading image paths')
    for img_path in ARGS.image:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
        label, prob = predictor.predict(img, ARGS.equalized)
        class_name = 'cat' if label == 0 else 'dog'
        print('{} [{:.4f}]: {}'.format(class_name, prob, img_path))
    
        cv2.rectangle(img, (0,0), (100,20), (255,255,255), cv2.FILLED)
        cv2.putText(img=img,
                    text='{}: {:.2f}%'.format(class_name, prob*100),
                    org=(0,15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0,0,255),
                    thickness=2)
    
        cv_window_name = os.path.basename(img_path)
        cv2.imshow(cv_window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(cv_window_name)
elif ARGS.action == 'validate':
    n = 0
    tp = 0
    probs = pd.DataFrame(np.zeros((len(ARGS.image), 2)), columns=['image','prob0'])
    for img_path in ARGS.image:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        class_num, prob = predictor.predict(img, ARGS.equalized)
        
        label = 'cat' if class_num == 0 else 'dog'
        filename = os.path.basename(img_path)
        true_label = filename[:3]

        prob0 = prob if class_num == 0 else 1.0 - prob
        probs.iloc[n,:] = [filename, prob0]
        
        n += 1
        if label == true_label:
            tp += 1
            if n % 100 == 0:
                print('Processed {} ...'.format(n))
    if ARGS.prediction_path:
        probs.to_csv(ARGS.prediction_path, index=False)
        print('Probabilities saved to ' + ARGS.prediction_path)
    print('n={}, tp={}, accuracy={:.4f}'.format(tp, n, tp/n))
