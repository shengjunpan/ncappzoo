import os
import sys
import re
import glob
import cv2
import caffe
import lmdb
import numpy as np
import argparse

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
        self.mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(input_shape)

        super().__init__()

    def transform_img(self, img, preprocessed):
        if not preprocessed:
            #Histogram Equalization
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
            #Image Resizing
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation = cv2.INTER_CUBIC)
            img = img.transpose((2,0,1)) # HWC -> CHW

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
        
    def predict(self, img, preprocessed=False):
        img = super(LocalPredictor, self).transform_img(img, preprocessed)
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
        
    def predict(self, img, preprocessed=False):
        img = super(DevicePredictor, self).transform_img(img, preprocessed)
        img = img.transpose((1,0,2)) # CHW -> HWC (for NCS)
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
    
        label, prob = predictor.predict(img)
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
    probs = []
    for validation_lmdb in ARGS.image:
        print('Validating', validation_lmdb)
        env = lmdb.open(validation_lmdb, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                n += 1
                datum = caffe_pb2.Datum()
                datum.ParseFromString(value)
                input_shape = (datum.channels, datum.height, datum.width)
                img = np.fromstring(datum.data, dtype=np.uint8).reshape(input_shape)
                label, prob = predictor.predict(img, preprocessed=True)
                probs.append(prob if label == 0 else 1.0 - prob)
                
                if label == datum.label:
                    tp += 1
                if n % 100 == 0:
                    print('Processed {} ...'.format(n))
        env.close()
    if ARGS.prediction_path:
        np.savetxt(ARGS.prediction_path, probs)
        print('Probabilities saved to ' + ARGS.prediction_path)
    print('n={}, tp={}, accuracy={:.4f}, avg(p)={:.4f}'.format(tp, n, tp/n, sum(probs)/n))
