import os
import sys
import re
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from mvnc import mvncapi as mvnc
from abc import ABC, abstractmethod

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

MEAN_PROTO = 'model_data/input/mean.binaryproto'
PROTOTXT = 'deploy.prototxt'
CAFFEMODEL = 'model_data/snapshots/caffenet_iter_5000.caffemodel'
NCS_GRAPH = 'graph'

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

    def transform_img(self, img):
        #Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
        #Image Resizing
        img = cv2.resize(img, (self.input_width, self.input_height), interpolation = cv2.INTER_CUBIC)

        img = img.transpose((2,0,1)).astype(np.float32) - self.mean_array
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
        
    def predict(self, img):
        img = super(LocalPredictor, self).transform_img(img)
        self.net.blobs['data'].data[...] = img
        
        out = self.net.forward()
        pred_probas = out['prob']
        prob = pred_probas.max()
        label = 'cat' if pred_probas.argmax() == 0 else 'dog'
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
        
    def predict(self, img):
        img = super(DevicePredictor, self).transform_img(img)
        self.graph.LoadTensor(img.astype(np.float16), 'user object')

        output, userobj = self.graph.GetResult()
        max_order = output.argmax()
        prob = output[max_order]
        label = 'cat' if max_order == 0 else 'dog'
        
        return label, prob

    def __del__(self):
        self.graph.DeallocateGraph()
        self.device.CloseDevice()

def debug_array(var, name):
    print('{}: type={}, dtype={}, shape={}'.format(name, type(var), var.dtype, var.shape))

device= sys.argv[1].lower()
if device == 'gpu':
  predictor = LocalPredictor(IMAGE_WIDTH, IMAGE_HEIGHT, MEAN_PROTO, PROTOTXT, CAFFEMODEL)
elif device == 'ncs':
    predictor = DevicePredictor(IMAGE_WIDTH, IMAGE_HEIGHT, MEAN_PROTO, NCS_GRAPH)
else:
    print("device must be 'gpu' or 'ncs'",file=sys.stderr)
    exit(2)

print('Reading image paths')
for img_path in sys.argv[2:]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    label, prob = predictor.predict(img)
    print('{} [{:.2f}%]: {}'.format(label, prob*100, img_path))

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
    cv2.waitKey(0)
    cv2.destroyWindow(cv_window_name)
