import os
import cv2
import lmdb
import numpy as np
import argparse

from caffe.proto import caffe_pb2

_parser = argparse.ArgumentParser()
_parser.add_argument("-o", "--output-parent-dir",
                     default='/tmp',
                     help="output parent directory. The lmdb name will be used as subfolder name.")
_parser.add_argument("-f", "--formatter",
                     default='{}.{:05d}.jpg',
                     help="formatter used for filenames, with two parameters label and number.")
_parser.add_argument("lmdb_path", nargs="+", help="image lmdb(s)")
ARGS = _parser.parse_args()

if __name__ == '__main__':
    for lmdb_path in ARGS.lmdb_path:
        output_dir = os.path.join(ARGS.output_parent_dir, os.path.basename(lmdb_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print('Extracting', lmdb_path)

        ncats = 0
        ndogs = 0
        env = lmdb.open(lmdb_path, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                datum = caffe_pb2.Datum()
                datum.ParseFromString(value)
                img = np.fromstring(datum.data, dtype=np.uint8)
                input_shape = (datum.channels, datum.height, datum.width)
                img = img.reshape(input_shape)
                img = img.transpose((1,2,0))
                if datum.label == 0:
                    ncats += 1
                    filename = ARGS.formatter.format('cat', ncats)
                else:
                    ndogs += 1
                    filename = ARGS.formatter.format('dog', ndogs)
                cv2.imwrite(os.path.join(output_dir, filename), img)
        env.close()
