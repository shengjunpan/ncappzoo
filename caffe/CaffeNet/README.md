```bash
if [ ! -e model_data/snapshots ]; then
    mkdir -p model_data/snapshots
fi

# Download and modify training prototxt
wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/train_val.prototxt -O train_val.prototxt.downloaded && \
awk ' \
/^ *num_output: 1000/ {print "num_output: 2"; next } \
/fc8/ { gsub(/fc8/, "fc8-cats-dogs"); print; next } \
/^ *batch_size: [0-9]+/ {print "batch_size: 64"; next} \
/^ *source: "examples\/imagenet\/ilsvrc12_train_lmdb"/ { \
  print "source: \"model_data/input/train_lmdb\""; \
  next \
} \
/^ *source: "examples\/imagenet\/ilsvrc12_val_lmdb"/ { \
  print "source: \"model_data/input/validation_lmdb\""; \
  next \
} \
/^ *mean_file: "data\/ilsvrc12\/imagenet_mean.binaryproto"/ { \
  print "mean_file: \"model_data/input/mean.binaryproto\""; \
  next \
} \
{print}' < train_val.prototxt.downloaded > train_val.prototxt && \
python3 ${CAFFE_HOME}/python/draw_net.py train_val.prototxt model_data/results/caffenet_train.png


# Download and modify deploy prototxt
wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt -O deploy.prototxt.downloaded && \
head -1 deploy.prototxt.downloaded > deploy.prototxt && \
cat input_shape.prototxt >> deploy.prototxt && \
awk ' \
NR<=7 {next} \
/^ *num_output: 1000/ {print "num_output: 2"; next } \
/fc8/ { gsub(/fc8/, "fc8-cats-dogs"); print; next } \
{print}' < deploy.prototxt.downloaded >> deploy.prototxt && \
python3 ${CAFFE_HOME}/python/draw_net.py deploy.prototxt model_data/results/caffenet_deploy.png

# Download initial weights
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

# Train a new model
caffe train \
      --solver solver.prototxt \
      --weights bvlc_reference_caffenet.caffemodel \
      2>&1 | tee model_data/snapshots/train.log

python3 plot_learning_curve.py model_data/snapshots/train.log model_data/results/learning_curve.png

# Profile deploy prototxt
mvNCProfile deploy.prototxt -w model_data/snapshots/caffenet_iter_5000.caffemodel -s 12

# Compile trained model into 'graph'
mvNCCompile deploy.prototxt -w model_data/snapshots/caffenet_iter_5000.caffemodel -s 12 

# Make predictions using trained model
python3 run.py --device=gpu --action=predict ../../data/images/dogs-and-cats/*.jpg
python3 run.py --device=ncs --action=predict ../../data/images/dogs-and-cats/*.jpg

python3 run.py --device=gpu --action=validate model_data/input/validation_lmdb
python3 run.py --device=ncs --action=validate model_data/input/validation_lmdb


##################################
if [ ! -e model_data/snapshots_0 ]; then
    mkdir -p model_data/snapshots_0
fi

# Train a new model
caffe train \
      --solver solver_0.prototxt \
      2>&1 | tee model_data/snapshots_0/train.log

python3 plot_learning_curve.py model_data/snapshots_0/train.log model_data/results/learning_curve_0.png

mvNCCompile deploy.prototxt -w model_data/snapshots_0/caffenet_iter_5000.caffemodel -s 12 -o graph_0

python3 run.py --device=gpu --action=predict --caffemodel=model_data/snapshots_0/caffenet_iter_5000.caffemodel ../../data/images/dogs-and-cats/*.jpg
python3 run.py --device=ncs --action=predict --graph=graph_0 ../../data/images/dogs-and-cats/*.jpg

python3 run.py --device=gpu --action=validate --caffemodel=model_data/snapshots_0/caffenet_iter_5000.caffemodel model_data/input/validation_lmdb
python3 run.py --device=ncs --action=validate --graph=graph_0 model_data/input/validation_lmdb
```
