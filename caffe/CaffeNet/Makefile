
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile

RAW_GIT_DIR=https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet
PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT = wget -P . ${RAW_GIT_DIR}/${PROTOTXT_FILENAME}

TRAIN_PROTOTXT_FILENAME= train_val.prototxt
GET_TRAIN_PROTOTXT = wget -P . ${RAW_GIT_DIR}/${TRAIN_PROTOTXT_FILENAME}
TRAIN_LOG_FILENAME=model_data/snapshots/train.log

SOLVER_PROTOTXT_FILENAME=solver.prototxt
MEAN_FILENAME=model_data/input/mean.binaryproto
TRAINING_DATA=model_data/input/train_lmdb
VALIDATION_DATA=model_data/input/validation_lmdb

CAFFEMODEL_FILENAME = bvlc_reference_caffenet.caffemodel
GET_CAFFEMODEL = wget -P . -N http://dl.caffe.berkeleyvision.org/${CAFFEMODEL_FILENAME}
RETRAINED_CAFFEMODEL_FILENAME = model_data/snapshots/caffenet_iter_5000.caffemodel

MODEL_DATA_DIR=model_data
SNAPSHOTS_DIR=model_data/snapshots

.PHONY: all
all: profile compile

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	chmod +x run.py
	@if [ ! -e ${MODEL_DATA_DIR} ]; \
	then \
		mkdir -p ${MODEL_DATA_DIR}; \
	fi
	@if [ ! -e ${SNAPSHOTS_DIR} ]; \
	then \
		mkdir -p ${SNAPSHOTS_DIR}; \
	fi

.PHONY: prototxt
prototxt: prereqs
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "Adding input shape to prototxt file."; \
head -1 ${PROTOTXT_FILENAME} > temp; \
cat input_shape.prototxt >> temp; \
awk ' \
NR<=7 {next} \
/^ *num_output: 1000/ {print "num_output: 2"; next } \
/fc8/ { gsub(/fc8/, "fc8-cats-dogs"); print; next } \
{print}' < ${PROTOTXT_FILENAME} >> temp; \
			mv temp ${PROTOTXT_FILENAME}; \
		else \
			echo "***\nError - Could not download prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

.PHONY: draw
draw: prototxt
	python3 ${CAFFE_HOME}/python/draw_net.py \
	${PROTOTXT_FILENAME} \
	model_data/results/caffenet_deploy.png

.PHONY: train-prototxt
train-prototxt: prereqs
	@echo "\nmaking train-prototxt"
	@if [ -e ${TRAIN_PROTOTXT_FILENAME} ] ; \
	then \
		echo "Training prototxt file already exists"; \
	else \
		echo "Downloading and modifing training prototxt file"; \
		${GET_TRAIN_PROTOTXT}; \
		if [ -e ${TRAIN_PROTOTXT_FILENAME} ] ; \
		then \
			echo "Modifying data paths in training prototxt file."; \
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
{print}' < ${TRAIN_PROTOTXT_FILENAME} > temp; \
mv temp ${TRAIN_PROTOTXT_FILENAME}; \
		else \
			echo "***\nError - Could not download training prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: draw-train
draw: train-prototxt
	python3 ${CAFFE_HOME}/python/draw_net.py \
	${TRAIN_PROTOTXT_FILENAME} \
	model_data/results/caffenet_train.png

.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel file already exists"; \
	else \
		echo "Downloading caffemodel file"; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "***\nError - Could not download caffemodel file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

.PHONY: train
train: caffemodel train-prototxt ${SOLVER_PROTOTXT_FILENAME} ${MEAN_FILENAME} ${TRAIN_DATA} ${VALIDATION_DATA}
	caffe train --solver ${SOLVER_PROTOTXT_FILENAME} --weights ${CAFFEMODEL_FILENAME} 2>&1 | tee ${TRAIN_LOG_FILENAME}

.PHONY: profile
profile: prototxt
	@echo "\nmaking profile"
	${NCPROFILE} ${PROTOTXT_FILENAME} -s 12

.PHONY: browse_profile
browse_profile: profile
	@echo "\nmaking browse_profile"
	@if [ -e output_report.html ] ; \
	then \
		firefox output_report.html & \
	else \
		@echo "***\nError - output_report.html not found" ; \
	fi ; 

.PHONY: compile
compile: prototxt ${RETRAINED_CAFFEMODEL_FILENAME}
	@echo "\nmaking compile"
	${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: predict-device
predict-device: compile
	@echo "\nmaking predictions on device"
	python3 predict-device.py ../../data/images/dogs-and-cats/*.jpg

.PHONY: predict-local
predict-local: predict-local.py ${RETRAINED_CAFFEMODEL_FILENAME}
	@echo "\nmaking predictions locally"
	python3 predict-local.py ../../data/images/dogs-and-cats/*.jpg

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile, run_py";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make train-prototxt - downloads Caffe training prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make train - train the downloaded model for dogs and cats"
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser";
	@echo "  make predict-device - run predict-device.py on example images";
	@echo "  make predict-local - run predict-local.py on example images";
	@echo "  make clean - removes all created content"

clean_caffe_prototxt:
	@echo "\nmaking clean_caffe_prototxt"
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${TRAIN_PROTOTXT_FILENAME}

clean_caffe_model: clean_caffe_prototxt
	@echo "\nmaking clean_caffe_model"
	rm -f ${CAFFEMODEL_FILENAME}

clean: clean_caffe_model
	@echo "\nmaking clean"
	rm -f graph
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f zero_weights.caffemodel
	rm -f output_result.npy
	rm -f output_val.csv
