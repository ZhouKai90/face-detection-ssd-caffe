# -*- coding: UTF-8 -*-
from easydict import EasyDict as edict
import os
import math
from caffe.model_libs import *

config = edict()

# The database file for training and test data. Created by tools/create_data.sh
# config.trainData = 'data/widerface/widerface_trainval_lmdb'
config.trainData = 'data/lmdb/train_clean_devkit_train_lmdb'
# config.testData = 'data/widerface/widerface_test_lnval_lmdb'
config.testData = 'data/lmdb/train_clean_devkit_test_lmdb'
config.modelPrefix = 'VGG16'
config.savePath = 'models/widerface'
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
config.preTrainModel = 'models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel'
config.dataPath = 'data/widerface'

config.gpus = "0,1"
# Specify the batch sampler.
config.resizeHeight = 300
config.resizeWidth = 300
# minimum dimension of input image
config.minDim = 300
config.minAspectRatio = 1
config.maxAspectRatio = 1
config.numClasses = 2
config.batchSize = 64
config.testBatchSize = 8
config.numTestImg = 930
config.baseLr = 0.0004
config.maxIter = 60000
config.snapshot = 200
config.testInterval = 100
config.display = 10

# Set true if you want to start training right after generating all files.
config.runSoon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
config.resumeTraining = True
# If true, Remove old model files.
config.removeOldModels = False

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
config.useBatchnorm = False
config.lrMult = 1

# Modify the job name if you want.
resize = "{}x{}".format(config.resizeWidth, config.resizeHeight)
job_name = "SSD_{}".format(resize)
# The name of the model. Modify it if you want.
model_name = "{}_{}".format(config.modelPrefix, job_name)

# Directory which stores the model .prototxt file.
save_dir = "{}/{}".format(config.savePath, job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "{}/{}".format(config.savePath, job_name)
# Directory which stores the job script and log file.
job_dir = "{}/job/{}".format(config.savePath, job_name)
# Directory which stores the detection results.
output_result_dir = "{}/results/Main".format(config.savePath)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Stores the test image names and sizes. Created by tools/create_list.sh
name_size_file = "{}/test_name_size.txt".format(config.dataPath)

# Stores LabelMapItem.
label_map_file = "{}/labelmap_voc.prototxt".format(config.dataPath)

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(config.trainData)
check_if_exist(config.testData)
check_if_exist(label_map_file)
check_if_exist(config.preTrainModel)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

normalization_mode = P.Loss.VALID

# MultiBoxLoss parameters.
share_location = True
background_label_id = 0
train_on_diff_gt = True
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': config.numClasses,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
}

# Defining which GPUs to use.
gpulist = config.gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
accum_batch_size = 32
iter_size = accum_batch_size / config.batchSize
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = config.batchSize
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(config.batchSize) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

