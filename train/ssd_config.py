# -*- coding: UTF-8 -*-
from easydict import EasyDict as edict
import math
from caffe.model_libs import *

config = edict()

config.modelPrefix = 'VGG16'
config.savePath = 'models/half_channels_VGG16_512'
# The database file for training and test data. Created by tools/create_data.sh
config.trainData = 'data/widerface/widerface_trainval_lmdb'
config.testData = 'data/widerface/widerface_test_lmdb'
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
config.preTrainModel = None
# config.preTrainModel = 'models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel'
config.dataPath = 'data/widerface'

# Specify the batch sampler.
config.resizeHeight = 512
config.resizeWidth = 512

# minimum dimension of input image
config.batchSize = 32
config.minDim = 512
config.minAspectRatio = 0.7
config.maxAspectRatio = 1.5
config.numClasses = 2
config.testBatchSize = 16
config.numTestImg = 930
#params for learn rate
config.baseLr = 0.0004     # *2.5
config.lrPolicy = 'multistep'
config.lrGamma = 0.8
config.stepValue = [30000, 40000, 45000, 50000, 550000, 60000]

config.maxIter = 60000
config.snapshot = 1000
config.testInterval = 200

config.display = 20

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

config.normalization_mode = P.Loss.VALID

default = edict()
# Modify the job name if you want.
default.resize = "{}x{}".format(config.resizeWidth, config.resizeHeight)
default.job_name = "SSD_{}".format(default.resize)
# Directory which stores the model .prototxt file.
default.save_dir = "{}/{}".format(config.savePath, default.job_name)

# The name of the model. Modify it if you want.
config.model_name = "{}_{}".format(config.modelPrefix, default.job_name)
# Directory which stores the snapshot of models.
config.snapshot_dir = "{}/{}".format(config.savePath, default.job_name)
# Directory which stores the job script and log file.
config.snapshot_dir = "{}/job/{}".format(config.savePath, default.job_name)
# Directory which stores the job script and log file.
config.job_dir = "{}/job/{}".format(config.savePath, default.job_name)
# Directory which stores the detection results.
config.output_result_dir = "{}/results/Main".format(config.savePath)
# model definition files.
config.train_net_file = "{}/train.prototxt".format(default.save_dir)
config.test_net_file = "{}/test.prototxt".format(default.save_dir)
config.deploy_net_file = "{}/deploy.prototxt".format(default.save_dir)
config.solver_file = "{}/solver.prototxt".format(default.save_dir)
# snapshot prefix.
config.snapshot_prefix = "{}/{}".format(config.snapshot_dir, config.model_name)
# job script path.
config.job_file = "{}/{}.sh".format(config.snapshot_dir, config.model_name)
# Stores the test image names and sizes. Created by tools/create_list.sh
config.name_size_file = "{}/test_name_size.txt".format(config.dataPath)
# Stores LabelMapItem.
config.label_map_file = "{}/labelmap_voc.prototxt".format(config.dataPath)

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(config.trainData)
check_if_exist(config.testData)
check_if_exist(config.label_map_file)
make_if_not_exist(default.save_dir)
make_if_not_exist(config.snapshot_dir)
make_if_not_exist(config.snapshot_dir)

config.LOSS_PARA = edict()
# MultiBoxLoss parameters.
config.LOSS_PARA.share_location = True
config.LOSS_PARA.background_label_id = 0
config.LOSS_PARA.train_on_diff_gt = True
config.LOSS_PARA.code_type = P.PriorBox.CENTER_SIZE
config.LOSS_PARA.ignore_cross_boundary_bbox = False
config.LOSS_PARA.mining_type = P.MultiBoxLoss.MAX_NEGATIVE
config.LOSS_PARA.neg_pos_ratio = 3.
config.LOSS_PARA.loc_weight = (config.LOSS_PARA.neg_pos_ratio + 1.) / 4.
config.LOSS_PARA.conf_loss_type = P.MultiBoxLoss.SOFTMAX

# Divide the mini-batch to different GPUs.
config.accum_batch_size = 32
config.iter_size = config.accum_batch_size / config.batchSize
config.solver_mode = P.Solver.CPU
config.device_id = 0
config.batch_size_per_device = config.batchSize

# Defining which GPUs to use.
config.gpus = "0,1"
gpulist = config.gpus.split(",")
num_gpus = len(gpulist)
if num_gpus > 0:
  config.batch_size_per_device = int(math.ceil(float(config.batchSize) / num_gpus))
  config.iter_size = int(math.ceil(float(config.accum_batch_size) / (config.batch_size_per_device * num_gpus)))
  config.solver_mode = P.Solver.GPU
  config.device_id = int(gpulist[0])
