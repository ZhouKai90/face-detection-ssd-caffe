import math
import caffe
from caffe.model_libs import *
import os
import shutil
import stat
import subprocess
from ssd_config import config as CF
from VGG16_half_channels_512 import getSymbol

def create_train_net():
    # Create train net.
    # create net parameters
    batch_sampler = [
        {
            'sampler': {
            },
            'max_trials': 1,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': CF.minAspectRatio,
                'max_aspect_ratio': CF.maxAspectRatio,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.1,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': CF.minAspectRatio,
                'max_aspect_ratio': CF.maxAspectRatio,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.3,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': CF.minAspectRatio,
                'max_aspect_ratio': CF.maxAspectRatio,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.5,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': CF.minAspectRatio,
                'max_aspect_ratio': CF.maxAspectRatio,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.7,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': CF.minAspectRatio,
                'max_aspect_ratio': CF.maxAspectRatio,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.9,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': CF.minAspectRatio,
                'max_aspect_ratio': CF.maxAspectRatio,
            },
            'sample_constraint': {
                'max_jaccard_overlap': 1.0,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
    ]
    train_transform_param = {
        'mirror': True,                 #镜像
        'mean_value': [104, 117, 123],  #均值
        'resize_param': {               #缩放
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            'height': CF.resizeHeight,
            'width': CF.resizeWidth,
            'interp_mode': [
                P.Resize.LINEAR,
                P.Resize.AREA,
                P.Resize.NEAREST,
                P.Resize.CUBIC,
                P.Resize.LANCZOS4,
            ],
        },
        'distort_param': {
            'brightness_prob': 0.5,     #调整亮度的概率
            'brightness_delta': 32,     #调增像素值的范围．对原图增加(-32, 32)中的随机像素
            'contrast_prob': 0.5,       #调整对比度的概率
            'contrast_lower': 0.5,      #随机对比因子的下限
            'contrast_upper': 1.5,      #随机对比因子的上限
            'hue_prob': 0.5,            #调整色调的概率
            'hue_delta': 18,            #调整色调通道数的概率
            'saturation_prob': 0.5,     #调整饱和度的概率
            'saturation_lower': 0.5,    #调整饱和因子的上限
            'saturation_upper': 1.5,    #调整饱和因子的下限
            'random_order_prob': 0.0,   #随机排列图像通道的概率
        },
        'expand_param': {
            'prob': 0.5,                #expand发生的概率
            'max_expand_ratio': 2.0,    #expand扩大的倍数
            # 'max_expand_ratio': 4.0,    #expand扩大的倍数
        },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
        }
    }
    multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': CF.LOSS_PARA.conf_loss_type,
        'loc_weight': CF.LOSS_PARA.loc_weight,
        'num_classes': CF.numClasses,
        'share_location': CF.LOSS_PARA.share_location,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': CF.LOSS_PARA.background_label_id,
        'use_difficult_gt': CF.LOSS_PARA.train_on_diff_gt,
        'mining_type': CF.LOSS_PARA.mining_type,
        'neg_pos_ratio': CF.LOSS_PARA.neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': CF.LOSS_PARA.code_type,
        'ignore_cross_boundary_bbox': CF.LOSS_PARA.ignore_cross_boundary_bbox,
    }

    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(CF.trainData, batch_size=CF.batch_size_per_device,
            train=True, output_label=True, label_map_file=CF.label_map_file,
            transform_param=train_transform_param, batch_sampler=batch_sampler)

    mbox_layers = getSymbol(net)
    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    loss_param = {
        'normalization': CF.normalization_mode,
    }

    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                               loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                               propagate_down=[True, True, False, False])

    with open(CF.train_net_file, 'w') as f:
        print('name: "{}_train"'.format(CF.model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(CF.train_net_file, CF.job_dir)

def create_test_net():
    # Create test net.
    test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': CF.resizeHeight,
                'width': CF.resizeWidth,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
    # parameters for generating detection output.
    det_out_param = {
        'num_classes': CF.numClasses,
        'share_location': CF.LOSS_PARA.share_location,
        'background_label_id': CF.LOSS_PARA.background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'save_output_param': {
            'output_directory': CF.output_result_dir,
            'output_name_prefix': "comp4_det_test_",
            'output_format': "VOC",
            'label_map_file': CF.label_map_file,
            'name_size_file': CF.name_size_file,
            'num_test_image': CF.numTestImg,
        },
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': CF.LOSS_PARA.code_type,
    }
    # parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': CF.numClasses,
        'background_label_id': CF.LOSS_PARA.background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        'name_size_file': CF.name_size_file,
    }

    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(CF.testData, batch_size=CF.testBatchSize,
            train=False, output_label=True, label_map_file=CF.label_map_file,
            transform_param=test_transform_param)

    mbox_layers = getSymbol(net)

    conf_name = "mbox_conf"
    if CF.LOSS_PARA.conf_loss_type == P.MultiBoxLoss.SOFTMAX:
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, CF.numClasses]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif CF.LOSS_PARA.conf_loss_type == P.MultiBoxLoss.LOGISTIC:
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(CF.test_net_file, 'w') as f:
        print('name: "{}_test"'.format(CF.model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(CF.test_net_file, CF.job_dir)
    create_deploy_net(net)

def create_deploy_net(net):
    # Create deploy net.
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(CF.deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(CF.model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, CF.resizeHeight, CF.resizeWidth])])
        print(net_param, file=f)
    shutil.copy(CF.deploy_net_file, CF.job_dir)

def create_solver():
    # Create solver.
    # Use different initial learning rate.
    if CF.useBatchnorm:
        base_lr = CF.baseLr
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = CF.baseLr * 0.1

    if CF.normalization_mode == P.Loss.NONE:
        base_lr /= CF.batch_size_per_device
    elif CF.normalization_mode == P.Loss.VALID:
        base_lr *= 25. / CF.LOSS_PARA.loc_weight
    elif CF.normalization_mode == P.Loss.FULL:
        # Roughly there are 2000 prior bboxes per image.
        # TODO(weiliu89): Estimate the exact # of priors.
        base_lr *= 2000.

    test_iter = int(math.ceil(float(CF.numTestImg) / CF.testBatchSize))

    # Solver parameters.
    solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': CF.stepValue,
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': CF.iter_size,
    'max_iter': CF.maxIter,
    'snapshot': CF.snapshot,
    'display': CF.display,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': CF.solver_mode,
    'device_id': CF.device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': CF.testInterval,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }
    solver = caffe_pb2.SolverParameter(
            train_net=CF.train_net_file,
            test_net=[CF.test_net_file],
            snapshot_prefix=CF.snapshot_prefix,
            **solver_param)

    with open(CF.solver_file, 'w') as f:
        print(solver, file=f)
    shutil.copy(CF.solver_file, CF.job_dir)

    max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(CF.snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(CF.model_name))[1])
        if iter > max_iter:
          max_iter = iter

    train_src_param = None
    if CF.preTrainModel is not None:
        train_src_param = '--weights="{}" \\\n'.format(CF.preTrainModel)
    if CF.resumeTraining:
      if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(CF.snapshot_prefix, max_iter)

    if CF.removeOldModels:
      # Remove any snapshots smaller than max_iter.
      for file in os.listdir(CF.snapshot_dir):
        if file.endswith(".solverstate"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(CF.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(CF.snapshot_dir, file))
        if file.endswith(".caffemodel"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(CF.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(CF.snapshot_dir, file))

    # Create job file.
    with open(CF.job_file, 'w') as f:
      f.write('./caffe_ssd/build/tools/caffe train \\\n')
      f.write('--solver="{}" \\\n'.format(CF.solver_file))
      if train_src_param is not None:
        f.write(train_src_param)
      if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(CF.gpus, CF.job_dir, CF.model_name))
      else:
        f.write('2>&1 | tee {}/{}.log\n'.format(CF.job_dir, CF.model_name))

if __name__ == '__main__':
    create_train_net()
    create_test_net()
    create_solver()
    # Copy the python script to CF.job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, CF.job_dir)

    # # Run the job.
    os.chmod(CF.job_file, stat.S_IRWXU)
    if CF.runSoon:
      subprocess.call(CF.job_file, shell=True)

