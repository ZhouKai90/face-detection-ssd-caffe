from __future__ import print_function
import caffe
from caffe.model_libs import *
from ssd_config import config as cf
from ssd_config import *
from ssd_symbol import AddExtraLayers
import math
import os
import shutil
import stat
import subprocess
from ssd_net_header import addSSDHeaderLayer

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
                'min_aspect_ratio': cf.minAspectRatio,
                'max_aspect_ratio': cf.maxAspectRatio,
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
                'min_aspect_ratio': cf.minAspectRatio,
                'max_aspect_ratio': cf.maxAspectRatio,
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
                'min_aspect_ratio': cf.minAspectRatio,
                'max_aspect_ratio': cf.maxAspectRatio,
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
                'min_aspect_ratio': cf.minAspectRatio,
                'max_aspect_ratio': cf.maxAspectRatio,
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
                'min_aspect_ratio': cf.minAspectRatio,
                'max_aspect_ratio': cf.maxAspectRatio,
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
                'min_aspect_ratio': cf.minAspectRatio,
                'max_aspect_ratio': cf.maxAspectRatio,
            },
            'sample_constraint': {
                'max_jaccard_overlap': 1.0,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
    ]
    train_transform_param = {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            'height': cf.resizeHeight,
            'width': cf.resizeWidth,
            'interp_mode': [
                P.Resize.LINEAR,
                P.Resize.AREA,
                P.Resize.NEAREST,
                P.Resize.CUBIC,
                P.Resize.LANCZOS4,
            ],
        },
        'distort_param': {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        },
        'expand_param': {
            'prob': 0.5,
            'max_expand_ratio': 4.0,
        },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
        }
    }
    loss_param = {
        'normalization': normalization_mode,
    }

    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(cf.trainData, batch_size=batch_size_per_device,
            train=True, output_label=True, label_map_file=label_map_file,
            transform_param=train_transform_param, batch_sampler=batch_sampler)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False)

    AddExtraLayers(net, cf.useBatchnorm, lr_mult=cf.lrMult)

    mbox_layers = addSSDHeaderLayer(net)

    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                               loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                               propagate_down=[True, True, False, False])

    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(train_net_file, job_dir)

def create_test_net():
    # Create test net.
    test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': cf.resizeHeight,
                'width': cf.resizeWidth,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
    # parameters for generating detection output.
    det_out_param = {
        'num_classes': config.numClasses,
        'share_location': share_location,
        'background_label_id': background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'save_output_param': {
            'output_directory': output_result_dir,
            'output_name_prefix': "comp4_det_test_",
            'output_format': "VOC",
            'label_map_file': label_map_file,
            'name_size_file': name_size_file,
            'num_test_image': config.numTestImg,
        },
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': code_type,
    }
    # parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': config.numClasses,
        'background_label_id': background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        'name_size_file': name_size_file,
    }

    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(cf.testData, batch_size=cf.testBatchSize,
            train=False, output_label=True, label_map_file=label_map_file,
            transform_param=test_transform_param)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False)

    AddExtraLayers(net, cf.useBatchnorm, lr_mult=cf.lrMult)

    mbox_layers = addSSDHeaderLayer(net)

    conf_name = "mbox_conf"
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, cf.numClasses]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(test_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(test_net_file, job_dir)
    create_deploy_net(net)

def create_deploy_net(net):
    # Create deploy net.
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, cf.resizeHeight, cf.resizeWidth])])
        print(net_param, file=f)
    shutil.copy(deploy_net_file, job_dir)

def create_solver():
    # Create solver.
    # Use different initial learning rate.
    if config.useBatchnorm:
        base_lr = config.baseLr
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = config.baseLr * 0.1

    if normalization_mode == P.Loss.NONE:
        base_lr /= batch_size_per_device
    elif normalization_mode == P.Loss.VALID:
        base_lr *= 25. / loc_weight
    elif normalization_mode == P.Loss.FULL:
        # Roughly there are 2000 prior bboxes per image.
        # TODO(weiliu89): Estimate the exact # of priors.
        base_lr *= 2000.

    test_iter = int(math.ceil(float(config.numTestImg) / config.testBatchSize))

    # Solver parameters.
    solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [8000, 10000, 12000, 20000, 40000, 60000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': cf.maxIter,
    'snapshot': cf.snapshot,
    'display': cf.display,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': cf.testInterval,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }
    solver = caffe_pb2.SolverParameter(
            train_net=train_net_file,
            test_net=[test_net_file],
            snapshot_prefix=snapshot_prefix,
            **solver_param)

    with open(solver_file, 'w') as f:
        print(solver, file=f)
    shutil.copy(solver_file, job_dir)

    max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if iter > max_iter:
          max_iter = iter

    train_src_param = '--weights="{}" \\\n'.format(cf.preTrainModel)
    if cf.resumeTraining:
      if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

    if cf.removeOldModels:
      # Remove any snapshots smaller than max_iter.
      for file in os.listdir(snapshot_dir):
        if file.endswith(".solverstate"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(snapshot_dir, file))
        if file.endswith(".caffemodel"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(snapshot_dir, file))

    # Create job file.
    with open(job_file, 'w') as f:
      f.write('./caffe_ssd/build/tools/caffe train \\\n')
      f.write('--solver="{}" \\\n'.format(solver_file))
      f.write(train_src_param)
      if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(cf.gpus, job_dir, model_name))
      else:
        f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

if __name__ == '__main__':
    create_train_net()
    create_test_net()
    create_solver()
    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, job_dir)

    # Run the job.
    os.chmod(job_file, stat.S_IRWXU)
    if cf.runSoon:
      subprocess.call(job_file, shell=True)

