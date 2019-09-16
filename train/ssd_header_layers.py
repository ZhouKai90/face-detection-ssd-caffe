from ssd_config import config as cf
from ssd_config import *

def addSSDHeaderLayer(net):
    # parameters for generating priors.
    # conv4_3 ==> 38 x 38
    # fc7 ==> 19 x 19
    # conv6_2 ==> 10 x 10
    # conv7_2 ==> 5 x 5
    # conv8_2 ==> 3 x 3
    # conv9_2 ==> 1 x 1
    # mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    mbox_source_layers = ['relu4_3', 'relu7', 'conv6_2_relu', 'conv7_2_relu', 'conv8_2_relu', 'conv9_2_relu']

    # in percent %
    min_ratio = 20
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
      min_sizes.append(cf.minDim * ratio / 100.)
      # max_sizes.append(cf.minDim * (ratio + step) / 100.)
    min_sizes = [cf.minDim * 10 / 100.] + min_sizes
    # max_sizes = [cf.minDim * 20 / 100.] + max_sizes
    steps = [8, 16, 32, 64, 100, 300]
    aspect_ratios = [[1], [1], [1], [1], [1], [1]]
    # aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1, -1]
    # variance used to encode/decode prior bboxes.
    if code_type == P.PriorBox.CENTER_SIZE:
      prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
      prior_variance = [0.1]

    flip = False
    clip = False

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=cf.useBatchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
            num_classes=cf.numClasses, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=cf.lrMult)
    return mbox_layers