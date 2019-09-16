import os
import caffe
from caffe.model_libs import *
from caffe import layers as L
from caffe import params as P
from caffe.model_libs import ConvBNLayer

def getMobileNetV1(net, use_batchnorm=True, use_relu=True, lr_mult=1):
    from_layer = net.keys()[0]
    print('from_layers:%s' % net.keys())
    out_layer = "conv1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 32, 3, 1, 2, lr_mult=lr_mult)  #1/2

    from_layer = out_layer
    out_layer = "conv2_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 32, 3, 1, 1, lr_mult=lr_mult)  #1/2
    from_layer = out_layer
    out_layer = "conv2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 1, 0, 1, lr_mult=lr_mult)  #1/2

    from_layer = out_layer
    out_layer = "conv3_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 3, 1, 2, lr_mult=lr_mult)  #1/4
    from_layer = out_layer
    out_layer = "conv3"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1, lr_mult=lr_mult)  #1/4
    from_layer = out_layer
    out_layer = "conv4_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1, lr_mult=lr_mult)  #1/4
    from_layer = out_layer
    out_layer = "conv4"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1, lr_mult=lr_mult)  #1/4

    from_layer = out_layer
    out_layer = "conv5_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 2, lr_mult=lr_mult)  #1/8
    from_layer = out_layer
    out_layer = "conv5"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1, lr_mult=lr_mult)  #1/8
    from_layer = out_layer
    out_layer = "conv6_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1, lr_mult=lr_mult)  #1/8
    from_layer = out_layer
    out_layer = "conv6"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1, lr_mult=lr_mult)  #1/8

    from_layer = out_layer
    out_layer = "conv7_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv7"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv8_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv8"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/16

    from_layer = out_layer
    out_layer = "conv9_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv9"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv10_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv10"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/16

    from_layer = out_layer
    out_layer = "conv11_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv11"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv12_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1, lr_mult=lr_mult)  #1/16
    from_layer = out_layer
    out_layer = "conv12"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/16

    from_layer = out_layer
    out_layer = "conv13_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2, lr_mult=lr_mult)  #1/32
    from_layer = out_layer
    out_layer = "conv13"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/32
    from_layer = out_layer
    out_layer = "conv14_dw"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1, lr_mult=lr_mult)  #1/32
    from_layer = out_layer
    out_layer = "conv14"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1, lr_mult=lr_mult)  #1/32
    return net

if __name__ == '__main__':
    net = caffe.NetSpec()
    net.data, net.label = L.Data(source='data', backend=P.Data.LMDB, batch_size=32, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))

    getMobileNetV1(net)
    net.pool1 = L.Pooling(from_layer=net.keys()[-1], pool=P.Pooling.AVE, kernel_size=7, stride=1)
    net.flatten = L.Flatten(net.pool1, axis=1)

    net.fc = L.InnerProduct(net.flatten, num_output=1000)
    test_file = os.path.join(os.getcwd(), 'test.prototxt')
    with open(test_file, 'w') as f:
        print('name: "{}_train"'.format('test'), file=f)
        print(net.to_proto(), file=f)
