# face detection SSD: Single Shot MultiBox Object Detector for face detection

SSD is an unified framework for object detection with a single network.

caffe base on `https://github.com/weiliu89/caffe.git`.

### Demo results

![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/test/images/image%20(6)_detection.jpg)
![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/test/images/image%20(7)_detection.jpg)
![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/test/images/image%20(5)_detection.jpg)

### Try the demo

```
mkdir build
cd build && cmake .. && make && cd ..
./build/unit_test/target images
```