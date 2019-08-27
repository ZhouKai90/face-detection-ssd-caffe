# face detection SSD: Single Shot MultiBox Object Detector for face detection

SSD is an unified framework for object detection with a single network.
caffe base on `https://github.com/weiliu89/caffe.git`.Making some small changes for supporting python3.x

### Train the model

This example only covers training on Pascal VOC format dataset. 

- Download the converted pretrained `VGG_ILSVRC_16_layers_fc_reduced.caffemodel` model.
- Compile the `caffe_ssd`,see `https://github.com/weiliu89/caffe.git` for details.
- Download the your Pascal VOC format dataset into `data/`
- modify `tools/create_list.sh`and`tools/create_data.sh` to your own parameters, and run to get the `lmdb`dataset.
- modify `train/ssd_config.py`to your own parameters, run to get the .rec  files for train.
- run `train/ssd_train.sh`for training.
- 

### Demo results

![](https://github.com/ZhouKai90/face-detection-ssd-caffe/blob/master/output/1920x1080_1.jpg)
![](https://github.com/ZhouKai90/face-detection-ssd-caffe/blob/master/output/1920x1080_4.jpg)

### Try the demo

Both python and c++ demo are provided.

```
#for c++
cd demo/cpp
mkdir build
cd build && cmake .. && make && cd ..
./build/demo/target ../../images

#for python
python3 demo/python/ssd_detect_face.py
```
