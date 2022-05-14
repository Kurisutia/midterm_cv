## Fast RCNN&Yolov3

Fast RCNN&Yolov3的形式相近，都采用如下的方式进行训练、测试。

### 1、数据准备

下载VOC2007数据集；解压放置VOCdevkit文件夹中，然后运行：

```
python voc_annotation.py
```

生成训练和验证的数据集的路径。

### 2、预训练模型准备

可以从网盘链接中下载已经转成Keras格式的模型。

对Yolov3可以从官方中下载对应网络输入设置的模型，后使用convert.py进行转换

```
python convert.py yolov3.cfg yolov3.weights model_data/yolo_weights.h5
```

详见 [keras-yolo3/README.md at master · qqwweee/keras-yolo3 (github.com)](https://github.com/qqwweee/keras-yolo3/blob/master/README.md) 

预训练模型下载好后，放置在对应的model_data文件夹中。

### 3、训练

```
python train.py
```

需要修改相关的训练设置，可以修改train.py中的相关参数4

### 4、检测图片

设置predict函数检测的模式以及相应的文件的路径，然后运行即可。

```
python predict.py
```

### 5、评价

下载网盘中的对应模型，放置在log文件夹内

修改frcnn.py或yolo.py中的模型路径，后运行get_map.py即可。

```
python get_map.py
```



模型网盘地址：

 链接:https://pan.baidu.com/s/1sU3ik_E-yaXOiioOC0iZyw?pwd=643p 
提取码:643p 