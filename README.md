## 特性：
- 使用.yml文件编写配置文件
- 使用开源模型库[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)或自定义模型
- 针对大尺寸遥感影像的推理、精度评估  
- 处理大尺度遥感影像栅格及矢量数据的常用工具（基于GDAL）

## 环境：
- torch>1.6
- torchsummay
- segmentation_models_pytorch  
    `pip install git+https://github.com/qubvel/segmentation_models.pytorch` (或者直接下载工程然后python setup.py install)
- albumentations
- gdal
- opencv
- matplotlib
- scipy
- scikit-learn
- scikit-image
- tqdm
- pandas

## 运行：
#### 预测
```shell
参考 test.sh
```
or  
```shell
    python main.py -c test -m infer -g 0
```
#### 训练
```shell
    python main.py -c test -m train -g 0,1
```
将会在configs目录中加载test.yml配置文件，并在runs目录内新建test目录存放训练数据。
使用时应以test.yml为模板编写自己的配置文件进行训练。


## 训练自己的数据：
当使用此工程用于其它分割任务时，你需要准备好训练用的数据集，并在dataset目录下实现加载数据的方法，可参考目录内的'myDataset.py'。  
推荐将数据集整理为以下目录结构：
```shell
    ├── {dataset_dir}
    │   ├── train
    │       ├── images
    │       ├── labels
    │   ├── val
    │       ├── images
    │       ├── labels
```
'images'目录存放图像，'labels'目录存放标签，并保证图像和对应的标签同名。  

## TODO：
- TTA
- 混合精度训练
- DDP分布式训练

## 注意事项：
图像读写用到opencv（训练时）和gdal，注意opencv读入是按BGR排序的；  
开源模型库支持的模型请参考 https://github.com/qubvel/segmentation_models.pytorch/blob/master/README.md ，通过修改配置文件network_params可调用不同模型；部署到新环境需要联网下载预训练权重；  
