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
修改train.py 中的config名称，yml中相关参数并运行train.py。
使用时应以.yml为模板编写自己的配置文件进行训练。


## 训练自己的数据：
当使用此工程用于其它分割任务时，你需要准备好训练用的数据集，并在dataset目录下实现加载数据的方法，可参考目录内的'myDataset.py'。  
推荐将数据集整理为以下目录结构：
```shell
    ├── {dataset_dir}
    │   ├── images
    │       ├── 1.tif
    │       ├── 2.tif
    │       ├── ...
    │   ├── labels
    │       ├── 1.png
    │       ├── 2.png
    │       ├── ...
    │   ├── ImageSets
            ├── train.txt
            ├── val.txt
            ├── test.txt
```
'images'目录存放图像，'labels'目录存放标签，并保证图像和对应的标签同名，ImageSets存放划分的数据集的文件名，tools -> split_data.py  

## 注意事项：
图像读写用到opencv（训练时）和gdal，注意opencv读入是按BGR排序的；  
开源模型库支持的模型请参考 https://github.com/qubvel/segmentation_models.pytorch/blob/master/README.md ，通过修改配置文件network_params可调用不同模型；部署到新环境需要联网下载预训练权重；  
